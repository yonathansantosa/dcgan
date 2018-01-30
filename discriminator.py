import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as vutils
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import argparse
import confusionmat as cnf
from sklearn.metrics import confusion_matrix
# from itertools import chain
from model import *

parser = argparse.ArgumentParser(description='discriminator')
parser.add_argument('--batch', default=512, 
    help='choose the batch size')
parser.add_argument('--split', default=0.8, 
    help='how much dataset you want to use as training split')
parser.add_argument('--gpu', default=0, 
    help='which gpu would you use')
parser.add_argument('--lr', default=0.01, 
    help='learning rate')
parser.add_argument('--epoch', default=10, 
    help='learning rate')
parser.add_argument('--test', default=False, action='store_true', 
    help='load pretrained model and run test')
args = parser.parse_args()

''' 
Fixed Parameters

ndf = number of feature in discriminator
nc = number of input channel
kernel_d = kernel size for each layer
img_size = image size
'''
ndf = 128
nc = 3 
kernel_d = [4,4,4,4,4]
img_size = 64

batch_size = int(args.batch)
learning_rate = float(args.lr)

feature_size = 28672
# traindir = './food-101/images'
traindir = '/home/data/food-101/images'
gpu_id = int(args.gpu)
torch.cuda.set_device(gpu_id)


class SVMclassifier(nn.Module):
    def __init__(self, feature_size, cls):
        super(SVMclassifier, self).__init__()
        self.fc = nn.Linear(feature_size, cls)

    def forward(self,input):
        return self.fc(input)

def transform_to_target(y):
    '''
    Target is defined as sparse vector
    where label indicated as 1 and the rest
    indicated as -1

    similar with one hot vector but the rest is
    -1 instead of 0
    '''
    yy = y.numpy()
    size = np.arange(yy.shape[0])
    target = np.ones((y.size(0), 101))
    target *= -1
    target[size, yy] = 1
    target = torch.FloatTensor(target)
    
    return target

def transform_to_list(y):
    '''
    Transforming tensor to list
    '''
    yy = y.cpu().numpy()
    yy = np.argmax(yy, 1)
    yy = yy.tolist()
    return yy
        
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02).cuda(gpu_id)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02).cuda(gpu_id)
#         m.bias.data.fill_(0).cuda(gpu_id)

def load_data(args):
    transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.ImageFolder(traindir, transform=transform)
    
    # Splitting train and test data
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(float(args.split) * num_train))

    np.random.shuffle(indices)

    train_idx = indices[:10]
    test_idx = indices[10:20]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idx)

    # Converting to loader for batch training and testing
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader

def train(args, train_loader):
    netD = D_layer(nc,ndf,kernel_d)
    netD.load_state_dict(torch.load("trained_model/netD_imagenet.pth", map_location=lambda storage, loc: storage))
    netD.cuda(gpu_id)
    netD.eval()

    svm = SVMclassifier(28672, 101)
    svm.cuda(gpu_id)

    optimizer = optim.SGD(svm.parameters(), learning_rate, 0.01, 0, 1e-10, True)
    f = open('graph/%s_graph_svm.txt'%args.epoch, 'w')
    f.write('epoch,iteration,loss\n')
    max_epoch = int(args.epoch)
    for epoch in range(max_epoch):
        for i, (x, y) in enumerate(train_loader):
            img_real = Variable(x).cuda(gpu_id)
            label = transform_to_target(y)
            target = Variable(label).cuda(gpu_id)

            feature = netD.generate_feature(img_real)

            output = svm(feature)
            loss = torch.mean(torch.pow(torch.clamp(1 - output * target, min=0),2))  # hinge loss
            loss += 0.01 * torch.mean(svm.fc.weight**2)  # l2 penalty
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print('[%d/%d] [%d/%d] %.4f' % (epoch, max_epoch, i, len(train_loader), loss))
            f.write(str(epoch)+','+str(i)+','+str(loss)+'\n')
        torch.save(svm.state_dict(), '%s/%s_svm.pth' % ('trained_model',args.epoch))

    print("Training Complete")
    f.close()
    # return svm

def test(args, test_loader):
    '''
    Testing SVM
    '''
    netD = D_layer(nc,ndf,kernel_d)
    netD.load_state_dict(torch.load("trained_model/netD_imagenet.pth", map_location=lambda storage, loc: storage))
    netD.cuda(gpu_id)
    netD.eval()

    svm = SVMclassifier(28672, 101)
    svm.load_state_dict(torch.load("trained_model/svm.pth", map_location=lambda storage, loc: storage))
    svm.cuda(gpu_id)
    
    svm.eval()
    total_loss = 0.0
    tgt = []
    out = []
    for i, (x, y) in enumerate(test_loader):
        img_real = Variable(x).cuda(gpu_id)
        label = transform_to_target(y)
        target = Variable(label).cuda(gpu_id)

        feature = netD.generate_feature(img_real)
        output = svm(feature)
        
        loss = torch.mean(torch.clamp(1 - output * target, min=0))  # hinge loss
        print(i," ",float(loss))
        total_loss += float(loss)

        output = transform_to_list(output.data)
        target = transform_to_list(target.data)

        out += output
        tgt += target
        
    print("Total Loss = ", total_loss)
    accuracy = accuracy_score(tgt, out)
    print("Accuracy = ", accuracy)
    cnf_matrix = confusion_matrix(np.array(tgt), np.array(out))
    np.set_printoptions(precision=2)
    plt.figure()
    cnf.plot_confusion_matrix(cnf_matrix, classes=np.arange(101), normalize=False,
                        title='Normalized confusion matrix')

    plt.savefig("confussion_mat.png")
    # plt.show()

def main(args):
    train_loader, test_loader = load_data(args)
    
    if not args.test:
        train(args, train_loader)
    
    test(args, test_loader)

main(args)