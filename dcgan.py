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
# import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import datetime
import argparse
import os
from model import *

# Read input argument
parser = argparse.ArgumentParser(description='dcgan')
parser.add_argument('--dataset', default='lsun', 
    help='choose the training data: {imagenet, lsun, mnist(not in the server)}')
parser.add_argument('--load', default=False, action='store_true', 
    help='load previousely trained model or not')
parser.add_argument('--nonstop', default=False, action='store_true', 
    help='training without time limit or not. You want to use this on the server')
parser.add_argument('--sample', default=1, 
    help='how much dataset you want to use')
parser.add_argument('--save', default=-1, 
    help='how much iteration interval to save image')
parser.add_argument('--batch', default=64, 
    help='size of batch training')
parser.add_argument('--gpu', default=0,
    help='which gpu you want to use')
args = parser.parse_args()

# Training time limit
max_hour = 19
max_minute = 58

'''
Load data
'''
def load_data(args):
    ''' 
    Parameters

    nz = number of input z
    ngf = number of feature in generator
    ndf = number of feature in discriminator
    img_size = image size

    '''
    nz = 100
    ngf = 128
    ndf = 128
    img_size = 64

    # Switch GPU automatically
    today = datetime.datetime.today().weekday()
    if args.gpu == None:
        if today == 6:
            gpu_id = 1
        else:
            gpu_id = 0
    else:
        gpu_id=int(args.gpu)
    torch.cuda.set_device(gpu_id)

    # Loading dataset
    transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    '''
    Loading dataset parameter
    it_save_image   : number of iteration to save image result
    nc              : number of channel
    batch_size      : size of batch
    lr_g            : learning rate for generator
    lr_d            : learning rate for discriminator
    '''
    if args.dataset == 'imagenet':
        traindir = '/home/data/imagenet-2012/train'
        # traindir = './imagenet'
        train_dataset = datasets.ImageFolder(traindir, transform=transform)
        if args.save == '-1':
            it_save_image = 100
        else: 
            it_save_image = int(args.save)
        nc = 3
        batch_size = 128
        lr_g = 0.0002
        lr_d = 0.0002
        beta_g = 0.5
        beta_d = 0.5
        kernel_g = [4,4,4,4,4]    
        kernel_d = [4,4,4,4,4]

    elif args.dataset == 'mnist':
        traindir = './MNIST_data'
        train_dataset = datasets.MNIST(traindir, train=True, download=True,
            transform=transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor()
            ]))
        
        if args.save == '-1':
            it_save_image = 100
        else: 
            it_save_image = int(args.save)
        nc = 1
        batch_size = int(args.batch_size)
        max_epoch = 200
        lr_g = 0.0002
        lr_d = 0.0002
        beta_g = 0.9
        beta_g = 0.5
        beta_d = 0.5
        kernel_g = [6,6,6,6,8]    
        kernel_d = [6,6,6,6,6]
        
    elif args.dataset == 'lsun':
        traindir = '/home/data/lsun/train'
        # traindir = '/media/yonathan/DOCUMENTS/lsun/lsun-master/'
        train_dataset = datasets.LSUN(traindir, classes=['church_outdoor_train'], 
            transform=transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        if args.save == '-1':
            it_save_image = 300
        else: 
            it_save_image = int(args.save)
        nc = 3
        batch_size = int(args.batch)
        max_epoch = 200        
        lr_g = 0.0002
        lr_d = 0.0002
        beta_d = 0.5
        kernel_g = [6,6,6,6,8]
        kernel_d = [6,6,6,6,6]

    elif args.dataset == 'pokemon':
        traindir = './pokemon'
        # transform=transforms.Compose([transforms.Resize(img_size),transforms.CenterCrop(img_size),transforms.ToTensor()]))
        train_dataset = datasets.ImageFolder(traindir, 
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        if args.save == '-1':
            it_save_image = 400
        else: 
            it_save_image = int(args.save)
        nc = 3
        batch_size = int(args.batch)
        lr_g = 0.0002
        lr_d = 0.0002
        beta_g = 0.5
        beta_d = 0.5
        kernel_g = [16,4,4,4,4]
        kernel_d = [4,4,4,4,16]
        
    '''
    Sample from whole dataset so it will be faster
    split_percentage    : how much percentage do we want
    '''
    split_percentage = float(args.sample)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(split_percentage * num_train))
    np.random.shuffle(indices)
    train_idx = indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)

    # Load the training dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    param = (
        nz, nc, ngf, ndf, 
        lr_g, lr_d, beta_g, beta_d, 
        kernel_g, kernel_d, 
        it_save_image, batch_size, gpu_id, max_epoch)
    # =================================================
    #  Sanity Check
    #  showing input image
    # =================================================
    # for x,y in train_loader:
    # 	for i in x:
    # 		# np.dstack((img[0],img[1],img[2]))
    # 		print(i)
    # 		# i = torch.clamp(i, 0, 1)
    # 		plt.imshow(np.dstack((i[0],i[1],i[2])))
    # 		plt.axis('off')
    # 		plt.show()
    return param, train_loader

# Create input Z with random normal
def create_input(batch_size):
    t = torch.FloatTensor(batch_size, 100, 1, 1).normal_(0, 1)
    return t

def create_interpolate(batch_size):
    maxmin = np.random.rand(100,2)
    maxmin *= 4
    maxmin -= 2

    noise = []
    for i in maxmin:
        linear = np.linspace(i[0],i[1],batch_size)
        noise += [linear.tolist()]
    noise = np.array(noise).astype('float32')
    noise = noise.T
    int_noise = torch.from_numpy(noise)
    int_noise = int_noise.unsqueeze(2)
    int_noise = int_noise.unsqueeze(3)

    return int_noise

# Initialize weight
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02).cuda(gpu_id)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02).cuda(gpu_id)
        m.bias.data.fill_(0).cuda(gpu_id)

def train(args, param, train_loader):
    (nz, nc, ngf, ndf, 
        lr_g, lr_d, beta_g, beta_d, 
        kernel_g, kernel_d, 
        it_save_image, batch_size, gpu_id, max_epoch) = param
    '''
    Defining generator model
    '''
    netG = G_layer(nz,ngf,kernel_g,nc)
    netG.cuda(gpu_id)
    # netG.apply(weights_init)
    if args.load:
        netG.load_state_dict(torch.load("trained_model/netG_"+args.dataset+".pth"))

    '''
    Defining discriminator model
    '''
    netD = D_layer(nc,ndf,kernel_d)
    netD.cuda(gpu_id)
    # netD.apply(weights_init)
    if args.load:
        netD.load_state_dict(torch.load("model/netD_"+args.dataset+".pth"))

    criterion = nn.BCELoss()

    '''
    Fixed noise for generating image in that will be saved
    every it_save_image
    
    Either use the fixed noise or the interpolation
    '''
    fixed_noise = create_input(batch_size)
    fixed_noise = Variable(fixed_noise).cuda(gpu_id)

    interpolate = create_interpolate(batch_size)
    interpolate = Variable(interpolate).cuda(gpu_id)
    '''
    Create folder for saving model and image result
    also create a text file containing training graph
    '''
    if not os.path.exists('trained_model/'): os.makedirs('trained_model/')
    if not os.path.exists('result/'): os.makedirs('result/')
    if not os.path.exists("result/"+args.dataset): os.makedirs("result/"+args.dataset)
    if not os.path.exists('graph/'): os.makedirs('graph/')

    if not args.load:
        f = open('graph/graph_'+args.dataset+'.txt', 'w')
        f.write(args.dataset+'\n')
        f.close()

    f = open('graph/graph_'+args.dataset+'.txt', 'a')

    # Training
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta_g, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta_d, 0.999))
    netG.eval()
    fake = netG(interpolate)
    vutils.save_image(fake.data, '%s/%s/fake_samples_epoch_%03d.png' % ('result', args.dataset, -1), nrow=2, normalize=True)
    netG.train()
    for epoch in range(max_epoch):
        for i, (x, y) in enumerate(train_loader, 0):
            # ==============
            # Forward
            # ==============
            netD.zero_grad()	
            z = create_input(len(x))
            z_variable = Variable(z).cuda(gpu_id)

            # Generator
            img_real = Variable(x).cuda(gpu_id)
            img_fake = netG(z_variable)

            class1 = Variable(torch.ones(len(y))).cuda(gpu_id) # real
            class0 = Variable(torch.zeros(len(y))).cuda(gpu_id) # fake

            # Discriminator
            outputr = netD(img_real) # output real
            outputf = netD(img_fake.detach()) # output fake
            
            D_x = outputr.data.mean()
            D_G_z1 = outputf.data.mean()

            # ==============
            # Backward
            # ==============

            # Discriminator
            errD_fake = criterion(outputf, class0) # Calculating Discriminator loss
            errD_fake.backward(retain_graph=True) # backprop
            errD_real = criterion(outputr, class1) # Calculating Discriminator loss
            errD_real.backward(retain_graph=True) # Backprop
            errD = errD_real + errD_fake # Calculating Discriminator total Loss
            optimizerD.step()
            optimizerD.zero_grad()
            
            # Generator
            netG.zero_grad()
            outputg = netD(img_fake)
            errG = criterion(outputg, class1)
            errG.backward()
            D_G_z2 = outputg.data.mean()
            optimizerG.step()
            optimizerG.zero_grad()

            # =================
            # Printing and Saving
            # =================
            print('[%d/%d][%d/%d] Loss_D: %.5f Loss_G: %.5f D(x): %.5f D(G(z)): %.4f / %.4f' % (epoch, max_epoch, i, len(train_loader),
                errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            f.write(str(epoch)+','+str(i)+','+str(errD.data[0])+','+str(errG.data[0])+'\n')
            
            if i % it_save_image == 0 or i == len(train_loader)-1:
                
            # if epoch % 10 == 0:	
                torch.save(netG.state_dict(), '%s/netG_%s.pth' % ('trained_model', args.dataset))
                torch.save(netD.state_dict(), '%s/netD_%s.pth' % ('trained_model', args.dataset))

                netG.eval()
                vutils.save_image(img_real.data, '%s/%s/real_samples.png' % ('result', args.dataset), nrow=2, normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.data, '%s/%s/fake_samples_it_%05d.png' % ('result', args.dataset, i), nrow=2, normalize=True)
                netG.train()
                
            # Checking time for auto stop
            now = datetime.datetime.now()
            if now.hour >= max_hour and now.minute > max_minute and not args.nonstop:
                break
        
        netG.eval()
        fake = netG(interpolate)
        vutils.save_image(fake.data, '%s/%s/fake_samples_epoch_%03d.png' % ('result', args.dataset, epoch), nrow=2, normalize=True)
        netG.train()
        
        # Checking time again for auto stop
        now = datetime.datetime.now()
        if now.hour >= max_hour and now.minute > max_minute and not args.nonstop:
            break
        # =================================================
        #  Sanity Check
        #  showing output image
        # =================================================
        # print(output[0])
        # out = output.data[0]
        # plt.imshow(np.dstack((out[0],out[1],out[2])))
        # plt.axis('off')
        # plt.show()	

def main(args):
    param, train_loader = load_data(args)
    
    train(args, param, train_loader)


main(args)