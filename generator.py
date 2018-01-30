import torch
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
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import datetime
import argparse
import os

# Read input argument
parser = argparse.ArgumentParser(description='dcgan')
parser.add_argument('--dataset', default='lsun', 
    help='choose the training data: {imagenet, lsun, mnist(not in the server)}')
parser.add_argument('--n', default=16, 
    help='how much image you want to generate')
parser.add_argument('--gpu', default=0,
    help='which gpu you want to use')
parser.add_argument('--arithmetic', default=False, action='store_true',
    help='whether do arithmetic or just generate image' )
parser.add_argument('--interpolate', default=False, action='store_true',
    help='whether use interpolation or use random number' )
args = parser.parse_args()

nz = 100
ngf = 128
ndf = 128
nc = 1 if args.dataset == 'mnist' else 3
k_size = 4
img_size = 64
out_size = int(args.n)

def create_random(n_size):
    t = torch.FloatTensor(n_size, nz, 1, 1).normal_(0, 1)
    return t

def show_image(img, size, name=None):
    img_np = img.data.cpu().numpy()
    
    fig = plt.figure()
    for j in range(img_np.shape[0]):
        img_show = np.transpose(img_np[j], (1,2,0))
        # print(np.max(img_show), np.min(img_show))
        img_show += 1
        img_show /= 2
        if nc == 1:
            img_show = img_show.reshape(img_np[0].shape[1], img_np[0].shape[2])
        sub = fig.add_subplot(size,size,j+1)
        plt.axis('off')
        
        sub.imshow(img_show, cmap='gray')
    
    if not name == None:
        plt.savefig(name)
    plt.show()

def show_arithmethic(A, B, C, D):
    arithmethic = torch.cat((A,B,C), 0)
    show_image(arithmethic, 4, 'mean')
    show_image(D, 1, 'result')


def choose_image(net_G):
    choosen_z = torch.FloatTensor(3, 100, 1, 1)
    for i in range(3):
        choice = -1
        while choice == -1:
            z = create_random(16)
            z = Variable(z).cuda()
            img = net_G(z)
            
            show_image(img, 4)

            choose = input('Choose image number: ')
            choice = int(choose)
            choosen_z[i] = z.data[choice]

    return choosen_z
            

def create_interpolation(n_size):
    maxmin = np.random.rand(100,2)
    maxmin *= 6
    maxmin -= 3

    noise = []
    for i in maxmin:
        linear = np.linspace(i[0],i[1],n_size)
        noise += [linear.tolist()]
    noise = np.array(noise).astype('float32')
    noise = noise.T
    int_noise = torch.from_numpy(noise)
    int_noise = int_noise.unsqueeze(2)
    int_noise = int_noise.unsqueeze(3)
    return int_noise

# Generator Model
class G_layer(nn.Module):
    def __init__(self):
        super(G_layer, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding)
            nn.ConvTranspose2d(nz, ngf * 8, k_size, 1, 0, bias=False), 
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, k_size, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, k_size, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, k_size, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, k_size, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        out = self.main(input)
        return out

netG = G_layer()
netG.cuda()
netG.load_state_dict(torch.load('trained_model/netG_%s.pth' % args.dataset, map_location=lambda storage, loc: storage))
netG.eval()

if not args.arithmetic:
    if not args.interpolate:
        z = create_random(out_size)
        z = Variable(z).cuda()    
    else:
        z = create_interpolation(out_size)
        sample = z.squeeze(3)
        sample = sample.squeeze(2)
        z = Variable(z).cuda()

    fake = netG(z)
    vutils.save_image(fake.data, 'test_generate.png', nrow=4, normalize=True)

else:
    print("==== Image A ====")
    A = choose_image(netG)
    A_mean = torch.mean(A, 0, keepdim=True)
    A = torch.cat((A, A_mean), 0)
    A = Variable(A).cuda()
    img_A = netG(A)
    # show_image(img_A)

    print("==== Image B ====")
    B = choose_image(netG)
    B_mean = torch.mean(B, 0, keepdim=True)
    B = torch.cat((B, B_mean), 0)
    B = Variable(B).cuda()
    img_B = netG(B)
    # show_image(img_B)

    print("==== Image C ====")
    C = choose_image(netG)
    C_mean = torch.mean(C, 0, keepdim=True)
    C = torch.cat((C, C_mean), 0)
    C = Variable(C).cuda()
    img_C = netG(C)
    # show_image(img_C)

    D = A_mean + B_mean
    D = Variable(D).cuda()
    img_D = netG(D)
    show_arithmethic(img_A, img_B, img_C, img_D)

