import torch
import torch.nn as nn

# Generator Model
class G_layer(nn.Module):
    def __init__(self, nz, ngf, kernel_g, nc):
        super(G_layer, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding)
            nn.ConvTranspose2d(nz, ngf * 8, kernel_g[0], 1, 0, bias=False), 
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4 || 16x16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_g[1], 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8 || 32x32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_g[2], 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16 || 64x64
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_g[3], 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32 || 128x128
            nn.ConvTranspose2d(ngf, nc, kernel_g[4], 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64 || 256x256
        )

    def forward(self, input):
        out = self.main(input)
        return out

# Discriminator
class D_layer(nn.Module):
    def __init__(self, nc, ndf, kernel_d):
        super(D_layer, self).__init__()
        # input is (nc) x 64 x 64	
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_d[0], 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_d[1], 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_d[2], 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_d[3], 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_d[4], 1, 0, bias=False),
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, input):
        output1 = self.conv1(input)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        output4 = self.conv4(output3)
        output5 = self.conv5(output4)
        return output5.view(-1, 1).squeeze(1)
    
    def generate_feature(self, input):
        output1 = self.conv1(input)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        output4 = self.conv4(output3)
        output5 = self.conv5(output4)

        maxpool0 = nn.MaxPool2d(16)
        maxpool1 = nn.MaxPool2d(4)
        maxpool2 = nn.MaxPool2d(2)

        # feature0 = maxpool0(output1).view(input.size(0), -1).squeeze(1)
        feature1 = maxpool1(output2).view(input.size(0), -1).squeeze(1)
        feature2 = maxpool2(output3).view(input.size(0), -1).squeeze(1)
        feature3 = output4.view(input.size(0), -1).squeeze(1)

        # print(feature1.size())
        # print(feature2.size())
        # print(feature3.size())

        return torch.cat((feature1, feature2, feature3), 1)
        # return torch.cat((feature1, feature2), 1)
        # return feature1
        # return feature0
        # return torch.cat((feature0, feature1), 1)
        # return output