import torch
import torch.nn as nn
from spectral import SpectralNorm
from torch.nn import functional as F

class generator(torch.nn.Module):
    def __init__(self, in_nc, out_nc, nf):
        super(generator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf

        self.downconv1 = nn.Sequential(#input H,W 3 output H,W 64  1
            SpectralNorm(nn.Conv2d(in_nc, nf, 5, 1, 2)),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
        )

        self.downconv2 = nn.Sequential(#input H,W 64 output H/2,W/2 128  2
            SpectralNorm(nn.Conv2d(nf, nf * 2, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )

        self.downconv3 = nn.Sequential(#input H/2,W/2 128 output H/4,W/4 256  3
            SpectralNorm(nn.Conv2d(nf * 2, nf * 4, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.downconv4 = nn.Sequential(# input H/4,W/4 256 output H/8,W/8 512  4
            SpectralNorm(nn.Conv2d(nf * 4, nf * 8, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.ReLU(True),
        )

        self.downconv5 = nn.Sequential(# input H/8,W/8 512 output H/8,W/8 512  5
            SpectralNorm(nn.Conv2d(nf * 8, nf * 8, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.ReLU(True),
        )


        self.upconv3 = nn.Sequential(# input H/8,W/8 1024 output H/4,W/4 256  6
            SpectralNorm(nn.ConvTranspose2d(nf * 16, nf * 4, 4, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.upconv2 = nn.Sequential(# input H/4,W/4 512 output H/2,W/2 128  7
            SpectralNorm(nn.ConvTranspose2d(nf * 8, nf * 2, 4, 2, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
        )

        self.upconv1 = nn.Sequential(#input H/2,W/2 256 output H,W 3  8
            SpectralNorm(nn.ConvTranspose2d(nf * 4, nf, 4, 2, 1)),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 5, 1, 2),
            nn.Tanh(),
        )

        # forward method
    def forward(self, input):

        x1 = self.downconv1(input) #64 H,W
        x2 = self.downconv2(x1) #128 H/2,W/2
        x3 = self.downconv3(x2) #256 H/4,W/4
        x4 = self.downconv4(x3) #512 H/8,W/8
        x5 = self.downconv5(x4) #512 H/8,W/8

        y3 = self.upconv3(torch.cat([x4, x5], dim=1)) #256 H/4,W/4
        y2 = self.upconv2(torch.cat([y3, x3], dim=1)) #128 H/2,W/2
        y1 = self.upconv1(torch.cat([y2, x2], dim=1)) #64 H,W
        output = y1

        return output


class discriminator(nn.Module):
    def __init__(self, in_nc, out_nc, nf=32):
        super(discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.dis = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_nc, nf, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(nf, nf * 2, 3, 1, 1)),
            nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),

            SpectralNorm(nn.Conv2d(nf * 2, nf * 4, 3, 2, 1)),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            SpectralNorm(nn.Conv2d(nf * 4, nf * 8, 3, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),

            SpectralNorm(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1)),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, out_nc, 3, 1, 1),
        )

    # forward method
    def forward(self, input):

        output = self.dis(input)

        return F.sigmoid(output)

class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs


