import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def define_G(input_dim=3, output_dim=3, ndf=32):                                        # Dual Variational Generator

    net_decoder = decoder(output_dim, ndf)                                              # Decoder G
    net_encoder_nir = encoder(input_dim, ndf, get_num_adain_params(net_decoder))        # Near-Infrared Encoder En
    net_encoder_vis = encoder(input_dim, ndf, get_num_adain_params(net_decoder))        # Visible Light Encoder Ev

    net_decoder = torch.nn.DataParallel(net_decoder).cuda()                    #
    net_encoder_nir = torch.nn.DataParallel(net_encoder_nir).cuda()            #
    net_encoder_vis = torch.nn.DataParallel(net_encoder_vis).cuda()            #Use the `nn.DataParallel` function to accelerate training with multiple GPUs.

    return net_encoder_nir, net_encoder_vis, net_decoder


class encoder(nn.Module):                                               # Encoder
    def __init__(self, input_dim, ndf=32, h_dim=256):                   # input_dim: Number of input channels
        super(encoder, self).__init__()                                 # ndf: The number of filters, which corresponds to the number of output channels
                                                                        # h_dim: Number of channels in the hidden layer

        self.conv = nn.Sequential(                                      #
            convblock(input_dim, ndf, 5, 1, 2),                         # conv1:output：32*128*128
            convblock(ndf, 2 * ndf, 3, 2, 1),                           # conv2:output：64*64*64
            convblock(2 * ndf, 4 * ndf, 3, 2, 1),                       # conv3:output：128*32*32
            convblock(4 * ndf, 8 * ndf, 3, 2, 1),                       # conv4:output：256*16*16
            convblock(8 * ndf, 8 * ndf, 3, 2, 1),                       # conv5:output：256*8*8
            convblock(8 * ndf, 8 * ndf, 3, 2, 1)                        # conv6:output：256*4*4
        )

        self.fc_enc = nn.Linear(8 * ndf * 4 * 4, 256)                   # (input_dim,256)

        self.fc_style = nn.Sequential(
            nn.Linear(256, 1024),                                       # (input_dim,1024)
            nn.LeakyReLU(0.2, inplace=True),                            # inplace=True：will alter the input data
            nn.Linear(1024, 2048),                                      # (1024,2048)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, h_dim)                                      # (input_dim,h_dim)
        )

    def forward(self, x, state='enc'):
        if state == 'style':
            x = F.normalize(x, p=2, dim=1)
            style = self.fc_style(x)
            return style

        elif state == 'enc':
            x = self.conv(x)
            x = x.view(x.size(0), -1)                   #
            x = self.fc_enc(x)
            return F.normalize(x, p=2, dim=1)                           #
                                                                        #dim=1

class decoder(nn.Module):                                               # Decoder
    def __init__(self, output_dim=3, ndf=32):
        super(decoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(256+256, 4 * ndf * 4 * 4),                        # 2 (NIR和VIS)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv = nn.Sequential(
            deconvblock(4 * ndf, 4 * ndf, 2, 2, 0),                     #deconv1:output：128*8*8
            resblock(4 * ndf, 4 * ndf),
            deconvblock(4 * ndf, 4 * ndf, 2, 2, 0),                     #deconv2:output：128*16*16
            resblock(4 * ndf, 4 * ndf),
            deconvblock(4 * ndf, 2 * ndf, 2, 2, 0),                     #deconv3:output：64*32*32
            resblock(2 * ndf, 2 * ndf),
            deconvblock(2 * ndf, 2 * ndf, 2, 2, 0),                     #deconv4:output：64*64*64
            resblock(2 * ndf, 2 * ndf),
            deconvblock(2 * ndf, ndf, 2, 2, 0, norm='adain'),           #deconv5:output：32*128*128
            resblock(ndf, ndf, norm='adain'),
            convblock(ndf, ndf, 3, 1, 1, norm='adain'),
            resblock(ndf, ndf, norm='adain'),
            convblock(ndf, ndf, 3, 1, 1, norm='adain')
        )

        self.nir_output = nn.Conv2d(ndf, output_dim, 1, 1, 0)
        self.vis_output = nn.Conv2d(ndf, output_dim, 1, 1, 0)

    def forward(self, x, modality='NIR'):
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.conv(x)

        if modality == 'NIR':
            x = self.nir_output(x)
        elif modality == 'VIS':
            x = self.vis_output(x)
        return torch.sigmoid(x)


# basic module
class resblock(nn.Module):                                                  # resblock模块 in Decoder
    def __init__(self, input_dim, output_dim, norm='in'):
        super(resblock, self).__init__()

        self.conv1 = convblock(input_dim, output_dim, 3, 1, 1, norm)
        self.conv2 = convblock(output_dim, output_dim, 3, 1, 1, norm)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y

class convblock(nn.Module):                                     # Encoder per conv
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, norm='in'):        #norm='in'
        super(convblock, self).__init__()

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)

        if norm == 'bn':                                        #(BN，Batch Normalization)
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'in':                                      #(IN，Instance Normalization)
            self.norm = nn.InstanceNorm2d(output_dim)
        elif norm == 'adain':                                   #Adaptive Instance Normalization（AdaIN）
            self.norm = AdaptiveInstanceNorm2d(output_dim)

        self.activation = nn.LeakyReLU(0.2, inplace=True)       #

    def forward(self, x):                       # Processing flow for each conv layer
        x = self.conv(x)                        #
        x = self.norm(x)                        #
        x = self.activation(x)                  #
        return x                                #

class deconvblock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, norm='in'):
        super(deconvblock, self).__init__()

        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding)     # Transposed Convolution

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(output_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(output_dim)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)                                                    # x output (output_dim)
        x = self.norm(x)
        x = self.activation(x)
        return x

# AdaIN from: https://github.com/NVlabs/MUNIT
def get_num_adain_params(model):
    # return AdaIN parameters required for model
    num_adain_params = 0
    for m in model.modules():                                               # Recursively traverse all submodules
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):               # num_features = (output_dim)
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # Weights and biases are dynamically assigned.
        self.weight = None
        self.bias = None

        # Tensors registered via `register_buffer()` automatically become parameters within the model. They move with the model (GPU/CPU) but are not updated during gradient propagation.
        # torch.zeros(num_features) is a tensor that requires registration.
        self.register_buffer('running_mean', torch.zeros(num_features))     #
        self.register_buffer('running_var', torch.ones(num_features))       #

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)                                         # b: Number of rows in num_features; c: Number of columns in num_features
        running_mean = self.running_mean.repeat(b)                          #
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])           #

        out = F.batch_norm(                                                 #
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):                                                     #
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
