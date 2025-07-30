import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def define_G(input_dim=3, output_dim=3, ndf=32):                                        #对偶变分发生器

    net_decoder = decoder(output_dim, ndf)                                              #解码器G
    net_encoder_nir = encoder(input_dim, ndf, get_num_adain_params(net_decoder))        #近红外编码器En
    net_encoder_vis = encoder(input_dim, ndf, get_num_adain_params(net_decoder))        #可见光编码器Ev

    net_decoder = torch.nn.DataParallel(net_decoder).cuda()                    #在多卡的GPU服务器，当我们在上面跑程序的时候，
    net_encoder_nir = torch.nn.DataParallel(net_encoder_nir).cuda()            #当迭代次数或者epoch足够大的时候，我们通常会
    net_encoder_vis = torch.nn.DataParallel(net_encoder_vis).cuda()            #使用nn.DataParallel函数来用多个GPU来加速训练

    return net_encoder_nir, net_encoder_vis, net_decoder


class encoder(nn.Module):                                               #编码器
    def __init__(self, input_dim, ndf=32, h_dim=256):                   #input_dim:输入通道数
        super(encoder, self).__init__()                                 #ndf:滤波器的数量，也就是输出通道数
                                                                        #h_dim:隐藏层的通道数

        self.conv = nn.Sequential(                                      #实现简单的顺序连接模型,相当于把多个模块封装成一个模块
            convblock(input_dim, ndf, 5, 1, 2),                         # conv1:output：32*128*128
            convblock(ndf, 2 * ndf, 3, 2, 1),                           # conv2:output：64*64*64
            convblock(2 * ndf, 4 * ndf, 3, 2, 1),                       # conv3:output：128*32*32
            convblock(4 * ndf, 8 * ndf, 3, 2, 1),                       # conv4:output：256*16*16
            convblock(8 * ndf, 8 * ndf, 3, 2, 1),                       # conv5:output：256*8*8
            convblock(8 * ndf, 8 * ndf, 3, 2, 1)                        # conv6:output：256*4*4
        )

        self.fc_enc = nn.Linear(8 * ndf * 4 * 4, 256)                   #线性变换(input_dim,256)

        self.fc_style = nn.Sequential(
            nn.Linear(256, 1024),                                       #(input_dim,1024)
            nn.LeakyReLU(0.2, inplace=True),                            #inplace=True：将会改变输入的数据
            nn.Linear(1024, 2048),                                      #(1024,2048)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, h_dim)                                      #(input_dim,h_dim)
        )

    def forward(self, x, state='enc'):
        if state == 'style':
            x = F.normalize(x, p=2, dim=1)
            style = self.fc_style(x)
            return style

        elif state == 'enc':
            x = self.conv(x)
            x = x.view(x.size(0), -1)                   #指转换后有几行，而-1指在不告诉函数有多少列的情况下，根据原数据和x.size(0)自动分配列数
            x = self.fc_enc(x)
            return F.normalize(x, p=2, dim=1)                           #归一化,p指的是求p范数的p值，函数默认p=2，那么就是求2范数
                                                                        #dim=1，指对每一行的每一列求2范数

class decoder(nn.Module):                                               #解码器
    def __init__(self, output_dim=3, ndf=32):
        super(decoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(256+256, 4 * ndf * 4 * 4),                        #因为有两个(NIR和VIS)
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
class resblock(nn.Module):                                                  #解码器resblock模块
    def __init__(self, input_dim, output_dim, norm='in'):
        super(resblock, self).__init__()

        self.conv1 = convblock(input_dim, output_dim, 3, 1, 1, norm)
        self.conv2 = convblock(output_dim, output_dim, 3, 1, 1, norm)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y

class convblock(nn.Module):                                     #编码器每层conv
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, norm='in'):     #norm='in':实力标准化(IN)层
        super(convblock, self).__init__()

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)

        if norm == 'bn':                                        #批量标准化(BN，Batch Normalization)
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'in':                                      #实例标准化(IN，Instance Normalization)
            self.norm = nn.InstanceNorm2d(output_dim)
        elif norm == 'adain':                                   #Adaptive Instance Normalization（AdaIN）
            self.norm = AdaptiveInstanceNorm2d(output_dim)

        self.activation = nn.LeakyReLU(0.2, inplace=True)       #激活函数

    def forward(self, x):                       #每一层conv的处理流程：      卷积层
        x = self.conv(x)                        #                          ||
        x = self.norm(x)                        #                         IN层
        x = self.activation(x)                  #                          ||
        return x                                #                         激活层

class deconvblock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, norm='in'):
        super(deconvblock, self).__init__()

        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding)     #转置卷积

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(output_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(output_dim)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)                                                    #x经过转置卷积层(nn.ConvTranspose2d) 得到输出(output_dim)
        x = self.norm(x)
        x = self.activation(x)
        return x

# AdaIN from: https://github.com/NVlabs/MUNIT
def get_num_adain_params(model):
    # 返回模型所需的AdaIN参数
    num_adain_params = 0
    for m in model.modules():                                               #递归遍历所有的子模块，包括子模块的子模块等等以此类推
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):               #这里 num_features 即为(output_dim)
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # 权重和偏置是动态分配的
        self.weight = None
        self.bias = None

        #通过register_buffer()登记过的张量：会自动成为模型中的参数，随着模型移动（gpu/cpu）而移动，但是不会随着梯度进行更新
        # torch.zeros(num_features) 是需要进行register登记的张量
        self.register_buffer('running_mean', torch.zeros(num_features))     #torch.zeros():返回一个形状为(num_features)，里面的每一个值都是0的tensor
        self.register_buffer('running_var', torch.ones(num_features))       #torch.ones():返回一个形状为(num_features)全为1的张量

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)                                         #b:num_features的行数； c:num_features的列数
        running_mean = self.running_mean.repeat(b)                          #repeat的参数是对应维度的复制个数
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])           #contiguous():首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列。

        out = F.batch_norm(                                                 #batch_nor():对跨一批数据的每个通道应用批归一化。
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):                                                     #定义了实例化对象的输出信息，重写该方法，可以输出我们想要的信息
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
