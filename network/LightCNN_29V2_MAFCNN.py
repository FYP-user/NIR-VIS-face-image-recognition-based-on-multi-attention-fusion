import torch
import torch.nn as nn
import torch.nn.functional as F
from caffe2.python.layers.concat import Concat


def LightCNN_29v2(is_train=True):
    net = network_29layers_v2(is_train)
    net = torch.nn.DataParallel(net).cuda()
    return net


class network_29layers_v2(nn.Module):
    def __init__(self, block, layers, is_train=False, num_classes=714, in_channels=None, out_channels=None):
        super(network_29layers_v2, self).__init__()
        self.is_train = is_train

        self.conv_downsampling1 = nn.Conv2d(3, 3, kernel_size=2, stride=2)
        self.MAFCNN1 = MAFCNN(in_channels, out_channels)
        self.DW1 = DepthWiseConv(in_channels, out_channels)
        self.mfm1 = mfm(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_downsampling2 = nn.Conv2d(3, 3, kernel_size=2, stride=2)
        self.MAFCNN2 = MAFCNN(in_channels, out_channels)
        self.DW2 = DepthWiseConv(in_channels, out_channels)
        self.mfm2 = mfm(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_downsampling2 = nn.Conv2d(3, 3, kernel_size=2, stride=2)
        self.MAFCNN3 = MAFCNN(in_channels, out_channels)
        self.DW3 = DepthWiseConv(in_channels, out_channels)
        self.mfm3 = mfm(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(8 * 8 * 128, 256)
        if self.is_train:
            self.fc2_ = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x = self.conv_downsampling1(x)
        x = self.MAFCNN1(x)
        x = self.DW1(x)
        x = self.mfm1(x)

        x = self.conv_downsampling2(x)
        x = self.MAFCNN2(x)
        x = self.DW2(x)
        x = self.mfm2(x)

        x = self.conv_downsampling3(x)
        x = self.MAFCNN3(x)
        x = self.DW3(x)
        x = self.mfm3(x)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        out = self.softmax(fc)


# 1. Multi-Attention Fusion Convolutional Neural Network Model
class MAFCNN(nn.Module):
    def __init__(self, x, in_channels, out_channels, input1=None, input2=None):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(x)
        self.max_pool = nn.MaxPool2d(x)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=(2, 1))
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1))
        self.concat = nn.Concat(input1, input2)

    def forward(self, x):
        avg_H = self.avg_pool(x)
        max_H = self.max_pool(x)
        out_H = self.conv1(Concat(avg_H + max_H))
        out_H = self.conv2(out_H)
        out_H = self.softmax(out_H)

        avg_W = self.avg_pool(x)
        max_W = self.max_pool(x)
        out_W = self.conv1(Concat(avg_W + max_W))
        out_W = self.conv2(out_W)
        out_W = self.softmax(out_W)

        avg_C = self.avg_pool(x)
        max_C = self.max_pool(x)
        out_C = Concat(self.conv1(avg_C) + self.conv1(max_C))
        out_C = self.Conv2(out_C)
        out_C = self.softmax(out_C)

        out = x * out_H * out_W * out_C
        return out

# Traditional Deep Separable Convolution
class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.point_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, out=None):
        out = self.point_conv(out)
        out = self.depth_conv(input)
        out = self.point_conv(out)
        out = self.PReLU(out)
        return out


# softmax
class Softmax(nn.Module):
    def __init__(self, x):
        super().__init__()
        x_output = torch.nn.functional.softmax(x)
        return x_output


# PReLU
class PReLU(nn.Module):
    def __init__(self, x):
        super(network_29layers_v2, self).__init__()
        x_output = torch.nn.PReLU(x)
        return x_output


# Maximum Feature Mapping Operation
class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)  # 
        return torch.max(out[0], out[1])




