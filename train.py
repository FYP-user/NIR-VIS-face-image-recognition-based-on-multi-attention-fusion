import os
import time
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.utils.data

from network import lightcnn, LightCNN_29V2
from utils import *
from network.lightcnn import LightCNN_29v2
from data.dataset_mix import Real_Dataset, Mix_Dataset



parser = argparse.ArgumentParser()                                  #创建解析器

parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--pre_epoch', default=0, type=int, help='train_param from previous model')

parser.add_argument('--print_iter', default=20, type=int, help='print frequency')
parser.add_argument('--save_epoch', default=1, type=int)
parser.add_argument('--output_path', default='./results', type=str)

parser.add_argument('--weights_lightcnn', default='./pre_train/LightCNN_29Layers_V2_checkpoint.pth.tar', type=str)
parser.add_argument('--weights_dec', default='./pre_train/dec_epoch_45.pth.tar', type=str, help='dec is the identity sampler')
parser.add_argument('--img_root',  default='dataset/CASIA_2.0/', type=str)
parser.add_argument('--train_list', default='dataset/CASIA_2.0/together.csv', type=str)

# 消融实验，融入创新点1和创新点2和创想点3

def main():
    global args
    args = args.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids                   #指定要使用的显卡
    cudnn.benchmark = True                                              #增加程序的运行效率
    cudnn.enabled = True                                                #让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速

    random.seed(args.seed)                                              #random.seed():初始化随机数生成器,当面对一个随机程序的时候，只要我们的运行环境一致（保证伪随机数生成程序一样），而我们设定的随机种子一样的话，那么我们就可以复现结果
    np.random.seed(args.seed)                                           #
    torch.manual_seed(args.seed)                                        #设置CPU生成随机数的种子，方便下次复现实验结果
    torch.cuda.manual_seed(args.seed)                                   #设置GPU生成随机数的种子，方便下次复现实验结果



def train(train_loader, model, optimizer, epoch):

    model.train()
    model = LightCNN_29V2


    for j, data in enumerate(train_loader):
        real_vis_j = Variable(data["dataset/BUAA_VisNir/VIS"].cuda())
        real_nir_j = Variable(data["dataset/BUAA_VisNir/NIR"].cuda())
        f_vis_j = model(real_vis_j)[0]
        f_nir_j = model(real_nir_j)[0]

        for k, data in enumerate(train_loader):
            real_vis_k = Variable(data["dataset/BUAA_VisNir/VIS"].cuda())
            real_nir_k = Variable(data["dataset/BUAA_VisNir/NIR"].cuda())
            f_vis_k = model(real_vis_k)[0]
            f_nir_k = model(real_nir_k)[0]

            # 识别对比损失
            loss_cont_pair = (1 - ang_loss(f_vis_j, f_nir_j)) + (1 - ang_loss(f_vis_k, f_nir_k))
            loss_cont_unpair = ang_loss(f_vis_j, f_nir_k) + ang_loss(f_vis_k, f_nir_j)
            loss_cont = 0.0015(loss_cont_pair + loss_cont_unpair)

            optimizer.zero_grad()
            loss_cont.backward()
            optimizer.step()

        # 存储模型
        if epoch % args.save_epoch == 0:
            save_checkpoint(model, epoch, "model/model_ablation/ablation_MAFCNN_DW_SA.tar")