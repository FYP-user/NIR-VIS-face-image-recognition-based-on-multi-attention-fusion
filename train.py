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



parser = argparse.ArgumentParser()                                  #Create parser

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

# Ablation experiment, incorporating Innovation Point 1, Innovation Point 2, and Innovation Point 3

def main():
    global args
    args = args.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids                   #Select the graphics card to use
    cudnn.benchmark = True                                              #Increase the program's running efficiency
    cudnn.enabled = True                                                #Allow the program to spend a little extra time at the beginning to search for the most suitable convolution implementation algorithm for each convolution layer in the network, thereby speeding up the network.

    random.seed(args.seed)                                              #random.seed():Initialize the random number generator. When dealing with a random program, as long as our operating environment is the same (ensuring the pseudorandom number generation program is the same) and we set the same random seed, we can reproduce the results.
    np.random.seed(args.seed)                                           #
    torch.manual_seed(args.seed)                                        #Set the CPU random number seed to facilitate reproducing the experiment results next time
    torch.cuda.manual_seed(args.seed)                                   #Set the CPU random number seed to facilitate reproducing the experiment results next time



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

            # Identification Contrast Loss
            loss_cont_pair = (1 - ang_loss(f_vis_j, f_nir_j)) + (1 - ang_loss(f_vis_k, f_nir_k))
            loss_cont_unpair = ang_loss(f_vis_j, f_nir_k) + ang_loss(f_vis_k, f_nir_j)
            loss_cont = 0.0015(loss_cont_pair + loss_cont_unpair)

            optimizer.zero_grad()
            loss_cont.backward()
            optimizer.step()

        # Storage Model
        if epoch % args.save_epoch == 0:
            save_checkpoint(model, epoch, "model/model_ablation/ablation_MAFCNN_DW_SA.tar")