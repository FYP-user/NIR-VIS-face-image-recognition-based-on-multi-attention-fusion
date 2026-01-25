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

from utils import *
from network.lightcnn import LightCNN_29v2
from data.dataset_mix import Real_Dataset, Mix_Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', default=719, type=int)
parser.add_argument('--gpu_ids', default='0,1', type=str)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--epochs', default=15, type=int)
parser.add_argument('--pre_epoch', default=0, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=2e-4)
parser.add_argument('--step_size', default=5, type=int)
parser.add_argument('--print_iter', default=5, type=int)
parser.add_argument('--save_name', default='LightCNN', type=str)
parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--weights_lightcnn', default='./pre_train/LightCNN_29Layers_V2_checkpoint.pth.tar', type=str)
parser.add_argument('--img_root_A', default='dataset/CASIA_2.0/', type=str)
parser.add_argument('--train_list_A', default='dataset/CASIA_2.0/together.csv', type=str)
parser.add_argument('--img_root_B', default='./gen_images/NIR', type=str)
parser.add_argument('--train_list_B', default='./gen_images/img_list.txt', type=str)


def main():
    global args
    args = parser.parse_args()                                          #Return all the 'add_argument' settings from the parser to the args subclass instance
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    cudnn.benchmark = True                                              #Increase the program's running efficiency
    cudnn.enabled = True                                                #Allow the program to spend a little extra time at the beginning to search for the most suitable convolution implementation algorithm for each convolution layer in the entire network, thereby speeding up the network.

    random.seed(args.seed)                                              #random.seed():Initialize the random number generator. When dealing with a random program, as long as our operating environment is the same (ensuring the pseudorandom number generation program is the same) and we set the same random seed, we can reproduce the results.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # lightcnn
    model = LightCNN_29v2(num_classes=args.num_classes)                 #The model is lightcnn

    # Load pretrained model
    if args.pre_epoch:
        print('load pretrained model of epoch %d' % args.pre_epoch)
        load_model(model, "./model/lightCNN_epoch_%d.pth.tar" % args.pre_epoch)
    else:
        print("=> loading pretrained lightcnn '{}'".format(args.weights_lightcnn))
        load_model(model, args.weights_lightcnn)

    # Load trained real data
    train_loader_real = torch.utils.data.DataLoader(
        Real_Dataset(args), batch_size=2*args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # Load the mixed training data (generated and real data)
    train_loader_mix = torch.utils.data.DataLoader(
        Mix_Dataset(args), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # Cross-Entropy Loss Function
    criterion = nn.CrossEntropyLoss().cuda()

    # Stage 1: Pre-train the model on the last fc2 parameter
    params_pretrain = []
    for name, value in model.named_parameters():
        if "fc2_" in name:
            params_pretrain += [{"params": value, "lr": 1 * args.lr}]

    # Optimization Algorithm (Stochastic Gradient)
    optimizer_pretrain = torch.optim.SGD(params_pretrain, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, 5):
        pre_train(train_loader_real, model, criterion, optimizer_pretrain, epoch)
        save_checkpoint(model, epoch, "LightCNN_pretrain")

    '''
    Stage 2: Full-network Model Fine-tuning
    '''
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start_epoch = args.pre_epoch + 1
    for epoch in range(start_epoch, args.epochs + 1):
        adjust_learning_rate(args.lr, args.step_size, optimizer, epoch)
        train(train_loader_mix, model, criterion, optimizer, epoch)
        save_checkpoint(model, epoch, args.save_name)


# Pre-train the last fc2 parameter
def pre_train(train_loader, model, criterion, optimizer, epoch):
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()               #Enable Batch Normalization and Dropout to ensure that the BN layer can use the mean and variance of each batch of data, and randomly select a portion of network connections to train and update parameters.
    for i, data in enumerate(train_loader):                             #i: index of the training data; data: the training data
        input = Variable(data["img"].cuda())                            #Here, input is the Variable parameter form of img.
        label = Variable(data["label"].cuda())                          #Here, 'label' is the Variable parameter form of 'label'.
        batch_size = input.size(0)                                      #batch_size is the number of rows in the input

        if batch_size < 2*args.batch_size:
            continue

        output = model(input)[0]                                        #Output identity representation
        loss = criterion(output, label)                                 #Cross-Entropy Loss Function
        optimizer.zero_grad()                                           #Zero the gradients
        loss.backward()                                                 #Backpropagation calculates the gradient value for each parameter
        optimizer.step()                                                #Perform a one-step parameter update using gradient descent

        # Measurement accuracy and record loss
        prec1, prec5 = accuracy(output.data, label.data, topk=(1, 5))
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        # Output log
        if i % args.print_iter == 0:
            info = "====> Epoch: [{:0>3d}][{:3d}/{:3d}] | ".format(epoch, i, len(train_loader))
            info += "Loss: ce: {:4.3f} | ".format(loss.item())
            info += "Prec@1: {:4.2f} ({:4.2f}) Prec@5: {:4.2f} ({:4.2f})".format(top1.val, top1.avg, top5.val, top5.avg)
            print(info)


def train(train_loader, model, criterion, optimizer, epoch):
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    for i, data in enumerate(train_loader):
        # Real data
        input_real = Variable(data["img_A"].cuda())                     #Here, input_real is the Variable form of img_A
        label = Variable(data["label"].cuda())                          #Here, 'label' is the Variable parameter form of 'label'.

        # Generate data
        fake_nir = Variable(data["img_B"].cuda())                       #Here, fake_nir is the variable parameter form of img_B.
        fake_vis = Variable(data["img_B_pair"].cuda())                  #Here, fake_vis is the Variable form of the img_B_pair parameter

        batch_size = input_real.size(0)                                 # batch_size is the number of rows in input_real
        if batch_size < args.batch_size:
            continue

        output = model(input_real)[0]                                   #Output identity representation
        loss_ce = criterion(output, label)                              #Calculate the cross-entropy loss function

        fc_nir = model(fake_nir)[1]
        fc_vis = model(fake_vis)[1]

        # For negative compilation of random indexes
        arange = torch.arange(batch_size).cuda()                        #Return a one-dimensional vector of size args.batch_size, i.e., [0,1,2,3]
        idx = torch.randperm(batch_size).cuda()                         #A numerical sequence obtained by randomly shuffling 0~args.batch_size (0,1,2,3)
        while 0.0 in (idx - arange):
            idx = torch.randperm(batch_size).cuda()

        # Contrastive Loss
        loss_ct = - ang_loss(fc_nir, fc_vis) + \
                  0.1 * F.relu((fc_nir * fc_vis[idx, :]).sum(dim=1) - 0.5).sum() / float(batch_size)    #Equation (19)

        loss = loss_ce + 0.001 * loss_ct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measurement accuracy and recording loss
        prec1, prec5 = accuracy(output.data, label.data, topk=(1, 5))
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        # Output log
        if i % args.print_iter == 0:
            info = "====> Epoch: [{:0>3d}][{:3d}/{:3d}] | ".format(epoch, i, len(train_loader))
            info += "Loss: ce: {:4.3f} ct: {:4.3f} | ".format(loss_ce.item(), loss_ct.item())
            info += "Prec@1: {:4.2f} ({:4.2f}) Prec@5: {:4.2f} ({:4.2f})".format(top1.val, top1.avg, top5.val, top5.avg)
            print(info)




if __name__ == "__main__":
    main()
