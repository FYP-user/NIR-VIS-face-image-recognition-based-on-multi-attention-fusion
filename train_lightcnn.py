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
    args = parser.parse_args()                                          #把parser中设置的所有"add_argument"给返回到args子类实例当中
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids                   #指定要使用的显卡
    cudnn.benchmark = True                                              #增加程序的运行效率
    cudnn.enabled = True                                                #让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速

    random.seed(args.seed)                                              #random.seed():初始化随机数生成器,当面对一个随机程序的时候，只要我们的运行环境一致（保证伪随机数生成程序一样），而我们设定的随机种子一样的话，那么我们就可以复现结果
    np.random.seed(args.seed)                                           #
    torch.manual_seed(args.seed)                                        #设置CPU生成随机数的种子，方便下次复现实验结果
    torch.cuda.manual_seed(args.seed)                                   #设置GPU生成随机数的种子，方便下次复现实验结果

    # lightcnn
    model = LightCNN_29v2(num_classes=args.num_classes)                 #模型为 lightcnn

    # 载入预训练模型
    if args.pre_epoch:
        print('load pretrained model of epoch %d' % args.pre_epoch)
        load_model(model, "./model/lightCNN_epoch_%d.pth.tar" % args.pre_epoch)
    else:
        print("=> loading pretrained lightcnn '{}'".format(args.weights_lightcnn))
        load_model(model, args.weights_lightcnn)

    # 载入训练的真实数据
    train_loader_real = torch.utils.data.DataLoader(
        Real_Dataset(args), batch_size=2*args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # 载入训练的混合数据(生成的和真实的数据)
    train_loader_mix = torch.utils.data.DataLoader(
        Mix_Dataset(args), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss().cuda()

    # 第一阶段：对最后一个fc2参数进行模型预训练
    params_pretrain = []
    for name, value in model.named_parameters():
        if "fc2_" in name:
            params_pretrain += [{"params": value, "lr": 1 * args.lr}]

    # 优化程序(随机梯度)
    optimizer_pretrain = torch.optim.SGD(params_pretrain, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, 5):
        pre_train(train_loader_real, model, criterion, optimizer_pretrain, epoch)
        save_checkpoint(model, epoch, "LightCNN_pretrain")

    '''
    第二阶段：全网模型微调
    '''
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start_epoch = args.pre_epoch + 1
    for epoch in range(start_epoch, args.epochs + 1):
        adjust_learning_rate(args.lr, args.step_size, optimizer, epoch)
        train(train_loader_mix, model, criterion, optimizer, epoch)
        save_checkpoint(model, epoch, args.save_name)


# 对最后一个fc2参数进行预训练
def pre_train(train_loader, model, criterion, optimizer, epoch):
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()               #启用 Batch Normalization 和 Dropout，保证BN层能够用到每一批数据的均值和方差，随机取一部分网络连接来训练更新参数
    for i, data in enumerate(train_loader):                             #i:训练数据的索引；data:训练的数据
        input = Variable(data["img"].cuda())                            #这里 input 是 img 的Variable参数形式
        label = Variable(data["label"].cuda())                          #这里 label 是 label 的Variable参数形式
        batch_size = input.size(0)                                      #batch_size 为 input 的行数

        if batch_size < 2*args.batch_size:
            continue

        output = model(input)[0]                                        #输出的身份表示
        loss = criterion(output, label)                                 #交叉熵损失函数
        optimizer.zero_grad()                                           #将梯度归零
        loss.backward()                                                 #反向传播计算得到每个参数的梯度值
        optimizer.step()                                                #通过梯度下降执行一步参数更新

        # 测量精度和记录损失
        prec1, prec5 = accuracy(output.data, label.data, topk=(1, 5))
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        # 输出日志
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
        # 真实数据
        input_real = Variable(data["img_A"].cuda())                     #这里 input_real 是 img_A 的Variable参数形式
        label = Variable(data["label"].cuda())                          #这里 label 是 label 的Variable参数形式

        # 生成数据
        fake_nir = Variable(data["img_B"].cuda())                       #这里 fake_nir 是 img_B 的Variable参数形式
        fake_vis = Variable(data["img_B_pair"].cuda())                  #这里 fake_vis 是 img_B_pair 的Variable参数形式

        batch_size = input_real.size(0)                                 # batch_size 为 input_real 的行数
        if batch_size < args.batch_size:
            continue

        output = model(input_real)[0]                                   #输出的身份表示
        loss_ce = criterion(output, label)                              #计算交叉熵损失函数

        fc_nir = model(fake_nir)[1]
        fc_vis = model(fake_vis)[1]

        # 为负对编制随机指数
        arange = torch.arange(batch_size).cuda()                        #返回一个一维向量,其大小为 args.batch_size 即[0,1,2,3]
        idx = torch.randperm(batch_size).cuda()                         #将 0~args.batch_size（0,1,2,3）随机打乱后获得的数字序列
        while 0.0 in (idx - arange):                                    # arange 与 idx 的每个对应位不能相同
            idx = torch.randperm(batch_size).cuda()

        # 对比损失
        loss_ct = - ang_loss(fc_nir, fc_vis) + \
                  0.1 * F.relu((fc_nir * fc_vis[idx, :]).sum(dim=1) - 0.5).sum() / float(batch_size)    #式(19)

        loss = loss_ce + 0.001 * loss_ct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 测量精度和记录损失
        prec1, prec5 = accuracy(output.data, label.data, topk=(1, 5))
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        # 输出日志
        if i % args.print_iter == 0:
            info = "====> Epoch: [{:0>3d}][{:3d}/{:3d}] | ".format(epoch, i, len(train_loader))
            info += "Loss: ce: {:4.3f} ct: {:4.3f} | ".format(loss_ce.item(), loss_ct.item())
            info += "Prec@1: {:4.2f} ({:4.2f}) Prec@5: {:4.2f} ({:4.2f})".format(top1.val, top1.avg, top5.val, top5.avg)
            print(info)




if __name__ == "__main__":
    main()
