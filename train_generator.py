import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from utils import *
from network.ID_net import define_ID
from network.G_net import define_G
from network.lightcnn import LightCNN_29v2
from data.dataset import Dataset


parser = argparse.ArgumentParser()           #创建解析器

parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--pre_epoch', default=0, type=int, help='train from previous model')

parser.add_argument('--print_iter', default=20, type=int, help='print frequency')
parser.add_argument('--save_epoch', default=1, type=int)
parser.add_argument('--output_path', default='./results', type=str)

parser.add_argument('--weights_lightcnn', default='./pre_train/LightCNN_29Layers_V2_checkpoint.pth.tar', type=str)
parser.add_argument('--weights_dec', default='./pre_train/dec_epoch_45.pth.tar', type=str, help='dec is the identity sampler')
parser.add_argument('--img_root',  default='dataset/CASIA_2.0/', type=str)
parser.add_argument('--train_list', default='dataset/CASIA_2.0/together.csv', type=str)


def main():
    global args                                                                    #args全局变量
    args = parser.parse_args()                                                     #把parser中设置的所有"add_argument"给返回到args子类实例当中
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids                              #指定要使用的显卡
    cudnn.benchmark = True                                                         #增加程序的运行效率

    if not os.path.exists(args.output_path):                                       #判断括号里的文件是否存在，如果不存在则创建目录
        os.makedirs(args.output_path)

    # lightcnn(预训练识别网络F)
    LightCNN = LightCNN_29v2(is_train=False)
    print("=> loading pretrained lightcnn '{}'".format(args.weights_lightcnn))
    load_model(LightCNN, args.weights_lightcnn)                                     #载入模型以及相关权重参数
    set_requires_grad([LightCNN], False)                           #不需要为这个[LightCNN]计算梯度
    LightCNN.eval()                                                   #返回传入字符串的表达式的结果。就是说：将字符串当成有效的表达式来求值并返回计算结果

    # id sampler 获取VIS的身份信息
    dec = define_ID()                                                                #dec:身份采样器(Fs)
    print("=> loading pretrained identity sampler '{}'".format(args.weights_dec))
    load_model(dec, args.weights_dec)                                                #载入模型以及相关权重参数
    set_requires_grad([dec], False)                                 #不需要为这个[dec]张量计算梯度
    dec.eval()                                                        #返回传入字符串的表达式的结果。就是说：将字符串当成有效的表达式来求值并返回计算结果

    # generator(对偶变分发生器)
    encoder_nir, encoder_vis, decoder = define_G(input_dim=3, output_dim=3, ndf=32)

    # load pretrained model# 载入预训练的模型
    if args.pre_epoch:
        print("load pretrained model %d" % args.pre_epoch)
        load_model(encoder_nir, "./model/encoder_nir_epoch_%d.pth.tar" % args.pre_epoch)       #载入 encoder_nir 模型以及相关权重参数
        load_model(encoder_vis, "./model/encoder_vis_epoch_%d.pth.tar" % args.pre_epoch)       #载入 encoder_vis 模型以及相关权重参数
        load_model(decoder, "./model/decoder_epoch_%d.pth.tar" % args.pre_epoch)               #载入 decoder 模型以及相关权重参数

    # dataset   获取配对的异构人脸数据，并对数据进行预处理。用这些数据来进行训练
    train_loader = torch.utils.data.DataLoader(
        Dataset(args), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # optimizer# 构造一个优化器对象optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数
    # parameters()会返回一个生成器（迭代器），生成器每次生成的是Tensor类型的数据，这些数据都是模型的参数
    optimizer = optim.Adam(list(encoder_nir.parameters()) + list(encoder_vis.parameters()) +      #lr:学习率
                           list(decoder.parameters()), lr=args.lr, betas=(0.5, 0.999))            #betas:权重衰减

    # criterion
    criterionPix = torch.nn.L1Loss().cuda()                                                     #绝对值误差损失函数，即L1损失函数

    # train  # 训练对偶变分发生器
    start_epoch = args.pre_epoch + 1
    for epoch in range(start_epoch, args.epochs + 1):                         # 1 <= epoch < 6,总共5个epoch

        # creat random index  # 编制随机指数
        arange = torch.arange(args.batch_size).cuda()                         #返回一个一维向量,其大小为 args.batch_size 即[0,1,2,3]
        idx = torch.randperm(args.batch_size).cuda()                          #将 0~args.batch_size（0,1,2,3）随机打乱后获得的数字序列
        while 0.0 in (idx - arange):                                          # arange 与 idx 的每个对应位不能相同
            idx = torch.randperm(args.batch_size).cuda()

        for iteration, data in enumerate(train_loader, start=1):              #iteration:训练数据的索引；data:训练的数据
            # get data
            # Variable是一种可以不断变化的变量，符合反向传播，参数更新的属性。pytorch的Variable是一个存放会变化值的地理位置，
            # 里面的值会不停变化，pytorch都是由tensor计算的，而tensor里面的参数是Variable形式
            nir = Variable(data["NIR"].cuda())                                #这里 NIR 是NIR属性数据的Variable参数形式
            vis = Variable(data["VIS"].cuda())                                #这里 VIS 是VIS属性数据的Variable参数形式

            batch_size = nir.size(0)                                          # batch_size 为 NIR 的行数
            if batch_size < args.batch_size:                                  # args.batch_size 为4
                continue

            id_vis = LightCNN(rgb2gray(vis))                                  #获取NIR-VIS的身份表示(𝑓)
            noise = torch.zeros(batch_size, 256).normal_(0, 1).cuda()   #torch.zeros():返回一个形状为(batch_size, 256)，里面的每一个值都是0的tensor
                                                                        # normal(0, 1):该函数返回从单独的正态分布中提取的随机数的张量
            id_noise = dec(noise)                                       #id_noise:从大量未配对的VIS图像提取的身份表示(𝑓 ̃)

            # forward
            z_nir = encoder_nir(nir, "enc")                             #z_nir:近红外图像的属性分布(Zn)
            z_vis = encoder_vis(vis, "enc")                             #z_vis:可见光图像的属性分布(Zv)

            style_nir = encoder_nir(z_nir, "style")                     #style_nir:NIR图像的风格
            style_vis = encoder_vis(z_vis, "style")
            # style_vis:VIS图像的风格

            assign_adain_params(style_nir, decoder)                                          #风格迁移(NIR风格)
            rec_nir = decoder(torch.cat([id_vis, z_nir], dim=1), "NIR")              #rec_nir:重构的NIR图像(𝐼 ̂𝑁)
                                                                                        #torch.cat():把多个tensor进行拼接
            rec_nir_idx = decoder(torch.cat([id_vis[idx, :], z_nir], dim=1), "NIR")   #rec_nir_idx:选择性获取数据重构图像
            fake_nir = decoder(torch.cat([id_noise, z_nir], dim=1), "NIR")            #fake_nir:生成的NIR图像(𝐼 ̃𝑁)

            assign_adain_params(style_vis, decoder)                                          #风格迁移(VIS风格)
            rec_vis = decoder(torch.cat([id_vis, z_vis], dim=1), "VIS")               #rec_vis:重构的VIS图像(𝐼 ̂𝑉)
            rec_vis_idx = decoder(torch.cat([id_vis[idx, :], z_vis], dim=1), "VIS")   #rec_vis_idx:选择性获取数据重构图像
            fake_vis = decoder(torch.cat([id_noise, z_vis], dim=1), "VIS")            #fake_vis:生成的VIS图像(𝐼 ̃𝑉)

            # orthogonal loss  # 角正交损失函数
            loss_ort = 50 * (ort_loss(z_nir, id_vis) + ort_loss(z_vis, id_vis))             #式(2)

            # pixel loss       # 分布学习损失函数
            loss_pix = 100 * ((criterionPix(rec_nir, nir) + criterionPix(rec_vis, vis)) +           #式(4)
                               0.1 * (criterionPix(rec_nir_idx, nir) + criterionPix(rec_vis_idx, vis)) +
                               0.1 * (criterionPix(fake_nir, nir) + criterionPix(fake_vis, vis)))   #式(13)

            # identity preserving loss  # 成对恒等保持损失
            id_nir_rec = LightCNN(rgb2gray(rec_nir))                          #id_nir_rec:重构NIR图像的身份表示(𝑓 ̂𝑁)
            id_vis_rec = LightCNN(rgb2gray(rec_vis))                          #id_vis_rec:重构VIS图像的身份表示(𝑓 ̂𝑉)
            id_nir_fake = LightCNN(rgb2gray(fake_nir))                        #id_nir_fake:生成的NIR图像的身份表示(𝑓 ̃𝑁)
            id_vis_fake = LightCNN(rgb2gray(fake_vis))                        #id_vis_fake:生成的VIS图像的身份表示(𝑓 ̃𝑉)

            real_ang_rec = ang_loss(id_nir_rec, id_vis) + ang_loss(id_vis_rec, id_vis)         #式(7),使训练过程更加稳定
            real_ang_pair = ang_loss(id_nir_rec, id_vis_rec)                                   #式(6),重构的一对图像身份一致性


            fake_ang_rec = ang_loss(id_nir_fake, id_noise) + ang_loss(id_vis_fake, id_noise)   #式(13),生成的图像身份一致性
            fake_ang_pair = ang_loss(id_nir_fake, id_vis_fake)                                 #式(14),生成的一对图像身份一致性

            loss_ip = - 0.1 * (real_ang_rec + 0.05 * real_ang_pair + fake_ang_rec + 0.05 * fake_ang_pair)

            # all losses # 总体损失
            loss = loss_ort + loss_pix + loss_ip

            optimizer.zero_grad()           #将梯度归零
            loss.backward()                 #反向传播计算得到每个参数的梯度值
            optimizer.step()                #通过梯度下降执行一步参数更新

            # print log    # 输出日志
            if iteration % args.print_iter == 0:
                info = "====> Epoch[{}][{}/{}] | ".format(epoch, iteration, len(train_loader))
                info += "Loss: pix: {:4.2f} ort: {:4.2f} | Ang-real rec: {:4.2f} pair: {:4.2f} | Ang-fake rec: {:4.2f} pair: {:4.2f}".format(
                    loss_pix.item(), loss_ort.item(), real_ang_rec.item(), real_ang_pair.item(), fake_ang_rec.item(), fake_ang_pair.item())
                print(info)

            # save images   # 存储图片
            if iteration % 500 == 0:
                vutils.save_image(torch.cat([nir, rec_nir, rec_nir_idx, fake_nir, nir[idx, :],
                                             vis, rec_vis, rec_vis_idx, fake_vis, vis[idx, :]], dim=0).data,
                                  "{}/Epoch_{:03d}_Iter_{:06d}_img.png".format(args.output_path, epoch, iteration), nrow=batch_size)

        # save model     # 存储模型
        if epoch % args.save_epoch == 0:                  #每个 epoch 结束存储模型
            save_checkpoint(encoder_nir, epoch, "encoder_nir")
            save_checkpoint(encoder_vis, epoch, "encoder_vis")
            save_checkpoint(decoder, epoch, "decoder")




if __name__ == "__main__":
    main()
