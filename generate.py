import os
import time
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable

from utils import *
from network.ID_net import define_ID
from network.G_net import define_G
from data.dataset import Dataset


parser = argparse.ArgumentParser()

parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--output_path_nir', default='gen_images/NIR', type=str)
parser.add_argument('--output_path_vis', default='gen_images/VIS', type=str)

parser.add_argument('--weights_dec', default='./pre_train/dec_epoch_45.pth.tar', type=str, help='dec is the identity sampler')
parser.add_argument('--weights_encoder_nir', default='./model/encoder_nir_epoch_5.pth.tar', type=str)
parser.add_argument('--weights_encoder_vis', default='./model/encoder_vis_epoch_5.pth.tar', type=str)
parser.add_argument('--weights_decoder', default='./model/decoder_epoch_5.pth.tar', type=str)

parser.add_argument('--img_root',  default='dataset/LFW/NIR/test_images/', type=str)
parser.add_argument('--train_list', default='dataset/LFW/NIR/csv_files/test_dataset.csv', type=str)


def main():
    global opt, model
    args = parser.parse_args()                                          #把parser中设置的所有"add_argument"给返回到args子类实例当中
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids                   #指定要使用的显卡
    cudnn.benchmark = True                                              #增加程序的运行效率

    if not os.path.exists(args.output_path_nir):                        #判断括号里的文件是否存在，如果不存在则创建目录
        os.makedirs(args.output_path_nir)

    if not os.path.exists(args.output_path_vis):                        #判断括号里的文件是否存在，如果不存在则创建目录
        os.makedirs(args.output_path_vis)

    # id sampler
    dec = define_ID()                                                   #dec:身份采样器(Fs)
    load_model(dec, args.weights_dec)                                   #载入模型以及相关权重参数
    set_requires_grad([dec], False)                                     #不需要为这个[dec]张量计算梯度
    dec.eval()                                                          #返回传入字符串的表达式的结果。就是说：将字符串当成有效的表达式来求值并返回计算结果

    # generator(对偶变分发生器)
    encoder_nir, encoder_vis, decoder = define_G(input_dim=3, output_dim=3, ndf=32)
    load_model(encoder_nir, args.weights_encoder_nir)                   #载入 encoder_nir 模型以及相关权重参数
    load_model(encoder_vis, args.weights_encoder_vis)                   #载入 encoder_vis 模型以及相关权重参数
    load_model(decoder, args.weights_decoder)                           #载入 decoder 模型以及相关权重参数

    set_requires_grad([encoder_nir, encoder_vis, decoder], False)       #不需要为这个[encoder_nir, encoder_vis, decoder]张量计算梯度
    encoder_nir.eval()                                                  #返回传入字符串的表达式的结果。就是说：将字符串当成有效的表达式来求值并返回计算结果
    encoder_vis.eval()                                                  #返回传入字符串的表达式的结果。就是说：将字符串当成有效的表达式来求值并返回计算结果
    decoder.eval()                                                      #返回传入字符串的表达式的结果。就是说：将字符串当成有效的表达式来求值并返回计算结果

    #获取配对的异构人脸数据，并对数据进行预处理。用这些数据来进行训练
    train_loader = torch.utils.data.DataLoader(
        Dataset(args), batch_size=50, shuffle=True, num_workers=8, pin_memory=True)

    img_num = 0
    list_file = open(args.output_path_nir.split("/")[0] + "/img_list.txt", "w")         #在目录 gen_images 里创建 img_list.txt 文件
    for epoch in range(1, 100000):
        for iteration, data in enumerate(train_loader):                                 #iteration:训练数据的索引；data:训练的数据
            nir = Variable(data["NIR"].cuda())                                          #这里 NIR 是NIR属性数据的Variable参数形式
            vis = Variable(data["VIS"].cuda())                                          #这里 VIS 是VIS属性数据的Variable参数形式

            batch_size = nir.size(0)                                                    # batch_size 为 NIR 的行数
            noise = torch.zeros(batch_size, 256).normal_(0, 1).cuda()                   #torch.zeros():返回一个形状为(batch_size, 256)，里面的每一个值都是0的tensor
                                                                                        #normal(0, 1):该函数返回从单独的正态分布中提取的随机数的张量
            id_noise = dec(noise)                                                       #id_noise:从大量未配对的VIS图像提取的身份表示(𝑓 ̃)

            z_nir = encoder_nir(nir, "enc")                                             #z_nir:近红外图像的属性分布(Zn)
            z_vis = encoder_vis(vis, "enc")                                             #z_vis:可见光图像的属性分布(Zv)

            style_nir = encoder_nir(z_nir, "style")                                     #style_nir:NIR图像的风格
            style_vis = encoder_vis(z_vis, "style")                                     #style_vis:VIS图像的风格

            assign_adain_params(style_nir, decoder)                                     #风格迁移(NIR风格)
            fake_nir = decoder(torch.cat([id_noise, z_nir], dim=1), "NIR")              #fake_nir:生成的NIR图像(𝐼 ̃𝑁)

            assign_adain_params(style_vis, decoder)                                     #风格迁移(VIS风格)
            fake_vis = decoder(torch.cat([id_noise, z_vis], dim=1), "VIS")              #fake_vis:生成的VIS图像(𝐼 ̃𝑉)

            # 存储图片
            fake_nir = fake_nir.data.cpu().numpy()                                      #cpu():将变量放在cpu上
            fake_vis = fake_vis.data.cpu().numpy()                                      #numpy():将tensor转换为numpy
            for i in range(batch_size):
                img_num = img_num + 1
                list_file.write(str(img_num) + ".jpg" + "\n")                           #在文件 img_list.txt 中写入要存储的图片标签
                print(img_num)

                save_img = fake_nir[i, :, :, :]                                         #存储的NIR图像(从 fake_nir 中切片读取数据)
                save_img = np.transpose((255 * save_img).astype("uint8"), (1, 2, 0))    #transpose():函数的作用就是调换数组的行列值的索引值，类似于求矩阵的转置
                                                                                        #transpose()函数的第二个参数(1,2,0)就是改变索引值的地方
                                                                                        #astype():转换numpy数组的数据类型

                output = Image.fromarray(save_img)                                      #Image.fromarray():实现array到image的转换
                save_name = str(img_num) + ".jpg"                                       #存储图片的名称
                output.save(os.path.join(args.output_path_nir, save_name))              #将NIR图片存储到路径'gen_images/NIR'的文件中

                save_img = fake_vis[i, :, :, :]                                         #存储的VIS图像(从 fake_vis 中切片读取数据)
                save_img = np.transpose((255 * save_img).astype("uint8"), (1, 2, 0))
                output = Image.fromarray(save_img)
                save_name = str(img_num) + ".jpg"
                output.save(os.path.join(args.output_path_vis, save_name))              #将VIS图片存储到路径'gen_images/VIS'的文件中

                if img_num == 100000:                                                   #生成10万张配对的异构人脸图像
                    print("we have generated 100k paired images")
                    list_file.close()
                    exit(0)




if __name__ == "__main__":
    main()
