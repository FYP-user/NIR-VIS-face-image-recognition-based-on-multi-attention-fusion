import numpy as np
import os, random
from PIL import Image
from collections import defaultdict

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class Real_Dataset(data.Dataset):
    def __init__(self, args):
        super(Real_Dataset, self).__init__()

        self.img_root = args.img_root_A                                 #真实数据的路径 'dataset/CASIA_2.0/'
        self.img_list = self.list_reader(args.train_list_A)             #真实数据的文件 'together.csv'

        self.transform = transforms.Compose([                           #对图像进行各种转换操作，并用函数compose将这些转换操作组合起来
            transforms.RandomCrop(128),                                 #随机中心裁剪(128*128)
            transforms.ToTensor()                                       #将原始的PILImage格式或者numpy.array格式的数据格式化为可被pytorch快速处理的张量类型
        ])

    def __getitem__(self, index):
        img_name, label = self.img_list[index]
        img_path = os.path.join(self.img_root, img_name)                #路径拼接文件路径
        img = Image.open(img_path).convert('L')                         #将图片转换为灰度图像，每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度
        img = self.transform(img)                                       #对图片进行处理

        return {'img': img, 'label': int(label)}

    def __len__(self):
        return len(self.img_list)                                       #返回 img_list 的表长

    def list_reader(self, list_file):                                   #读取文件
        img_list = []
        with open(list_file, 'r') as f:
            for line in f.readlines():                                  #每次按行读取整个文件内容，将读取到的内容放到一个列表中，返回list类型
                img_name, label = line.strip().split(',')               #将空白符删除，并将图片名和标签分割开
                img_list.append((img_name, label))                      #将图片和其标签加入到 img_list 这个列表中
        return img_list


class Mix_Dataset(data.Dataset):
    def __init__(self, args):
        super(Mix_Dataset, self).__init__()

        self.img_root_A = args.img_root_A                               #真实数据的路径 'dataset/CASIA_2.0/'
        self.img_root_B = args.img_root_B                               #生成数据的路径 './gen_images/NIR'

        self.list_file_A = args.train_list_A                            #真实数据文件 'dataset/CASIA_2.0/together.csv'
        self.list_file_B = args.train_list_B                            #生成数据文件 './gen_images/img_list.txt'

        self.transform = transforms.Compose([                           #对图像进行各种转换操作，并用函数compose将这些转换操作组合起来
            transforms.RandomCrop(128),                                 #随机中心裁剪(128*128)
            transforms.ToTensor()                                       #将原始的PILImage格式或者numpy.array格式的数据格式化为可被pytorch快速处理的张量类型
        ])

        self.img_list_A, self.img_list_B = self.list_reader()           #读取数据，这里 img_list_A 为 img_list_A_extend

    def __getitem__(self, index):
        img_name, label = self.img_list_A[index]
        img_path_A = os.path.join(self.img_root_A, img_name)            #路径拼接文件路径
        img_A = Image.open(img_path_A).convert('L')                     #将图片转换为灰度图像，每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度
        img_A = self.transform(img_A)                                   #对图片进行处理

        img_name = self.img_list_B[index]
        img_path_B = os.path.join(self.img_root_B, img_name)            #路径拼接文件路径
        img_B = Image.open(img_path_B).convert('L')                     #将图片转换为灰度图像，每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度
        img_B = self.transform(img_B)                                   #对图片进行处理

        img_path_B_pair = img_path_B.replace('gen_images/NIR', 'gen_images/VIS')    #'gen_images/NIR' 替换成 'gen_images/VIS'
        img_B_pair = Image.open(img_path_B_pair).convert('L')                       #将图片转换为灰度图像，每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度
        img_B_pair = self.transform(img_B_pair)                                     #对图片进行处理

        return {'img_A': img_A, 'label': int(label),                    #返回真实数据的图像和标签
                'img_B': img_B, 'img_B_pair': img_B_pair}               #返回配对的生成数据

    def __len__(self):
        return len(self.img_list_B)                                     #返回 img_list_B 的表长

    def list_reader(self):
        img_list_A = []
        with open(self.list_file_A) as f:
            img_names = f.readlines()                                   #每次按行读取整个文件内容，将读取到的内容放到一个列表中，返回list类型
            for img_name in img_names:
                img_name, label = img_name.strip('\n').split(',')       #将空白符删除，并将图片名和标签分割开
                img_list_A.append((img_name, label))                    #将图片和其标签加入到 img_list_A 这个列表中

        img_list_B = []
        with open(self.list_file_B) as f:
            img_names = f.readlines()                                   #每次按行读取整个文件内容，将读取到的内容放到一个列表中，返回list类型
            for img_name in img_names:
                img_name = img_name.strip()                             #将空白符删除，并将图片名和标签分割开
                img_list_B.append(img_name)                             #将图片和其标签加入到 img_list_B 这个列表中

        rep_num = int(len(img_list_B) / len(img_list_A))                #计算生成数据是真实数据的数量的倍数 rep_num
        img_list_A_extend = []                                          #创建一个列表 img_list_A_extend
        for i in range(rep_num):
            img_list_A_extend.extend(img_list_A)                        #将 img_list_A 重复 rep_num 次拼接到 img_list_A_extend 中

        res_num = len(img_list_B) - rep_num * len(img_list_A)           #计算生成数据比真实数据多多少 rep_num
        for i in range(res_num):
            img_list_A_extend.append(img_list_A[i])                     #将 img_list_A 中前 rep_num 个数据添加到 img_list_A_extend 中去
                                                                        #此时 img_list_A_extend 和 img_list_B 包含的数据数量一样
        return img_list_A_extend, img_list_B
