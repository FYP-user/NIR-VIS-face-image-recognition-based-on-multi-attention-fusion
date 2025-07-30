import numpy as np
import os, random
from PIL import Image
from collections import defaultdict

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class Dataset(data.Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()

        self.img_root = args.img_root                             #'dataset/CASIA_2.0/'
        self.list_file = args.train_list                          #'dataset/CASIA_2.0/together.csv'

        self.transform = transforms.Compose([                     #对图像进行各种转换操作，并用函数compose将这些转换操作组合起来
            transforms.CenterCrop(128),                           #随机中心裁剪(128*128)
            transforms.ToTensor()                                 #将原始的PILImage格式或者numpy.array格式的数据格式化为可被pytorch快速处理的张量类型
        ])

        self.img_list, self.pair_dict = self.list_reader()

    def __getitem__(self, index):
        img_name, label = self.img_list[index]

        if 'NIR' in img_name:
            img_path = os.path.join(self.img_root, img_name)      #路径拼接文件路径
            nir = Image.open(img_path)                            #读取图片
            nir = self.transform(nir)                             #对图片进行处理

            # paired VIS
            img_name = self.get_pair(label, 'VIS')        #获得配对的 'VIS' 图片
            img_path = os.path.join(self.img_root, img_name)
            vis = Image.open(img_path)
            vis = self.transform(vis)
        elif 'VIS' in img_name:
            img_path = os.path.join(self.img_root, img_name)
            vis = Image.open(img_path)
            vis = self.transform(vis)

            # paired NIR
            img_name = self.get_pair(label, 'NIR')         #获得配对的 'NIR' 图片
            img_path = os.path.join(self.img_root, img_name)
            nir = Image.open(img_path)
            nir = self.transform(nir)

        return {'NIR': nir, 'VIS': vis, 'label': int(label)}

    def __len__(self):
        return len(self.img_list)

    def list_reader(self):
        def dict_profile():                                          #创建一个 dict_profile 字典
            return {'NIR': [], 'VIS': []}

        img_list = []                                                #创建一个 img_list 列表
        pair_dict = defaultdict(dict_profile)                        #defaultdict():dict的子类，可以调用提供默认值的函数,Python中通过Key访问字典，
                                                                     # 当Key不存在时，会引发‘KeyError’异常。为了避免这种情况的发生，
                                                                     # 可以使用collections类中的defaultdict()方法来为字典提供默认值
        with open(self.list_file) as f:
            img_names = f.readlines()                                #每次按行读取整个文件内容，将读取到的内容放到一个列表中，返回list类型
            for img_name in img_names:
                img_name, label = img_name.strip().split(',')        #将空白符删除，并将图片名和标签分割开
                img_list.append((img_name, label))                   #将图片和其标签加入到 img_list 这个列表中

                if 'NIR' in img_name:
                    pair_dict[label]['NIR'].append(img_name)
                elif 'VIS' in img_name:
                    pair_dict[label]['VIS'].append(img_name)

        # find IDs with paired data
        img_list_update = []
        for img_name, label in img_list:
            if (len(pair_dict[label]['NIR']) > 0) and (len(pair_dict[label]['VIS']) > 0):
                img_list_update.append((img_name, label))

        return img_list_update, pair_dict

    def get_pair(self, label, modality):
        img_name = random.choice(self.pair_dict[label][modality])    #随机抽取相同标签相同模态的图片
        return img_name
