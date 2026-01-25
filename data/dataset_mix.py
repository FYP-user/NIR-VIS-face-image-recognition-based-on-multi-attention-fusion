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

        self.img_root = args.img_root_A                                 # Path to Real Data 'dataset/CASIA_2.0/'
        self.img_list = self.list_reader(args.train_list_A)             # 'together.csv'

        self.transform = transforms.Compose([                           # Perform various transformation operations on images
            transforms.RandomCrop(128),                                 # Random Center Crop (128×128)
            transforms.ToTensor()                                       # Convert raw PILImage or numpy.array data formats into tensor types that can be efficiently processed by PyTorch.
        ])

    def __getitem__(self, index):
        img_name, label = self.img_list[index]
        img_path = os.path.join(self.img_root, img_name)                # Concatenate file paths
        img = Image.open(img_path).convert('L')                         # Convert the image to grayscale, with each pixel represented by 8 bits: 0 represents black, 255 represents white, and other values represent different shades of gray.
        img = self.transform(img)                                       # Convert the image

        return {'img': img, 'label': int(label)}

    def __len__(self):
        return len(self.img_list)                                       #

    def list_reader(self, list_file):                                   # Read image
        img_list = []
        with open(list_file, 'r') as f:
            for line in f.readlines():                                  # Read the entire file content line by line, place the read content into a list, and return a list type.
                img_name, label = line.strip().split(',')               # Remove whitespace and separate the image name from the tag.
                img_list.append((img_name, label))                      # Add the image and its label to the list named `img_list`.
        return img_list


class Mix_Dataset(data.Dataset):
    def __init__(self, args):
        super(Mix_Dataset, self).__init__()

        self.img_root_A = args.img_root_A                               # The True Path 'dataset/CASIA_2.0/'
        self.img_root_B = args.img_root_B                               # Path for generating data './gen_images/NIR'

        self.list_file_A = args.train_list_A                            # Actual Data File 'dataset/CASIA_2.0/together.csv'
        self.list_file_B = args.train_list_B                            # Generate data files './gen_images/img_list.txt'

        self.transform = transforms.Compose([                           #Perform various transformation operations on images
            transforms.RandomCrop(128),                                 #
            transforms.ToTensor()                                       #
        ])

        self.img_list_A, self.img_list_B = self.list_reader()           # Read data, where img_list_A is img_list_A_extend.

    def __getitem__(self, index):
        img_name, label = self.img_list_A[index]
        img_path_A = os.path.join(self.img_root_A, img_name)            # Concatenate file paths
        img_A = Image.open(img_path_A).convert('L')                     # Convert the image to grayscale, with each pixel represented by 8 bits: 0 represents black, 255 represents white, and other values represent different shades of gray.
        img_A = self.transform(img_A)                                   # Convert the image

        img_name = self.img_list_B[index]
        img_path_B = os.path.join(self.img_root_B, img_name)            #
        img_B = Image.open(img_path_B).convert('L')                     #
        img_B = self.transform(img_B)                                   #

        img_path_B_pair = img_path_B.replace('gen_images/NIR', 'gen_images/VIS')    # Replace ‘gen_images/NIR’ with ‘gen_images/VIS’.
        img_B_pair = Image.open(img_path_B_pair).convert('L')                       #Convert the image to grayscale
        img_B_pair = self.transform(img_B_pair)                                     #

        return {'img_A': img_A, 'label': int(label),                    # Images and labels that return real data
                'img_B': img_B, 'img_B_pair': img_B_pair}               # Return the generated data for the pairing

    def __len__(self):
        return len(self.img_list_B)                                     #

    def list_reader(self):
        img_list_A = []
        with open(self.list_file_A) as f:
            img_names = f.readlines()                                   # Read the entire file content line by line, place the read content into a list, and return a list type.
            for img_name in img_names:
                img_name, label = img_name.strip('\n').split(',')       # Remove whitespace and separate the image name from the tag.
                img_list_A.append((img_name, label))                    # Add the image and its label to the list img_list_A.

        img_list_B = []
        with open(self.list_file_B) as f:
            img_names = f.readlines()                                   #
            for img_name in img_names:
                img_name = img_name.strip()                             #
                img_list_B.append(img_name)                             #

        rep_num = int(len(img_list_B) / len(img_list_A))                # Computational data is several times the volume of real data. rep_num
        img_list_A_extend = []                                          # img_list_A_extend
        for i in range(rep_num):
            img_list_A_extend.extend(img_list_A)                        # Append img_list_A to img_list_A_extend rep_num times.

        res_num = len(img_list_B) - rep_num * len(img_list_A)           # Calculate how much more the generated data exceeds the actual data. rep_num
        for i in range(res_num):
            img_list_A_extend.append(img_list_A[i])                     # Add the first rep_num entries from img_list_A to img_list_A_extend.
                                                                        # At this point, img_list_A_extend and img_list_B contain the same number of entries.
        return img_list_A_extend, img_list_B
