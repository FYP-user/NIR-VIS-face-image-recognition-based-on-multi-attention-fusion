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
    args = parser.parse_args()                                          # Return all "add_argument" set in the parser to the args subclass instance.
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids                   # Specify GPU to be used
    cudnn.benchmark = True                                              #

    if not os.path.exists(args.output_path_nir):                        #D etermine if the file in parentheses exists, and create a directory if it doesn't.
        os.makedirs(args.output_path_nir)

    if not os.path.exists(args.output_path_vis):                        #
        os.makedirs(args.output_path_vis)

    # id sampler
    dec = define_ID()                                                   # dec:identity sampler(Fs)
    load_model(dec, args.weights_dec)                                   # Load model and associated weighting parameters
    set_requires_grad([dec], False)                                     #There is no need to compute the gradient for this [dec] tensor
    dec.eval()                                                          # Returns the result of an expression passed into a string. That is: evaluates the string as if it were a valid expression and returns the result of the calculation

    # generator(Dyadic variational generator)
    encoder_nir, encoder_vis, decoder = define_G(input_dim=3, output_dim=3, ndf=32)
    load_model(encoder_nir, args.weights_encoder_nir)                   # Load encoder_nir model and associated weight parameters
    load_model(encoder_vis, args.weights_encoder_vis)                   # Load encoder_vis model and associated weight parameters
    load_model(decoder, args.weights_decoder)                           # Load decoder model and associated weight parameters

    set_requires_grad([encoder_nir, encoder_vis, decoder], False)       # There is no need to compute the gradient for this [encoder_nir, encoder_vis, decoder] tensor
    encoder_nir.eval()                                                  # Returns the result of an expression passed into a string.
    encoder_vis.eval()                                                  # That is: evaluates the string as if it were a valid expression
    decoder.eval()                                                      # and returns the result of the calculation

    # Obtain paired heterogeneous face data and preprocess the data. Use this data for training
    train_loader = torch.utils.data.DataLoader(
        Dataset(args), batch_size=50, shuffle=True, num_workers=8, pin_memory=True)

    img_num = 0
    list_file = open(args.output_path_nir.split("/")[0] + "/img_list.txt", "w")         # Create the file img_list.txt in the directory gen_images.
    for epoch in range(1, 100000):
        for iteration, data in enumerate(train_loader):                                 # iteration:index of training data; data:trained data
            nir = Variable(data["NIR"].cuda())                                          # Here NIR is the Variable parameter form of the NIR attribute data
            vis = Variable(data["VIS"].cuda())                                          # Here VIS is the Variable parameter form of the VIS attribute data

            batch_size = nir.size(0)                                                    # batch_size
            noise = torch.zeros(batch_size, 256).normal_(0, 1).cuda()                   #
                                                                                        #
            id_noise = dec(noise)                                                       #id_noise:Identity representation (𝑓 ̃) extracted from a large number of unpaired VIS images

            z_nir = encoder_nir(nir, "enc")                                             # z_nir:Attribute distribution of near-infrared images(Zn)
            z_vis = encoder_vis(vis, "enc")                                             # z_vis:Distribution of properties of visible light images(Zv)

            style_nir = encoder_nir(z_nir, "style")                                     # style_nir:Style of NIR images
            style_vis = encoder_vis(z_vis, "style")                                     # style_vis:Style of VISimages

            assign_adain_params(style_nir, decoder)                                     # Style migration (NIR style)
            fake_nir = decoder(torch.cat([id_noise, z_nir], dim=1), "NIR")              # fake_nir:Generated NIR images(𝐼 ̃𝑁)

            assign_adain_params(style_vis, decoder)                                     # Style migration (VIS style)
            fake_vis = decoder(torch.cat([id_noise, z_vis], dim=1), "VIS")              #fake_vis:Generated VIS images(𝐼 ̃𝑉)

            # Storing Pictures
            fake_nir = fake_nir.data.cpu().numpy()                                      #cpu
            fake_vis = fake_vis.data.cpu().numpy()                                      #
            for i in range(batch_size):
                img_num = img_num + 1
                list_file.write(str(img_num) + ".jpg" + "\n")                           # Write the image tags to be stored in the file img_list.txt
                print(img_num)

                save_img = fake_nir[i, :, :, :]                                         # Stored NIR images (sliced read data from fake_nir)
                save_img = np.transpose((255 * save_img).astype("uint8"), (1, 2, 0))    #
                                                                                        #
                                                                                        #

                output = Image.fromarray(save_img)                                      # Image.fromarray(): from array to image
                save_name = str(img_num) + ".jpg"                                       # Name of the stored image
                output.save(os.path.join(args.output_path_nir, save_name))              # Store NIR images in a file at path 'gen_images/NIR'

                save_img = fake_vis[i, :, :, :]                                         # Stored VIS images (sliced read data from fake_vis)
                save_img = np.transpose((255 * save_img).astype("uint8"), (1, 2, 0))
                output = Image.fromarray(save_img)
                save_name = str(img_num) + ".jpg"
                output.save(os.path.join(args.output_path_vis, save_name))              # Store VIS images in a file in path 'gen_images/VIS'

                if img_num == 100000:                                                   # Generate 100,000 paired heterogeneous face images
                    print("we have generated 100k paired images")
                    list_file.close()
                    exit(0)




if __name__ == "__main__":
    main()
