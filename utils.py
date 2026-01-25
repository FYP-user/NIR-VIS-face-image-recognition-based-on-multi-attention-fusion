import os
import numpy as np
import torch
import torch.nn.functional as F


# Orthogonal Angle Loss Function
def ort_loss(x, y):
    loss = torch.abs((x * y).sum(dim=1)).sum()
    loss = loss / float(x.size(0))
    return loss


# Pairwise Identity Preservation Loss Function
def ang_loss(x, y):
    loss = (x * y).sum(dim=1).sum()
    loss = loss / float(x.size(0))
    return loss


# Convert to grayscale image
def rgb2gray(img):
    r, g, b = torch.split(img, 1, dim=1)                                        #Split the tensor into block structures, where dim is the dimension to split.
    return torch.mul(r, 0.299) + torch.mul(g, 0.587) + torch.mul(b, 0.114)      #Convert RGB images or color maps to grayscale images by removing color hue and saturation information while retaining brightness.


# Storage Model
def save_checkpoint(model, epoch, name=""):
    if not os.path.exists("model/"):                                            #Check if the file in the parentheses exists
        os.makedirs("model/")                                                   #Create the directory if it does not exist
    model_path = "model/" + name + "_epoch_{}.pth.tar".format(epoch)            #Model storage path
    state = {"epoch": epoch, "state_dict": model.state_dict()}                  #state:Model parameters
    torch.save(state, model_path)                                               #Save and load model parameters
    print("checkpoint saved to {}".format(model_path))


# Load Model
def load_model(model, pretrained):
    weights = torch.load(pretrained)                            #Loaded weight parameters
    pretrained_dict = weights["state_dict"]
    model_dict = model.state_dict()                             #model_dict:Store the weights and bias coefficients that need to be learned during training
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)                          #Update parameters
    # 3. load the new state dict
    model.load_state_dict(model_dict)


# Whether to compute gradients for this tensor (True means needed, False means not needed)
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# Style Transfer
def assign_adain_params(adain_params, model):                               #adain_params:style_nir/VIS; model:decoder
    for m in model.modules():                                               #Recursively traverse all submodules, including the submodules of submodules, and so on.
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]                         #Calculate the mean
            std = adain_params[:, m.num_features:2*m.num_features]          #Calculate the standard deviation
            m.bias = mean.contiguous().view(-1)                             #Bias
            m.weight = std.contiguous().view(-1)                            #Weight
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


# Accuracy Calculation
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(lr, step, optimizer, epoch):
    scale = 0.457305051927326
    lr = lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale

def Rank(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def FAR1(output, label, label_nums):
    count = 0
    similarity = ort_loss(output, label)
    if similarity > 0.999:
        count += 1
    return count/label_nums

def FAR2(output, label, label_nums):
    count = 0
    similarity = ort_loss(output, label)
    if similarity > 0.99:
        count += 1
    return count/label_nums

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
