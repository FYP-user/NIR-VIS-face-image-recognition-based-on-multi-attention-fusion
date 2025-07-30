import os
import numpy as np
import torch
import torch.nn.functional as F


# 角正交损失函数
def ort_loss(x, y):
    loss = torch.abs((x * y).sum(dim=1)).sum()
    loss = loss / float(x.size(0))
    return loss


# 成对恒等保持损失函数
def ang_loss(x, y):
    loss = (x * y).sum(dim=1).sum()
    loss = loss / float(x.size(0))
    return loss


# 转换为灰度值图像
def rgb2gray(img):
    r, g, b = torch.split(img, 1, dim=1)                                        #将tensor分成块结构，dim是切分维度
    return torch.mul(r, 0.299) + torch.mul(g, 0.587) + torch.mul(b, 0.114)      #消除图像色调和饱和度信息的同时保留亮度实现将将RGB图像或彩色图转换为灰度图像


# 存储模型
def save_checkpoint(model, epoch, name=""):
    if not os.path.exists("model/"):                                            #判断括号里的文件是否存在
        os.makedirs("model/")                                                   #如果不存在则创建目录
    model_path = "model/" + name + "_epoch_{}.pth.tar".format(epoch)            #模型的存放路径
    state = {"epoch": epoch, "state_dict": model.state_dict()}                  #state:模型参数
    torch.save(state, model_path)                                               #保存加载模型参数
    print("checkpoint saved to {}".format(model_path))


# 载入模型
def load_model(model, pretrained):
    weights = torch.load(pretrained)                            #载入的权重参数
    pretrained_dict = weights["state_dict"]
    model_dict = model.state_dict()                             #model_dict:存放训练过程中需要学习的权重和偏置系数
    # 1. filter out unnecessary keys(过滤掉不必要的关键值)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict(改写现有的参数)
    model_dict.update(pretrained_dict)                          #更新参数
    # 3. load the new state dict(载入新的参数)
    model.load_state_dict(model_dict)


# 是否为这个张量计算梯度(Ture为需要，False为不需要)
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# 风格迁移
def assign_adain_params(adain_params, model):                               #adain_params:style_nir/VIS; model:decoder
    for m in model.modules():                                               #递归遍历所有的子模块，包括子模块的子模块等等以此类推
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]                         #计算均值
            std = adain_params[:, m.num_features:2*m.num_features]          #计算标准差
            m.bias = mean.contiguous().view(-1)                             #偏置
            m.weight = std.contiguous().view(-1)                            #权重
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


# 精度计算
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
