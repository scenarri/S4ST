import torch
import numpy as np
import cv2
import os
from transformations.dct import dct_2d, idct_2d
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torch_dct as dct
import torchvision.transforms as transforms
from scipy import stats as st
import torchvision
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def DI(x, resize_rate_H=1.1, diversity_prob=0.7):
    if torch.rand(1) < diversity_prob:
        ori_size = x.shape[-1]
        img_size = int(ori_size * 1)
        img_resize = int(ori_size * resize_rate_H)

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode="bilinear", align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], mode='constant', value=0)
        return padded
    else:
        return x

def RDI(x, resize_rate_H=1.1, diversity_prob=0.7):
    if torch.rand(1) < diversity_prob:
        ori_size = x.shape[-1]
        img_size = int(ori_size * 1)
        img_resize = int(ori_size * resize_rate_H)

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode="bilinear", align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
                       mode='constant', value=0)
        return transforms.Resize((ori_size, ori_size), interpolation=InterpolationMode.NEAREST)(padded)
    else:
        return x

def ukern(kernlen=15):
    kernel = np.ones((kernlen, kernlen)) * 1.0 / (kernlen * kernlen)
    return kernel

def lkern(kernlen=15):
    kern1d = 1 - np.abs(
        np.linspace((-kernlen + 1) / 2, (kernlen - 1) / 2, kernlen)
        / (kernlen + 1)
        * 2
    )
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def gkern(kernlen=15, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def kernel_generation(len_kernel=5, nsig=3, kernel_name="gaussian"):
    if kernel_name == "gaussian":
        kernel = gkern(len_kernel, nsig).astype(np.float32)
    elif kernel_name == "linear":
        kernel = lkern(len_kernel).astype(np.float32)
    elif kernel_name == "uniform":
        kernel = ukern(len_kernel).astype(np.float32)
    else:
        raise NotImplementedError

    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    return stack_kernel

########### SSA
def SSA_aug(adv_X):
    gauss = torch.randn_like(adv_X).cuda() * (16/255)
    x_dct = dct_2d(adv_X + gauss)
    mask = (torch.rand_like(adv_X) * 2 * 0.5 + 1 - 0.5).cuda()
    x_idct = idct_2d(x_dct * mask)
    return x_idct

