import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils import *
import torch.nn.functional as F
import torch_dct as dct
import scipy.stats as st
import torchvision.transforms.functional as TF

class S4ST_transformer():
    def __init__(self, num_block=[2,3], pR=0.9, pAug=1.0, r=1.7, pop_idx=-1):
        super().__init__()
        self.rand_augs = [self.flip, self.contrast, self.brightness, self.saturation, self.hue]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pR = pR
        self.pAug = pAug
        self.num_block = num_block
        self.r = r
        if pop_idx != -1:
            self.rand_augs.pop(pop_idx)
        
    def flip(self, x):
        r = torch.randint(low=0, high=2, size=[1])
        if float(r) == 0.:
            return TF.hflip(x)
        if float(r) == 1.:
            return TF.vflip(x)

    def hue(self, x):
        r = float((torch.rand(1,) * 2 - 1) * 0.5)
        return TF.adjust_hue(x, r)

    def contrast(self, x):
        r = float((torch.rand(1,) * 2 - 1) * 1 + 1)
        return TF.adjust_contrast(x, r)

    def saturation(self, x):
        r = float((torch.rand(1,) * 2 - 1) * 1 + 1)
        return TF.adjust_saturation(x, r)

    def brightness(self, x):
        r = float((torch.rand(1,) * 2 - 1) * 1 + 1)
        return TF.adjust_brightness(x, r)

    def solarize(self, x):
        r =  1 - torch.rand(1).item()
        return TF.solarize(x, r)

    def _base(self, x):
        if torch.rand(1) < self.pR:
            _, _, ori_w, ori_h, = x.shape
            r_sample = ((self.r - 1/self.r) * torch.rand(1,) + 1/self.r).item()
            r_w, r_h = int(math.floor(ori_w * r_sample)), int(math.floor(ori_h * r_sample))
            if r_sample < 1.:
                rescaled = F.interpolate(x, size=[r_w, r_h], mode="bilinear", align_corners=False, antialias=True)
                w_rem = ori_w - r_w
                h_rem = ori_h - r_h
                pad_top = torch.randint(low=0, high=h_rem, size=(1,), dtype=torch.int32)
                pad_bottom = h_rem - pad_top
                pad_left = torch.randint(low=0, high=w_rem, size=(1,), dtype=torch.int32)
                pad_right = w_rem - pad_left
                padded = F.pad(rescaled,[pad_top.item(), pad_bottom.item(),pad_left.item(), pad_right.item()], mode='constant',value=0)
                return padded
            else:
                rescaled = torchvision.transforms.RandomResizedCrop([ori_w, ori_h], scale=((1 / r_sample) ** 2, (1 / r_sample) ** 2), ratio=(ori_h/ori_w, ori_h/ori_w), antialias=True)(x)
                return rescaled
        else:
            return x

    def get_length_with_min_spacing(self, num, total_length, min_spacing=30):
        if num > 1:
            points = np.sort(np.random.uniform(min_spacing, total_length - min_spacing, num-1))
            lengths = np.diff(np.concatenate(([0], points, [total_length])))
            while any(lengths < min_spacing):
                points = np.sort(np.random.uniform(min_spacing, total_length - min_spacing, num-1))
                lengths = np.diff(np.concatenate(([0], points, [total_length])))
            return [0,] + list(points.astype(int)) + [total_length,]
        else:
            return torch.linspace(0, total_length, num + 1, dtype=int).tolist()

    def blocktransform(self, x):
        if torch.rand(1,) < self.pAug:      #_aug
            x = self.rand_augs[np.random.randint(0, len(self.rand_augs), dtype=np.int32)](x)
        _, _, w, h = x.shape
        np.random.shuffle(self.num_block)
        x_axis = self.get_length_with_min_spacing(self.num_block[0], w)
        y_axis = self.get_length_with_min_spacing(self.num_block[1], h)
        x_copy = x.clone()
        for i, idx_x in enumerate(x_axis[1:]):        #_block
            for j, idx_y in enumerate(y_axis[1:]):
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = self._base(x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])
        return x_copy

    def transform(self, x):
        return self.blocktransform(x)

    def scaling(self, x):
        if torch.rand(1) < self.pR:
            _, _, ori_w, ori_h, = x.shape
            r = ((self.r - 1/self.r) * torch.rand(1,) + 1/self.r).item()
            r_w, r_h = int(ori_w * r), int(ori_h * r)
            return F.interpolate(x, size=[r_w, r_h], mode="bilinear", align_corners=False, antialias=True)
        else:
            return x

