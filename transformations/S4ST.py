import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils import *
import torch.nn.functional as F
import torch_dct as dct
import scipy.stats as st

class S4ST_transformer():
    def __init__(self, num_block=[2,3], pR=0.9, pAug=1.0, r=1.9):
        super().__init__()
        self.rand_augs = [self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip,
                   self.rotate180, self.scale, self.add_noise, self.dct, self.drop_out]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pR = pR
        self.pAug = pAug
        self.num_block = num_block
        self.r = r
    def vertical_shift(self, x):
        _, _, w, _ = x.shape
        step = np.random.randint(low=0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)

    def horizontal_shift(self, x):
        _, _, _, h = x.shape
        step = np.random.randint(low=0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        return x.flip(dims=(2,))

    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def rotate180(self, x):
        return x.rot90(k=2, dims=(2, 3))

    def scale(self, x):
        return torch.rand(1)[0] * x

    def dct(self, x):
        """
        Discrete Fourier Transform
        """
        dctx = dct.dct_2d(x)  # torch.fft.fft2(x, dim=(-2, -1))
        _, _, w, h = dctx.shape
        low_ratio = 0.4
        low_w = int(w * low_ratio)
        low_h = int(h * low_ratio)
        # dctx[:, :, -low_w:, -low_h:] = 0
        dctx[:, :, -low_w:, :] = 0
        dctx[:, :, :, -low_h:] = 0
        dctx = dctx  # * self.mask.reshape(1, 1, w, h)
        idctx = dct.idct_2d(dctx)
        return idctx

    def add_noise(self, x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16 / 255, 16 / 255), 0, 1)

    def gkern(self, kernel_size=3, nsig=3):
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)

    def drop_out(self, x):
        return F.dropout2d(x, p=0.1, training=True)

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

