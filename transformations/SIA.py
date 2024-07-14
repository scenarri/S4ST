import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils import *
import torch.nn.functional as F
import torch_dct as dct
import scipy.stats as st

class SIA_transformer():
    def __init__(self, num_block=3):
        super().__init__()
        self.op = [self.resize, self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip,
                   self.rotate180, self.scale, self.add_noise, self.dct, self.drop_out]
        self.num_block = num_block
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    def resize(self, x):
        """
        Resize the input
        """
        _, _, w, h = x.shape
        scale_factor = 0.8
        new_h = int(h * scale_factor) + 1
        new_w = int(w * scale_factor) + 1
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(0, 1)
        return x

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

    def blocktransform(self, x, choice=-1):
        _, _, w, h = x.shape
        y_axis = [0, ] + np.random.choice(list(range(1, h)), self.num_block - 1, replace=False).tolist() + [h, ]
        x_axis = [0, ] + np.random.choice(list(range(1, w)), self.num_block - 1, replace=False).tolist() + [w, ]
        y_axis.sort()
        x_axis.sort()

        x_copy = x.clone()
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = self.op[chosen](
                    x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])

        return x_copy

    def transform(self, x):
        #return self.op[np.random.randint(0, len(self.op), dtype=np.int32)](x)
        return self.blocktransform(x)

