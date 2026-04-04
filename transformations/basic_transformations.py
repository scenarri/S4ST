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
import torchvision.transforms.functional as TF


class scaling_trans():
    def __init__(self, r=0.5, p=1.0):
        self.r = r
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            _, _, ori_w, ori_h, = x.shape
            r_w_a, r_h_a = int(ori_w * (1 + 1.5 * self.r)), int(ori_h * (1 + 1.5 * self.r))
            r_w_b, r_h_b = int(ori_w / (1 + 1.5 * self.r)), int(ori_h / (1 + 1.5 * self.r))
            return [F.interpolate(x, size=[r_w_a, r_h_a], mode="bilinear", align_corners=False, antialias=True),
                    F.interpolate(x, size=[r_w_b, r_h_b], mode="bilinear", align_corners=False, antialias=True)]
        else:
            return x

class scaling_base():
    def __init__(self, r=0.5, p=1.0):
        self.r = r
        self.p = p
    
    def __call__(self, x):
        if torch.rand(1) < self.p:
            _, _, ori_w, ori_h, = x.shape
            r_w_a, r_h_a = int(ori_w * (1 + 1.5 * self.r)), int(ori_h * (1 + 1.5 * self.r))
            r_w_b, r_h_b = int(ori_w / (1 + 1.5 * self.r)), int(ori_h / (1 + 1.5 * self.r))
            
            rescaled_a = torchvision.transforms.CenterCrop([int(ori_w**2/r_w_a), int(ori_h**2/r_h_a)])(x)
            rescaled_a = F.interpolate(rescaled_a, size=[ori_w, ori_h], mode="bilinear", align_corners=False, antialias=True)
            
            rescaled_b = F.interpolate(x, size=[r_w_b, r_h_b], mode="bilinear", align_corners=False, antialias=True)
            w_rem = ori_w - r_w_b
            h_rem = ori_h - r_h_b
            pad_top = h_rem//2
            pad_bottom = h_rem - pad_top
            pad_left = w_rem//2
            pad_right = w_rem - pad_left
            padded = F.pad(rescaled_b, [pad_top, pad_bottom, pad_left, pad_right],
                           mode='constant', value=0)
            return [rescaled_a, padded]
        else:
            return x
            

class rotation_trans():
    def __init__(self, r=0.5, p=1.0):
        self.r = r
        self.p = p
    def __call__(self, x):
        if torch.rand(1) < self.p:
            deg = 180 * self.r
            return [TF.rotate(x, angle=deg, interpolation=InterpolationMode.BILINEAR), TF.rotate(x, angle=-deg, interpolation=InterpolationMode.BILINEAR)]
        else:
            return x

class crop_trans():
    def __init__(self, r=0.5, p=1.0):
        self.r = r
        self.p = p

    @staticmethod
    def get_params(img, scale, ratio):
        _, height, width = TF.get_dimensions(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, x):
        if torch.rand(1) < self.p:
            s = 1 - 0.95 * self.r
            i, j, h, w = self.get_params(x, [s, s], [1.0 / 1.0, 1.0 / 1.0])
            return [TF.crop(x, i, j, h, w)]
        else:
            return x


class crop_trans_fill():
    def __init__(self, r=0.5, p=1.0):
        self.r = r
        self.p = p

    @staticmethod
    def get_params(img, scale, ratio):
        _, height, width = TF.get_dimensions(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, x):
        if torch.rand(1) < self.p:
            _, _, ori_w, ori_h, = x.shape
            s = 1 - 0.95 * self.r
            i, j, h, w = self.get_params(x, [s, s], [1.0 / 1.0, 1.0 / 1.0])
            cropped = TF.crop(x, i, j, h, w)
            _, _, new_h, new_w = cropped.shape
            w_rem = ori_w - new_w
            h_rem = ori_h - new_h
            pad_top = h_rem // 2
            pad_bottom = h_rem - pad_top
            pad_left = w_rem // 2
            pad_right = w_rem - pad_left
            padded = F.pad(cropped, [pad_top, pad_bottom, pad_left, pad_right],
                           mode='constant', value=0)
            return [padded]
        else:
            return x
        
class translate_trans():
    def __init__(self, r=[0.5, 0.5], p=1.0):
        self.r = r
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            step_x = int(self.r[0] * x.shape[2])
            step_y = int(self.r[1] * x.shape[2])
            return [TF.affine(x, angle=0.0, translate=[step_x, step_y], scale=1.0, shear=[0., 0.], interpolation=InterpolationMode.BILINEAR),
                    TF.affine(x, angle=0.0, translate=[-step_x, step_y], scale=1.0, shear=[0., 0.], interpolation=InterpolationMode.BILINEAR),
                    TF.affine(x, angle=0.0, translate=[step_x, -step_y], scale=1.0, shear=[0., 0.], interpolation=InterpolationMode.BILINEAR),
                    TF.affine(x, angle=0.0, translate=[-step_x, -step_y], scale=1.0, shear=[0., 0.], interpolation=InterpolationMode.BILINEAR)]
        else:
            return x

class shear_trans():
    def __init__(self, r=[0.5, 0.5], p=1.0):
        self.r = r
        self.p = p

    def __call__(self, x):
        deg_x = self.r[0] * 90
        deg_y = self.r[1] * 90
        if torch.rand(1) < self.p:
            return [TF.affine(x, angle=0.0, translate=[0,0], scale=1.0, shear=[deg_x, deg_y], interpolation=InterpolationMode.BILINEAR),
                    TF.affine(x, angle=0.0, translate=[0,0], scale=1.0, shear=[-deg_x, deg_y], interpolation=InterpolationMode.BILINEAR),
                    TF.affine(x, angle=0.0, translate=[0,0], scale=1.0, shear=[deg_x, -deg_y], interpolation=InterpolationMode.BILINEAR),
                    TF.affine(x, angle=0.0, translate=[0,0], scale=1.0, shear=[-deg_x, -deg_y], interpolation=InterpolationMode.BILINEAR)]
        else:
            return x

class elastic_trans():
    def __init__(self, r=0.5, sigma=5.0, p=1.0):
        r = [float(r) * 150, float(r) * 150]
        r = [r[0], r[0]]
        sigma = [float(sigma), float(sigma)]
        sigma = [sigma[0], sigma[0]]
        self.r = r
        self.sigma = sigma
        self.p = p

    @staticmethod
    def get_params(alpha, sigma, size):
        dx = torch.rand([1, 1] + size) * 2 - 1
        if sigma[0] > 0.0:
            kx = int(8 * sigma[0] + 1)
            # if kernel size is even we have to make it odd
            if kx % 2 == 0:
                kx += 1
            dx = TF.gaussian_blur(dx, [kx, kx], sigma)
        dx = dx * alpha[0] / size[0]

        dy = torch.rand([1, 1] + size) * 2 - 1
        if sigma[1] > 0.0:
            ky = int(8 * sigma[1] + 1)
            # if kernel size is even we have to make it odd
            if ky % 2 == 0:
                ky += 1
            dy = TF.gaussian_blur(dy, [ky, ky], sigma)
        dy = dy * alpha[1] / size[1]
        return torch.concat([dx, dy], 1).permute([0, 2, 3, 1])  # 1 x H x W x 2

    def __call__(self, x):
        if torch.rand(1) < self.p:
            _, height, width = TF.get_dimensions(x)
            displacement = self.get_params(self.r, self.sigma, [height, width])
            return [TF.elastic_transform(x, displacement=displacement, interpolation=InterpolationMode.BILINEAR)]
        else:
            return x

class solarize_trans():     # [0,2]
    def __init__(self, r=0.5, p=1.0):
        self.r = r
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            return [TF.solarize(x, 1-self.r)]
        else:
            return x

class flip_trans():     # [0,2]
    def __init__(self, r=0.5, p=1.0):
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            return [TF.hflip(x), TF.vflip(x)]
        else:
            return x

class contrast_trans():     # [0,2]
    def __init__(self, r=0.5, p=1.0):
        self.r = r
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            return [TF.adjust_contrast(x, (1 + 4 * self.r)), TF.adjust_contrast(x, 1/(1 + 4 * self.r))]
        else:
            return x

class brightness_trans():     # [0,2]
    def __init__(self, r=0.5, p=1.0):
        self.r = r
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            return [TF.adjust_brightness(x, (1 + 4 * self.r)), TF.adjust_brightness(x, 1/(1 + 4 * self.r))]
        else:
            return x

class saturation_trans():     # [0,2]
    def __init__(self, r=0.5, p=1.0):
        self.r = r
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            return [TF.adjust_saturation(x, (1 + 4 * self.r)), TF.adjust_saturation(x, 1/(1 + 4 * self.r))]
        else:
            return x

class hue_trans():     # [-0.5,0.5]
    def __init__(self, r=0.5, p=1.0):
        self.r = r
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            return [TF.adjust_hue(x, 0.5*self.r), TF.adjust_hue(x, -0.5*self.r)]
        else:
            return x

class perspective_trans():
    def __init__(self, r=0.5, p=1.0):
        self.r = r
        self.p = p

    @staticmethod
    def get_params(width: int, height: int, distortion_scale: float):
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]

        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __call__(self, x):
        if torch.rand(1) < self.p:
            channels, height, width = TF.get_dimensions(x)
            startpoints, endpoints = self.get_params(width, height, self.r)
            return [TF.perspective(x, startpoints, endpoints, InterpolationMode.BILINEAR, 0)]
        else:
            return x
