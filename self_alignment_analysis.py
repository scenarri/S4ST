import os
import torch
import torchvision.transforms.v2
from torchvision.datasets import CIFAR100, ImageNet
from tqdm import tqdm, tqdm_notebook
import csv
from PIL import Image
import matplotlib.pyplot as plt
import json
import pandas as pd
from utils import *
import argparse
from transformations.edi import *
from transformations.transforms import *
from openpyxl import Workbook
import torchvision.transforms.functional as TF
import random
import transformations.basic_transformations as bt

device = "cuda" if torch.cuda.is_available() else "cpu"

data_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()])

target_model_names = ['mobilenet_v2', 'efficientnet_b0', 'convnext', 'inception_v3', 'inception_v4_timm', 'inception_resnet_v2',
                      'xception', 'vit_base_patch16_224', 'swin', 'maxvit', 'twins_svt_base', 'pit', 'tnt', 'deit']

mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet
inceptions = ['xception', 'inception_v3', 'inception_resnet_v2', 'inception_v4_timm', 'tf_ens3_adv_inc_v3', 'tf_ens4_adv_inc_v3', 'tf_ens_adv_inc_res_v2']
models = [WrapperModel(load_model(x), mean, stddev, True if x in inceptions else False, x).to(device).eval() for x in target_model_names]

surrogate = WrapperModel(load_model('resnet50'), mean, stddev, False).to(device).eval()

def sample_mags(range):
    if len(range) == 0:
        return [0]
    elif len(range) == 2:
        return np.linspace(range[0], range[1], 100)
    elif len(range) == 4:
        x_samples = np.linspace(range[0], range[1], 10)
        y_samples = np.linspace(range[2], range[3], 10)
        xx, yy = np.meshgrid(x_samples, y_samples)
        points = np.c_[xx.ravel(), yy.ravel()]
        return points

trans_names = ['Rotation', 'Scaling', 'Shear', 'Perspective', 'Flip', 'Elastic', 'Crop', 'Translate',
               'Solarize', 'Hue', 'Brightness', 'Contrast', 'Saturation', 'Sharpeness']

trans_func = {'None': torchvision.transforms.v2.Identity(),
              'Rotation': lambda r_value: bt.rotation_trans(r=r_value),
              'Scaling': lambda r_value: bt.scaling_trans(r=r_value),
              'Crop': lambda r_value: bt.crop_trans(r=r_value),
              'Shear': lambda r_value: bt.shear_trans(r=r_value),
              'Perspective': lambda r_value: bt.perspective_trans(r=r_value),
              'Brightness': lambda r_value: bt.brightness_trans(r=r_value),
              'Saturation': lambda r_value: bt.saturation_trans(r=r_value),
              'Hue': lambda r_value: bt.hue_trans(r=r_value),
              'Solarize': lambda r_value: bt.solarize_trans(r=r_value),
              'Contrast': lambda r_value: bt.contrast_trans(r=r_value),
              'Flip': lambda r_value: bt.flip_trans(r=r_value),
              'Translate': lambda r_value: bt.translate_trans(r=r_value)
              }

trans_magitude = \
    {'Rotation': [0., 1.], 'Scaling': [0., 1.], 'Shear': [0., 1., 0., 1.], 'Perspective': [0., 1.],
     'Flip': [], 'Elastic': [0., 1.], 'Crop': [0., 1.], 'Translate': [0., 1., 0., 1.], 'Solarize': [0., 1.],
     'Hue': [0., 1.], 'Brightness': [0., 1.], 'Contrast': [0., 1.], 'Saturation': [0., 1.], 'Sharpeness': [0., 1.]}


image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images.csv')

adv_path = f'./results/tab3_vanilla/resnet50_None_L-CE_Margin_Iter-900_Copies-1_Eps-16/images/'
input_path = './dataset/images/'

batchsize = 10
num_batches = int(np.ceil(len(image_id_list) / batchsize))
total_num = len(image_id_list)


def compute_nearest_neighbors(feats, topk=100):
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    knn = ((feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk])
    return knn

def get_mask(feat, topk=100):
    feat = F.normalize(feat, p=2, dim=1)
    knn_f = compute_nearest_neighbors(feat, topk)
    n = knn_f.shape[0]
    range_tensor = torch.arange(n, device=knn_f.device).unsqueeze(1)
    f_mask = torch.zeros(n, n, device=knn_f.device)
    f_mask[range_tensor, knn_f] = 1.0
    return f_mask

def compute_multual_alignment(x_mask, y_mask, topk=50):
    return ((x_mask * y_mask).sum(dim=1) / topk).mean().item()


topk = 100
with torch.no_grad():
    # Table 1
    
    # collect features
    f_x = torch.zeros([total_num, 1000]).cuda()
    f_x_adv = torch.zeros([total_num, 1000]).cuda()
    g_x = {m: torch.zeros([total_num, 1000]).cuda() for m in target_model_names}
    g_x_adv = {m: torch.zeros([total_num, 1000]).cuda() for m in target_model_names}
    
    for k in tqdm(range(0, num_batches)):
        batch_size_cur = min(batchsize, len(image_id_list) - k * batchsize)
        ori_X = torch.zeros(batch_size_cur, 3, 224, 224).cuda()
        adv_X = torch.zeros(batch_size_cur, 3, 224, 224).cuda()
        for i in range(batch_size_cur):
            ori_X[i] = data_transform(load_img(input_path + image_id_list[k * batchsize + i] + '.png'))
            adv_X[i] = data_transform(load_img(adv_path + image_id_list[k * batchsize + i] + '.png'))
        f_x[k * batchsize: k * batchsize + batch_size_cur] = surrogate(ori_X)
        f_x_adv[k * batchsize: k * batchsize + batch_size_cur] = surrogate(adv_X)
        for i, m in enumerate(target_model_names):
            g_x[m][k * batchsize: k * batchsize + batch_size_cur] = models[i](ori_X)
            g_x_adv[m][k * batchsize: k * batchsize + batch_size_cur] = models[i](adv_X)
    
    print('alignment <f_x, g_x>')
    acc = 0
    x_mask = get_mask(f_x, topk=topk)
    for m in target_model_names:
        y_mask = get_mask(g_x[m], topk=topk)
        acc += compute_multual_alignment(x_mask, y_mask, topk=topk) / len(target_model_names)
    print(acc)
    
    print('alignment <g_x, g_x^adv>')
    acc = 0
    for m in target_model_names:
        x_mask = get_mask(g_x[m], topk=topk)
        y_mask = get_mask(g_x_adv[m], topk=topk)
        acc += compute_multual_alignment(x_mask, y_mask, topk=topk) / len(target_model_names)
    print(acc)
    
    print('alignment <f_x^adv, g_x^adv>')
    acc = 0
    x_mask = get_mask(f_x_adv, topk=topk)
    for m in target_model_names:
        y_mask = get_mask(g_x_adv[m], topk=topk)
        acc += compute_multual_alignment(x_mask, y_mask, topk=topk) / len(target_model_names)
    print(acc)
    
    print('alignment <f_x^adv, f_x>')
    x_mask = get_mask(f_x_adv, topk=topk)
    y_mask = get_mask(f_x, topk=topk)
    acc = compute_multual_alignment(x_mask, y_mask, topk=topk) / len(target_model_names)
    print(acc)
    
    # Figure 9
    # collect f(T_s(x^adv)) at varying s
    f_trans_s = {}
    for idx, name in enumerate(trans_names):
        magnitudes = sample_mags(trans_magitude[name])
        num_points = len(magnitudes)
        f_trans_s[name] = {}
        for magi, mag in enumerate(tqdm(magnitudes)):
            f_trans_s[name][f'mag_{magi}'] = {}
            num_trans_x = len(trans_func[name](mag)(torch.zeros([1, 3, 224, 224]).cuda()))
            for trans_X_idx in range(num_trans_x):
                f_trans_s[name][f'mag_{magi}'][f'{trans_X_idx}'] = torch.zeros([total_num, 1000]).cuda()
            for k in range(0, num_batches):
                batch_size_cur = min(batchsize, len(image_id_list) - k * batchsize)
                adv_x = torch.zeros(batch_size_cur, 3, 224, 224).cuda()
                for i in range(batch_size_cur):
                    adv_x[i] = data_transform(load_img(adv_path + image_id_list[k * batchsize + i] + '.png'))
                trans_adv_x = trans_func[name](mag)(adv_x)
                for trans_X_idx in range(num_trans_x):
                    f_trans_s[name][f'mag_{magi}'][f'{trans_X_idx}'][k * batchsize: k * batchsize + batch_size_cur] = surrogate(trans_adv_x[trans_X_idx])
                
    # self-alignment curves
    save_dic = {}
    x_mask = get_mask(f_x, topk=topk)
    for idx, name in enumerate(trans_names):
        num_points = len(f_trans_s[name].keys())
        save_dic[name] = [0.] * num_points
        for i in tqdm(range(num_points)):
            num_imgs = len(f_trans_s[name][f'mag_{i}'].keys())
            for j in range(num_imgs):
                y_mask = get_mask(f_trans_s[name][f'mag_{i}'][f'{j}'], topk=topk)
                acc = compute_multual_alignment(x_mask, y_mask, topk=topk)
                save_dic[name][i] += acc / num_imgs
    max_length = max(len(item) for item in save_dic.values())
    for key, value in save_dic.items():
        save_dic[key] = value + [None] * (max_length - len(value))
    df = pd.DataFrame(save_dic)
    df.to_excel('./self_alignment_curves.xlsx')
    
    
    # black-box alignment curves
    save_dic = {}
    for idx, name in enumerate(trans_names):
        num_points = len(f_trans_s[name].keys())
        save_dic[name] = [0.] * num_points
        for i in tqdm(range(num_points)):
            num_imgs = len(f_trans_s[name][f'mag_{i}'].keys())
            for j in range(num_imgs):
                x_mask = get_mask(f_trans_s[m][name][f'mag_{i}'][f'{j}'], topk=topk)
                for m in target_model_names:
                    y_mask = get_mask(g_x_adv[m], topk=topk)
                    acc = compute_multual_alignment(x_mask, y_mask, topk=topk)
                    save_dic[name][i] += acc / num_imgs / len(target_model_names)
    max_length = max(len(item) for item in save_dic.values())
    for key, value in save_dic.items():
        save_dic[key] = value + [None] * (max_length - len(value))
    df = pd.DataFrame(save_dic)
    torch.save(f_trans_s, './black_box_alignment_curves.pth')
