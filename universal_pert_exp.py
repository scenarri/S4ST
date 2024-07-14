import torch
import numpy as np
import scipy
import argparse
import time
from tqdm import tqdm, tqdm_notebook
from utils import *
from pprint import pprint
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.optim as optim
from transformations.transforms import *
from transformations.BSR import *

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #seed_everything(729729)

    data_transform = {'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]),
        'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()])}

    mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    target_model_names = ['convnext', 'inception_v3', 'inception_v4_timm', 'inception_resnet_v2', 'xception',
                          'vit_base_patch16_224', 'swin', 'pit', 'tnt', 'deit',
                          'resnet50_Augmix', 'resnet50_SIN', 'resnet50_SIN_IN', 'resnet50_l2_eps0_5', 'resnet50_linf_eps0_5']

    inceptions = ['xception', 'inception_v3', 'inception_resnet_v2', 'inception_v4_timm', 'tf_ens3_adv_inc_v3',
                  'tf_ens_adv_inc_res_v2']

    target_models = [WrapperModel(load_model(x), mean, stddev, True if x in inceptions else False, x).to(device).eval()
                     for x in target_model_names]

    asr_cont = {m: 0. for m in target_model_names}

    surrogate = WrapperModel(load_model('resnet50'), mean, stddev, False, 'resnet50').to(device).eval()
    for param in surrogate.model.parameters():
        param.requires_grad = False

    stacked_kernel = torch.from_numpy(kernel_generation())
    stacked_kernel = stacked_kernel.to(device)

    image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images.csv')
    input_path = './dataset/images/'
    num_batches = int(np.ceil(len(image_id_list) / args.batch_size))

    targets = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]

    IN_labels = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.meta['categories']

    if args.attack == 'BSR':
        BSR_ = BSR_transformer()

    for idx, target in enumerate(targets):

        current_label = IN_labels[target].replace(' ', '')
        save_dir = f'./results/10targetes_all_source/{args.attack}_UAP/'
        Mkdir(save_dir)

        val_dataset = datasets.ImageFolder(root=args.valsetpath, transform=data_transform['train'])
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=True, num_workers=0, drop_last=True)
        total_num = len(val_dataset)
        tq_loader = tqdm(val_loader)
        target_label = torch.tensor([target]).cuda()

        UAP = torch.zeros([1, 3, 224, 224]).cuda()
        UAP = Variable(UAP, requires_grad=True)
        optimizer = optim.Adam([UAP], lr=0.005)
        iter = 0
        for epoch in range(1):
            for i, (x, y) in enumerate(tq_loader):
                iter += 1
                # if i == 1000: break
                x, y = x.cuda(), y.cuda()
                target_labels = target_label.expand(x.shape[0], 1).squeeze()
                target_labels_hot_labels = torch.zeros((target_labels.shape[0], 1000)).cuda()
                target_labels_hot_labels.scatter_(1, target_labels.unsqueeze(1), 1)

                if args.attack == 'None':
                    input_trans = torch.clamp(x + UAP, 0, 1)
                if args.attack == 'BSR' or args.attack == 'HAug':
                    input_trans = BSR_.transform(torch.clamp(x + UAP, 0, 1))

                outputs = surrogate(input_trans)

                outc = outputs.detach().clone()
                outc[torch.arange(x.shape[0]), target_labels] = -1e5
                second_idx = outc.max(dim=1)[1]
                targeted_logit = outputs[torch.arange(x.shape[0]), target_labels]
                second_logit = outc[torch.arange(x.shape[0]), second_idx]
                loss = (torch.maximum(second_logit - targeted_logit, -10 * torch.ones_like(targeted_logit))).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                UAP.data = UAP.data.clamp(-args.epsilon, args.epsilon)


        uap_ = UAP.clone().detach()
        for k in range(0, num_batches):
            batch_size_cur = min(args.batch_size, len(image_id_list) - k * args.batch_size)
            X = torch.zeros(batch_size_cur, 3, 224, 224).cuda()
            for i in range(batch_size_cur):
                X[i] = data_transform['test'](load_img(input_path + image_id_list[k * args.batch_size + i] + '.png'))
            target_labels = target_label.expand(X.shape[0], 1).squeeze()

            adv_X = torch.clamp(X + uap_, 0, 1)
            with torch.no_grad():
                for m in range(len(target_model_names)):
                    adv_output = target_models[m](adv_X)
                    adv_prediction = adv_output.argmax(1)
                    asr_cont[target_model_names[m]] += (torch.sum(adv_prediction == target_labels).float().item() / (len(targets) * len(image_id_list)) * 100)


        save_uap = UAP.clone().detach()
        torch.save(save_uap, f'./results/10targetes_all_source/{args.attack}_UAP/{current_label}.pt')
        saveimg_result(save_uap/args.epsilon, f'./results/10targetes_all_source/{args.attack}_UAP/{current_label}.png', idx=0)

    pprint(asr_cont, sort_dicts=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--surrogate', type=str, default='resnet50')
    parser.add_argument('--epsilon', type=float, default=16/255)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--valsetpath', type=str)

    parser.add_argument('--attack', type=str, default='None', help='None/BSR')
    main(parser.parse_args())


