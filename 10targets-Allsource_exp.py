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
from transformations.transforms import *
from models.ttp_generators import GeneratorResnet
from transformations.gaussian_smoothing import *
from transformations.S4ST import *
from transformations.BSR import *
from transformations.SIA import *

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()])

    mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    target_model_names = ['mobilenet_v2', 'efficientnet_b0', 'convnext', 'inception_v3', 'inception_v4_timm', 'inception_resnet_v2', 'xception',
                          'vit_base_patch16_224', 'swin', 'maxvit', 'twins_svt_base', 'pit', 'tnt', 'deit',
                          'resnet50_Augmix', 'resnet50_SIN', 'resnet50_SIN_IN', 'resnet50_l2_eps0_1','resnet50_l2_eps0_5','resnet50_linf_eps0_5', 'resnet50_linf_eps1_0', 'tf_ens3_adv_inc_v3', 'tf_ens4_adv_inc_v3','tf_ens_adv_inc_res_v2']

    inceptions = ['xception', 'inception_v3', 'inception_resnet_v2', 'inception_v4_timm', 'tf_ens3_adv_inc_v3','tf_ens_adv_inc_res_v2', 'tf_ens4_adv_inc_v3']

    target_models = [WrapperModel(load_model(x), mean, stddev, True if x in inceptions else False, x).to(device).eval() for x in target_model_names]
    asr_cont = {m: 0. for m in target_model_names}

    stacked_kernel = torch.from_numpy(kernel_generation())
    stacked_kernel = stacked_kernel.to(device)

    image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images.csv')

    input_path = './dataset/images/'
    num_batches = int(np.ceil(len(image_id_list) / args.batch_size))

    # Set-up Kernel
    kernel_size = 3
    pad = 2
    sigma = 1
    kernel = get_gaussian_kernel(kernel_size=kernel_size, pad=pad, sigma=sigma).cuda()

    targets = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]

    IN_labels = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.meta['categories']


    for idx, target in enumerate(targets):

        current_label = IN_labels[target].replace(' ', '')

        save_dir = f'./results/10targetes_all_source/{args.attack}/{current_label}'

        Mkdir(save_dir)
        Mkdir(os.path.join(save_dir, 'images'))
        Mkdir(os.path.join(save_dir, 'perts'))

        if args.attack == 'TTP':
            model_name = f'./models/netG/TTP/netG_res50_IN_19_{target}.pth'
            netG = GeneratorResnet()
            netG.load_state_dict(torch.load(model_name))
            netG.cuda()
            netG.eval()

        if args.attack == 'M3D':
            model_name = f'./models/netG/M3D/netG_resnet50_9_{target}.pth'
            netG = GeneratorResnet()
            netG.load_state_dict(torch.load(model_name))
            netG.cuda()
            netG.eval()

        if args.attack == 'SIA':
            SIA_ = SIA_transformer()

        if args.attack == 'BSR':
            BSR_ = BSR_transformer()

        if 'EOS' in args.attack:
            S4ST_ = S4ST_transformer(num_block=[2,3], pR=0.9, pAug=1.0, r=1.9)

        for k in tqdm(range(0, num_batches)):
            batch_size_cur = min(args.batch_size, len(image_id_list) - k * args.batch_size)
            X = torch.zeros(batch_size_cur, 3, 224, 224).cuda()
            for i in range(batch_size_cur):
                X[i] = data_transform(load_img(input_path + image_id_list[k * args.batch_size + i] + '.png'))
            labels = torch.tensor(label_ori_list[k * args.batch_size:k * args.batch_size + batch_size_cur]).cuda()

            target_labels = torch.LongTensor(X.size(0)).cuda()
            target_labels.fill_(target)


            if args.attack == 'TTP' or args.attack == 'M3D':
                X_ori = X.clone().detach()
                adv = kernel(netG(X)).detach()
                adv = torch.min(torch.max(adv, X - args.epsilon), X + args.epsilon)
                adv_X = torch.clamp(adv, 0.0, 1.0)

            else:
                adv_X = X.clone().detach()
                X_ori = X.clone().detach()
                momentum = torch.zeros_like(X).detach().to(device)
                delta = torch.zeros_like(adv_X).detach().to(device)

                surrogate = WrapperModel(load_model(args.surrogate), mean, stddev, True if args.surrogate in inceptions else False, args.surrogate).to(device).eval()
                for param in surrogate.model.parameters():
                    param.requires_grad = False

                for iter in range(args.atkiter):
                    delta.requires_grad_()

                    if args.attack == 'S4ST':
                        X_trans = S4ST_.transform(X_ori + delta)
                    elif args.attack == 'SIA':
                        X_trans = SIA_.blocktransform(X_ori + delta)
                    elif args.attack == 'BSR':
                        X_trans = BSR_.transform(X_ori + delta)
                    elif args.attack == 'None':
                        X_trans = X_ori + delta
                    outputs = surrogate(X_trans)
                    cost = CE_Margin(outputs, target_labels)

                    cost.backward(retain_graph=False)
                    grad = delta.grad.clone()
                    grad = F.conv2d(grad, stacked_kernel, stride=1, padding="same", groups=3)  # TI
                    grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)  # MI
                    grad = grad + momentum * args.decay
                    momentum = grad

                    delta = delta.detach() + args.alpha * grad.sign()
                    delta = torch.clamp(delta, min=-args.epsilon, max=args.epsilon)
                    delta = (torch.clamp(X + delta, min=0, max=1) - X).detach().clone()

                adv_X = torch.clamp(X_ori + delta, 0, 1)
                torch.cuda.empty_cache()


            with torch.no_grad():
                for m in range(len(target_model_names)):
                    adv_output = target_models[m](adv_X)
                    adv_prediction = adv_output.argmax(1)
                    asr_cont[target_model_names[m]] += (torch.sum(adv_prediction == target_labels).float().item() / (len(targets) * len(image_id_list)) * 100)

            for i in range(adv_X.shape[0]):
                pert = (adv_X - X_ori).clone()
                pert = pert/args.epsilon
                saveimg_result(adv_X, os.path.join(save_dir, 'images', image_id_list[k * args.batch_size + i] + '.png'), idx=i)
                saveimg_result(pert, os.path.join(save_dir, 'perts', image_id_list[k * args.batch_size + i] + '_pert.png'), idx=i)

    pprint(asr_cont, sort_dicts=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--surrogate', type=str, default='resnet50')
    parser.add_argument('--epsilon', type=float, default=16/255)
    parser.add_argument('--alpha', type=float, default=2/255)
    parser.add_argument('--atkiter', type=int, default=900)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--decay', type=float, default =1.)

    parser.add_argument('--attack', type=str, default='TTP', help='TTP/M3D/S4ST/')
    main(parser.parse_args())


