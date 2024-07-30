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
from transformations.SIA import *
import transformations.BSR as BSR
from transformations.transforms import *
from torchvision import datasets
from transformations.S4ST import *
from transformations.decowa import *
from transformations.BSR import *

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.methods == 'ODI':
        from transformations.ODI import Render3D, render_3d_aug_input

    seed_everything(729729)

    data_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()])
    mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


    inceptions = ['xception', 'inception_v3', 'inception_resnet_v2', 'inception_v4_timm', 'tf_ens3_adv_inc_v3',
                  'tf_ens_adv_inc_res_v2', 'tf_ens4_adv_inc_v3']

    surrogate_names = args.surrogate.split(',')
    surrogate = [WrapperModel(load_model(x), mean, stddev, True if x in inceptions else False, x).to(device).eval() for x in surrogate_names]
    for s_m in surrogate:
        for param in s_m.model.parameters():
            param.requires_grad = False

    stacked_kernel = torch.from_numpy(kernel_generation(len_kernel=5, nsig=3, kernel_name="gaussian"))
    stacked_kernel = stacked_kernel.to(device)

    if args.save:
        save_sur_name = args.surrogate.replace(',', '-')
        eps_str = str(args.epsilon * 255).split('.')[0]
        save_name_start = f'{save_sur_name}_{args.methods}_Iter-{args.atkiter}_Copies-{args.copies}_Eps-{eps_str}'
        if args.methods == 'S4ST':
            save_name_start += f'_pR-{args.pR}_pAug-{args.pAug}_Block-{args.block}_r-{args.r}'
        print(f'save_dir: {save_name_start}')
        save_dir = os.path.join('./results/', save_name_start)
        Mkdir(save_dir)

    total_time = 0

    image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images.csv')
    input_path = './dataset/images/'
    num_batches = int(np.ceil(len(image_id_list) / args.batch_size))
    total_num = len(image_id_list)

    for k in tqdm(range(0, num_batches)):
        batch_size_cur = min(args.batch_size, len(image_id_list) - k * args.batch_size)
        X = torch.zeros(batch_size_cur, 3, 224, 224).cuda()
        for i in range(batch_size_cur):
            X[i] = data_transform(load_img(input_path + image_id_list[k * args.batch_size + i] + '.png'))
        labels = torch.tensor(label_ori_list[k * args.batch_size:k * args.batch_size + batch_size_cur]).cuda()
        target_labels = torch.tensor(label_tar_list[k * args.batch_size:k * args.batch_size + batch_size_cur]).cuda()

        adv_X = X.clone().detach()
        X_ori = X.clone().detach()
        momentum = torch.zeros_like(X).detach().to(device)
        delta = torch.zeros_like(adv_X).detach().to(device)

        if args.methods == 'SIA':
            SIA_ = SIA_transformer()
        if args.methods == 'BSR':
            BSR_ = BSR_transformer()
        if args.methods == 'S4ST':
            S4ST_ = S4ST_transformer(num_block=args.block, pR=args.pR, pAug=args.pAug, r=args.r)
        if args.methods == 'ODI':
            renderer = Render3D(config_idx=580)
        if args.methods == 'decowa':
            decowa_ = decowa_transformer(rho=0.0001)
        if args.methods == 'Scaling':
            scaling_ = S4ST_transformer(pR=args.pR, r=args.r)

        for iter in range(args.atkiter):
            time_start = time.time()
            delta.requires_grad_()

            grad = 0
            for copy in range(args.copies):
                outputs = 0
                for i_m_ in range(len(surrogate_names)):
                    if args.methods == 'None':
                        X_trans = (X_ori + delta)
                    elif args.methods == 'DI':
                        X_trans = DI(X_ori + delta, resize_rate_H=330/300, diversity_prob=0.7)
                    elif args.methods == 'RDI':
                        X_trans = RDI(X_ori + delta, resize_rate_H=340/300, diversity_prob=0.7)
                    elif args.methods == 'Scaling':
                        X_trans = scaling_.scaling(X_ori + delta)
                    elif args.methods == 'ODI':
                        X_trans = render_3d_aug_input(x_adv=(X_ori + delta), renderer=renderer, prob=0.7)
                    elif args.methods == 'S4ST':
                        X_trans = S4ST_.transform(X_ori + delta)
                    elif args.methods == 'SSA':
                        X_trans = SSA_aug(X_ori + delta)
                    elif args.methods == 'SIA':
                        X_trans = SIA_.blocktransform(X_ori + delta)
                    elif args.methods == 'BSR':
                        X_trans = BSR_.transform(X_ori + delta)
                    elif args.methods == 'decowa':
                        X_trans = decowa_.transform(X_ori + delta, target_labels, surrogate[i_m_])
                    elif args.methods == 'SI':
                        X_trans = (X_ori + delta) / torch.pow(torch.tensor(2), torch.tensor(copy))   # copies = 5
                    elif args.methods == 'Admix':
                        img_other = (X_ori + delta)[torch.randperm(X_ori.shape[0])].view(X_ori.size())
                        X_trans = X_ori + delta + 0.2 * img_other   # copies = 5

                    outputs += (surrogate[i_m_](X_trans) / len(surrogate_names))

                cost = CE_Margin(outputs, target_labels)

                if args.methods == 'ODI':
                    grad = grad + torch.autograd.grad(cost, delta, retain_graph=False, create_graph=False)[0]
                else:
                    try:
                        cost.backward(retain_graph=False)
                        grad = grad + delta.grad.clone()
                        delta.grad.zero_()
                    except:
                        pass

            grad = grad / args.copies

            grad = F.conv2d(grad, stacked_kernel, stride=1, padding="same", groups=3)  # TI
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)  # MI
            grad = grad + momentum * args.decay
            momentum = grad

            delta = delta.detach() + args.alpha * grad.sign()
            delta = torch.clamp(delta, min=-args.epsilon, max=args.epsilon)
            delta = (torch.clamp(X + delta, min=0, max=1) - X).detach().clone()

            total_time += time.time() - time_start

        adv_X = torch.clamp(X_ori + delta, 0, 1)
        torch.cuda.empty_cache()

        if args.save:
            for i in range(X.shape[0]):
                Mkdir(os.path.join(save_dir, 'images'))
                save_name = os.path.join(save_dir, 'images', image_id_list[k * args.batch_size + i] + '.png')
                saveimg_result(adv_X, save_name, idx=i)

    print(f'total time: {total_time}, per image: {round(total_time/total_num,5)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--surrogate', type=str, default='resnet50', help='ens: resnet50,resnet152,densenet121,vgg16_bn')
    parser.add_argument('--epsilon', type=float, default=16 / 255)
    parser.add_argument('--alpha', type=float, default=2 / 255)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--decay', type=float, default=1.)
    parser.add_argument('--save', action='store_false')
    parser.add_argument('--methods', type=str, default='S4ST', help='None/DI/RDI/ODI/SI/Admix/SSA/SIA/BSR/S4ST/Scaling')
    parser.add_argument('--r', type=float, default=1.9)
    parser.add_argument('--pR', type=float, default=0.9)
    parser.add_argument('--pAug', type=float, default=1.0)
    parser.add_argument('--block', type=parse_list, default=[2,3])

    parser.add_argument('--copies', type=int, default=1)
    parser.add_argument('--atkiter', type=int, default=900)

    main(parser.parse_args())
