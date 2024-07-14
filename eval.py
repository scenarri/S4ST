import argparse
from pprint import pprint
from utils import *
from tqdm import tqdm, tqdm_notebook


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_transform = transforms.Compose([
        transforms.ToTensor()])



    if args.target == 'normal':
        target_model_names = ['mobilenet_v2', 'efficientnet_b0', 'convnext', 'inception_v3', 'inception_v4_timm',
                              'inception_resnet_v2', 'xception', 'vit_base_patch16_224', 'swin', 'maxvit', 'twins_svt_base', 'pit', 'tnt', 'deit']
    elif args.target == 'secured':
        target_model_names = ['resnet50_Augmix', 'resnet50_SIN', 'resnet50_SIN_IN', 'resnet50_l2_eps0_1', 'resnet50_l2_eps0_5',
                              'resnet50_linf_eps0_5', 'resnet50_linf_eps1_0', 'tf_ens3_adv_inc_v3', 'tf_ens4_adv_inc_v3', 'tf_ens_adv_inc_res_v2']

    mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet

    inceptions = ['xception', 'inception_v3', 'inception_resnet_v2', 'inception_v4_timm', 'tf_ens3_adv_inc_v3', 'tf_ens4_adv_inc_v3', 'tf_ens_adv_inc_res_v2']

    models = [WrapperModel(load_model(x), mean, stddev, True if x in inceptions else False, x).to(device).eval()
                     for x in target_model_names]

    asr_cont = {m: 0. for m in target_model_names}
    usuc_cont = {m: 0. for m in target_model_names}
    cor_cont = {m: 0. for m in target_model_names}

    batch_size = 10
    image_id_list,label_ori_list,label_tar_list = load_ground_truth('./dataset/images.csv')
    clean_path = f'./dataset/images/'
    input_path = f'./results/{args.path}/'

    num_batches = int(np.ceil(len(image_id_list) / batch_size))
    for k in tqdm(range(0, num_batches)):
        batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
        if args.metric == 'usuc':
            X_ori = torch.zeros(batch_size_cur, 3, 224, 224).cuda()
        X_adv = torch.zeros(batch_size_cur, 3, 224, 224).cuda()
        for i in range(batch_size_cur):
            if args.metric == 'usuc':
                X_ori[i] = data_transform(load_img(clean_path + image_id_list[k * batch_size + i] + '.png'))
            X_adv[i] = data_transform(load_img(input_path + image_id_list[k * batch_size + i] + '.png'))
        labels = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).cuda()
        target_labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).cuda()

        with torch.no_grad():
            for i in range(len(models)):
                advpred = models[i](X_adv).argmax(1)
                if args.metric == 'tsuc':
                    asr_cont[target_model_names[i]] = (torch.sum(advpred == target_labels).float().item()/len(image_id_list)*100)
                if args.metric == 'usuc':
                    cleanpred = models[i](X_ori).argmax(1)
                    cor_cont[target_model_names[i]] += (torch.sum(cleanpred == labels).float().item())
                    usuc_cont[target_model_names[i]] += (torch.sum((advpred != labels) * (cleanpred == labels)).float().item())

    if args.metric == 'tsuc':
        pprint(asr_cont, sort_dicts=False)
    if args.metric == 'usuc':
        for m in range(len(target_model_names)):
            usuc_cont[target_model_names[m]] /= cor_cont[target_model_names[m]]
            usuc_cont[target_model_names[m]] *= 100
        pprint(usuc_cont)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='normal', help='normal/secured')
    parser.add_argument('--path', type=str, default='/resnet50_EOS')
    parser.add_argument('--metric', type=str, default='tsuc', help='tsuc/usuc')
    main(parser.parse_args())