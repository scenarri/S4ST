import torch
import numpy as np
import scipy
import argparse
import time
from tqdm import tqdm
from models.CFM import *
from transformations.transforms import *
import losses
from transformations.edi import *
import skopt
from skopt.space import Real, Integer
from skopt import gp_minimize
from skopt.utils import use_named_args
import transformations.basic_transformations as bt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()])
mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

target_model_names = ['mobilenet_v2', 'efficientnet_b0', 'convnext', 'inception_v3', 'inception_v4_timm',
                      'inception_resnet_v2', 'xception',
                      'vit_base_patch16_224', 'swin', 'maxvit', 'twins_svt_base', 'pit', 'tnt', 'deit']

inceptions = ['xception', 'inception_v3', 'inception_resnet_v2', 'inception_v4_timm', 'tf_ens3_adv_inc_v3',
              'tf_ens_adv_inc_res_v2']


surrogate = WrapperModel(load_model('resnet50'), mean, stddev, False).to(device).eval() #inception_v3 densenet121
for param in surrogate.model.parameters():
    param.requires_grad = False

batch_size = 5
stacked_kernel = torch.from_numpy(kernel_generation())
stacked_kernel = stacked_kernel.to(device)
image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images_50_random.csv')
input_path = './dataset/images/'
num_batches = int(np.ceil(len(image_id_list) / batch_size))

# def sample_mags(range):
#     if len(range) == 0:
#         return [0]
#     elif len(range) == 2:
#         return np.linspace(range[0], range[1], 100)
#     elif len(range) == 4:
#         x_samples = np.linspace(range[0], range[1], 10)
#         y_samples = np.linspace(range[2], range[3], 10)
#         xx, yy = np.meshgrid(x_samples, y_samples)
#         points = np.c_[xx.ravel(), yy.ravel()]
#         return points

def sample_mags(range):
    if len(range) == 0:
        return [0]
    elif len(range) == 2:
        return np.linspace(range[0], range[1], 25)
    elif len(range) == 4:
        x_samples = np.linspace(range[0], range[1], 5)
        y_samples = np.linspace(range[2], range[3], 5)
        xx, yy = np.meshgrid(x_samples, y_samples)
        points = np.c_[xx.ravel(), yy.ravel()]
        return points

trans_names = ['Rotation', 'Scaling', 'Shear', 'Perspective', 'Flip', 'Crop', 'Translate',
               'Solarize', 'Hue', 'Brightness', 'Contrast', 'Saturation']

trans_func = {#'None': torchvision.transforms.v2.Identity(),
              'Rotation': lambda r_value: bt.rotation_trans(r=r_value),
              'Scaling': lambda r_value: bt.scaling_trans(r=r_value),
              'Crop': lambda r_value: bt.crop_trans(r=r_value),
              'Shear': lambda r_value: bt.shear_trans(r=r_value),
              'Elastic': lambda r_value: bt.elastic_trans(r=r_value),
              'Perspective': lambda r_value: bt.perspective_trans(r=r_value),
              'Brightness': lambda r_value: bt.brightness_trans(r=r_value),
              'Saturation': lambda r_value: bt.saturation_trans(r=r_value),
              'Hue': lambda r_value: bt.hue_trans(r=r_value),
              'Sharpeness': lambda r_value: bt.sharpness_trans(r=r_value),
              'Solarize': lambda r_value: bt.solarize_trans(r=r_value),
              'Contrast': lambda r_value: bt.contrast_trans(r=r_value),
              'Flip': lambda r_value: bt.flip_trans(r=r_value),
              'Translate': lambda r_value: bt.translate_trans(r=r_value)
              }

trans_magitude = \
    {'Rotation': [0., 1.], 'Scaling': [0., 1.], 'Shear': [0., 1., 0., 1.], 'Perspective': [0., 1.],
     'Flip': [], 'Elastic': [0., 1.], 'Crop': [0., 1.], 'Translate': [0., 1., 0., 1.], 'Solarize': [0., 1.],
     'Hue': [0., 1.], 'Brightness': [0., 1.], 'Contrast': [0., 1.], 'Saturation': [0., 1.], 'Sharpeness': [0., 1.]}

def to_named_params(results, search_space):
    params = results.x
    param_dict = {}
    params_list  =[(dimension.name, param) for dimension, param in zip(search_space, params)]
    for item in params_list:
        param_dict[item[0]] = item[1]
    return(param_dict)

space = [
         Integer(0, 10, name='pR'),
         Integer(0, 10, name='pAug'),
         Integer(0, 6, name='block'),
         Integer(10, 30, name='r'),
]
@use_named_args(space)
def objective(pR, pAug, block, r):
    seed_everything(729729)


    BLOCK = [[1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4], [4, 4]]

    EDI = EDI_transformer(num_block=BLOCK[block], pR=pR/10, pAug=pAug/10, r=r/10)
    transferability = 0

    for k in tqdm(range(0, num_batches)):
        batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
        X = torch.zeros(batch_size_cur, 3, 224, 224).cuda()
        for i in range(batch_size_cur):
            X[i] = data_transform(load_img(input_path + image_id_list[k * batch_size + i] + '.png'))
        labels = torch.tensor(label_ori_list[k * batch_size:k * batch_size + batch_size_cur]).cuda()
        target_labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).cuda()

        target_one_hot_labels = torch.zeros((target_labels.shape[0], 1000)).cuda()
        target_one_hot_labels.scatter_(1, target_labels.unsqueeze(1), 1)
        source_one_hot_labels = torch.zeros((labels.shape[0], 1000)).cuda()
        source_one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

        adv_X = X.clone().detach()
        X_ori = X.clone().detach()
        momentum = torch.zeros_like(X).detach().to(device)
        delta = torch.zeros_like(adv_X).detach().to(device)

        for iter in range(900):
            delta.requires_grad_()
            X_trans = EDI.transform(X_ori + delta)
            outputs = surrogate(X_trans)
            cost = losses.loss_pool(outputs, target_labels, labels, target_one_hot_labels, source_one_hot_labels, 'CE_Margin')
            cost.backward(retain_graph=False)
            grad = delta.grad.clone()
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding="same", groups=3)  # TI
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)  # MI
            grad = grad + momentum * 1.0
            momentum = grad
            delta = delta.detach() + 2/255 * grad.sign()
            delta = torch.clamp(delta, min=-16/255, max=16/255)
            delta = (torch.clamp(X + delta, min=0, max=1) - X).detach()
        adv_X = torch.clamp(X_ori + delta, 0, 1)
        torch.cuda.empty_cache()

        with torch.no_grad():
            for name in trans_names:
                magnitudes = sample_mags(trans_magitude[name])
                num_points = len(magnitudes)
                for magi, mag in enumerate(magnitudes):
                    num_trans_x = len(trans_func[name](mag)(torch.zeros([1, 3, 224, 224]).cuda()))
                    for trans_X_idx in range(num_trans_x):
                        trans_input = trans_func[name](mag)(adv_X.clone())[trans_X_idx].detach()
                        adv_output = surrogate(trans_input)
                        clean_output = surrogate(X_ori.clone())
                        transferability += ((adv_output.softmax(1) - clean_output.softmax(1)) * target_one_hot_labels).sum().item()/num_trans_x/num_points/len(trans_names)/50


    return -1 * transferability


results = gp_minimize(objective, space, n_calls=100, random_state=None, n_random_starts=10, verbose=True, kappa=1.96, xi=0.01)
best_res = to_named_params(results, space)

print(best_res)
skopt.dump(results, 'BayesOpt-selftrans.pkl')
skopt.plots.plot_convergence(results)
plt.show()

results = skopt.load('BayesOpt-selftrans.pkl')
print()
paras = results['x_iters']
vals = results['func_vals']
pR = []
pAug = []
block = []
r = []
value = []
for i in range(len(paras)):
    pR.append(paras[i][0]/10)
    pAug.append(paras[i][1]/10)
    block.append(paras[i][2])
    r.append(paras[i][3]/10)
    value.append(-1*vals[i])

csv = df = pd.DataFrame({'pR': pR, 'pAug': pAug, 'block': block, 'r': r, 'vals': value})
csv.to_csv('optimizestatistic-selftrans.csv')


# skopt.plots.plot_objective(results)
# plt.show()
# skopt.plots.plot_evaluations(results)
# plt.show()