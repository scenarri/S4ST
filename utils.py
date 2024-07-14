import os
import numpy as np
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import csv
import torch.nn.functional as F
import math
from models.MadryLab import model_utils as Madry_model_utils
from models.MadryLab.datasets import ImageNet as Madry_ImageNet
import numpy as np
from models.tf_models import tf_ens3_adv_inc_v3, tf_ens4_adv_inc_v3, tf_ens_adv_inc_res_v2

def seed_everything(seed):
    import os
    import random
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensure that CuDNN uses deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Disable TF32 (TensorFloat-32), which can cause nondeterministic behavior on NVIDIA A100 GPUs and newer
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, 'allow_tf32'):
        torch.allow_tf32 = False

    print(f"All random seeds set to {seed}")


def CE_Margin(logits, target_labels):
    value, _ = torch.sort(logits, dim=1, descending=True)
    logits = logits / torch.unsqueeze(value[:, 0] - value[:, 1], 1).detach()
    loss = -1 * nn.CrossEntropyLoss(reduction='sum')(logits, target_labels)
    return loss

def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

def load_img(path):
    return Image.open(path)

def showimg(data, l=0):
    data = data[l].detach().cpu().swapaxes(0,1)
    data = data.swapaxes(1,2)
    plt.imshow(data)
    plt.show()

def saveimg_result(data, name, idx=0):
    data_np = 255 * np.transpose(data[idx].detach().cpu().numpy(), (1, 2, 0))
    img = Image.fromarray(data_np.astype('uint8'))
    img.save(name)

def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append( row['ImageId'] )
            label_ori_list.append( int(row['TrueLabel']) - 1 )
            label_tar_list.append( int(row['TargetClass']) - 1 )

    return image_id_list,label_ori_list,label_tar_list

class WrapperModel(nn.Module):
    def __init__(self, model,  mean, std, inc=False, name='torch'):
        super(WrapperModel, self).__init__()
        self.mean = torch.Tensor(mean)
        self.model=model
        self.inc=inc
        self.std = torch.Tensor(std)
        self.training = model.training
        self.name = name
    def forward(self, x):
        if self.inc == True:
            x = transforms.Resize((299, 299), interpolation=InterpolationMode.NEAREST)(x)
            if 'tf' in self.name:
                out = self.model(x * 2.0 - 1.0)[0][:, 1:]
            else:
                out = self.model(x*2.0 - 1.0)
        else:
            if self.name != 'tnt':
                out = self.model((x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None])
            else:
                out = self.model(x * 2.0 - 1.0)
        return out

dir_path = os.path.dirname(os.path.realpath(__file__))
def load_model(model_name):

    if model_name == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

    elif model_name == 'resnet50_SIN':
        model = torchvision.models.resnet50()
        checkpoint = torch.load(os.path.join('./models/weights', 'resnet50_train_60_epochs-c8e5653e.pth.tar'))
        new_state_dict = {}
        for key, value in checkpoint["state_dict"].items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)

    elif model_name == 'resnet50_SIN_IN':
        model = torchvision.models.resnet50()
        checkpoint = torch.load(os.path.join('./models/weights', 'resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar'))
        new_state_dict = {}
        for key, value in checkpoint["state_dict"].items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)

    elif model_name == 'resnet50_Augmix':
        model = torchvision.models.resnet50()
        checkpoint = torch.load(os.path.join('./models/weights', 'res50Augmix.pth.tar'))
        new_state_dict = {}
        for key, value in checkpoint["state_dict"].items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)

    elif model_name == 'convnext':
        model = torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1)

    elif model_name == "vit_base_patch16_224":
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)

    elif model_name == 'maxvit':
        model = torchvision.models.maxvit_t(weights=torchvision.models.MaxVit_T_Weights.IMAGENET1K_V1)

    elif model_name == 'swin':
        model = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)

    elif model_name == "resnet152":
        model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)

    elif model_name == "vgg16_bn":
        model = torchvision.models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1)

    elif model_name == "vgg19_bn":
        model = torchvision.models.vgg19_bn(weights=torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1)

    elif model_name == "inception_v3":
        model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)

    elif model_name == 'densenet121':
        model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)

    elif model_name == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)

    elif model_name == 'tf_ens3_adv_inc_v3':
        model_path = os.path.join('./models/weights', model_name + '.npy')
        model = tf_ens3_adv_inc_v3
        model = model.KitModel(model_path).eval().cuda()

    elif model_name == 'tf_ens4_adv_inc_v3':
        model_path = os.path.join('./models/weights', model_name + '.npy')
        model = tf_ens4_adv_inc_v3
        model = model.KitModel(model_path).eval().cuda()

    elif model_name == 'tf_ens_adv_inc_res_v2':
        model_path = os.path.join('./models/weights', model_name + '.npy')
        model = tf_ens_adv_inc_res_v2
        model = model.KitModel(model_path).eval().cuda()

    # timm models
    elif model_name == "efficientnet_b0":
        model = timm.create_model('efficientnet_b0', pretrained=True)

    elif model_name == "inception_resnet_v2":
        model = timm.create_model("inception_resnet_v2", pretrained=True)

    elif model_name == "inception_v3_timm":
        model = timm.create_model("inception_v3", pretrained=True)

    elif model_name == "inception_v4_timm":
        model = timm.create_model("inception_v4", pretrained=True)

    elif model_name == "xception":
        model = timm.create_model("legacy_xception", pretrained=True)

    elif model_name == "levit_384":
        model = timm.create_model("levit_384", pretrained=True)

    elif model_name == "convit_base":
        model = timm.create_model("convit_base", pretrained=True)

    elif model_name == "twins_svt_base":
        model = timm.create_model("twins_svt_base", pretrained=True)

    elif model_name == "pit":
        model = timm.create_model('pit_s_224', pretrained=True)

    elif model_name == "tnt":
        model = timm.create_model('pit_s_224', pretrained=True)

    elif model_name == "deit":
        model = timm.create_model('pit_s_224', pretrained=True)

    # https://github.com/microsoft/robust-models-transfer
    elif model_name == "resnet50_l2_eps0_1":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''), resume_path=os.path.join('./models/weights', 'resnet50_l2_eps0.1.ckpt'))
        model = m.model

    elif model_name == "resnet50_l2_eps0_25":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''), resume_path=os.path.join('./models/weights', 'resnet50_l2_eps0.25.ckpt'))
        model = m.model

    elif model_name == "resnet50_l2_eps0_5":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''), resume_path=os.path.join('./models/weights', 'resnet50_l2_eps0.5.ckpt'))
        model = m.model

    elif model_name == "resnet50_linf_eps0_5":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''), resume_path=os.path.join('./models/weights', 'resnet50_linf_eps0.5.ckpt'))
        model = m.model

    elif model_name == "resnet50_linf_eps1_0":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''), resume_path=os.path.join('./models/weights', 'resnet50_linf_eps1.0.ckpt'))
        model = m.model

    else:
        raise ValueError(f"Not supported model name. {model_name}")
    return model.eval()

def parse_list(string):
    return list(map(int, string.strip('[]').split(',')))