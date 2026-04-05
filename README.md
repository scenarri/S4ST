<h1 align="center"> S<sup>4</sup>ST: A Strong, Self-transferable, faSt, and Simple Scale Transformation for Data-free Transferable Targeted Attack </h1> 
<h5 align="center"><em> Yongxiang Liu, Bowen Peng, Li Liu, Xiang Li </em></h5>

<p align="center">
    <a href="#contributions">Contributions</a> |
    <a href="#evaluation">Evaluation</a> |
    <a href="#analysis">Analysis</a> |
    <a href="#resources">Resources</a> |
    <a href="#acknowledgements">Acknowledgements</a> |
    <a href="#statement">Statement</a>
</p >
<p align="center">
	<a href="https://ieeexplore.ieee.org/document/"><img src="https://img.shields.io/badge/Paper-TPAMI-blue"></a>
    <a href="https://arxiv.org/abs/2410.13891"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>
    <a href="https://pan.baidu.com/s/1lI1KRHAyris49v5DDRW5bw?pwd=1huy"><img src="https://img.shields.io/badge/Resource-BaiduNetDisk-blue"></a>
</p>

## Contributions

1. We propose self-alignment and self-transferability as blind estimation measures. They serve as effective proxies to analyze basic transformations' effectiveness and synergies without accessing victims or extra data, reducing the reliance on empirical choices common in prior art.
2. We pioneer the discovery of simple scaling's superior efficacy in enhancing targeted transferability. This stems from visual data's inherent nature and the universal adoption of scale augmentation during training, revealing a dual-edged sword: practices enhancing generalization simultaneously introduce transfer attack vulnerabilities.
3. We propose S<sup>4</sup>ST, an advanced scaling-centered transformation integrating modified scaling, complementary transformations, and block-wise operations under strict black-box constraints.
4. Extensive evaluations across natural images, medical imaging, and face verification validate our framework's transferability. S<sup>4</sup>ST outperforms existing transformation methods and data-reliant SoTA TTAs (using 50k-1.2M samples), showing robust transferability to commercial APIs and vision-language models (VLMs).

## Evaluation

### Requirements

torch==2.1.0, torchvision==0.16.0, timm==0.9.11, or
```
conda create --name edi -y python=3.10
conda activate edi
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
python -m pip install -r requirements.txt
```

:arrow_down: [Optional] for [ODI](https://github.com/dreamflake/ODI)
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d
```

### Generating Targeted Adversarial Examples

You can perform targeted attacks with various input transformation methods, and using a single or multiple surrogate models, with the following command:

```
python attack.py  --surrogate resnet50 (or use multiple models by comma, e.g., resnet50,resnet152,densenet121,vgg16_bn)
                  --methods None (for baseline, DI/RDI/ODI/SI/Admix/SSA/SIA/BSR/S4ST)
                  --r 1.9 (to define the scale range [1/s, s])
                  --pR 0.9 (probability to perform scaling)
                  --pAug 1.0 (probability to pre-perform orthogonal transformations)
                  --block [2,3] (blocks for scaling)
                  --atkiter 900 (more diverse inputs require more steps to converge, previously set to 300)
```

This will print the save dir. and save all adversarial examples there (at './results/').

### ImageNet-Compatible dataset evaluation

Just run the command below to evaluate the generated examples.
```
python eval.py --path xxx (as the attack.py prints)
               --target normal (for CNNs and Vits, 'secured' for robust models)
```

### Comparison with TTP and M3D

Run the following command; it will print the results (before that, please download pretrained generators for [TTP](https://github.com/Muzammal-Naseer/TTP) and [M3D](https://github.com/Asteriajojo/M3D) and drop them to './models/netG/TTP(M3D)/'). 

```
python eval_10targets.py --attack TTP (TTP/M3D/SIA/BSR/S4ST)
```

## Analysis

### Self-alignment, self-transferability, and beyond

please see [self_alignment_analysis.py](https://github.com/scenarri/S4ST/blob/main/self_alignment_analysis.py) and [self_transferability_correlation_analysis.py](https://github.com/scenarri/S4ST/blob/main/self_transferability_correlation_analysis.py) for details.

## Resources

Most evaluated models can be automatically downloaded by *torchvision* and *[timm](https://github.com/huggingface/pytorch-image-models)*. 
Please manually download other pretrained weights ([SIN&IN](https://github.com/rgeirhos/texture-vs-shape/blob/master/models/load_pretrained_models.py), [Augmix](https://drive.google.com/file/d/1z-1V3rdFiwqSECz7Wkmn4VJVefJGJGiF/view?usp=sharing), [AT](https://huggingface.co/madrylab/robust-imagenet-models), [Ensemble AT](https://github.com/ylhz/tf_to_pytorch_model)) and drop them to './models/weights/'.

The generated adversarial examples for most cases are provided at [BaiduNetDisk](https://pan.baidu.com/s/1lI1KRHAyris49v5DDRW5bw?pwd=1huy) for further analysis and evaluation, including the RN50-halfRRC and RN50-woRRC weights and results obtained by commercial APIs and VLMs.

## Acknowledgements

This repository benefits a lot from previous works, including [CFM](https://github.com/dreamflake/CFM), [Targeted-Transfer](https://github.com/ZhengyuZhao/Targeted-Tansfer), [TransferAttackEval
](https://github.com/ZhengyuZhao/TransferAttackEval), [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack). Sincere thanks for their contributions to the adversarial machine learning community.

## Statement

- If you have any questions, please contact us via pbow16@nudt.edu.cn. 

- If you find our work is useful, please give us a star 🌟 in GitHub and cite our paper in the following BibTex format:

```
@ARTICLE{liu2026s4st,
  author={Liu, Yongxiang and Peng, Bowen and Liu, Li and Li, Xiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={{S4ST}: A Strong, Self-transferable, faSt, and Simple Scale Transformation for Data-free Transferable Targeted Attack}, 
  year={2026},
  volume={},
  number={},
  pages={1-17}
}
```
