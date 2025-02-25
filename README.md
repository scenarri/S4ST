# S4ST_TTA
:wave: This is a pytorch script to reproduce the experiments in our paper for simple transferable targeted attacks.


## Contents  

1) [Contributions](#Contributions) 
2) [Acknowledgements](#Acknowledgements)
3) [Pretrained Weights](#Pretrained-Weights) 
4) [Generating Targeted Adversarial Examples](#Generating-Targeted-Adversarial-Examples) 
5) [Evaluation](#Evaluation)


## Contributions

1. We propose two blind estimation measures: surrogate self-alignment and self-transferability against basic transformations. Serving as effective proxies for the black-box transferability of targeted AEs, they enable feasible analyses of the effectiveness and synergistic effect of various basic transformations without accessing any additional data and victim models, significantly diminishing the dependence on empirical or intuitive choices common in the prior art.
2. To the best of our knowledge, we are the pioneers in revealing the unique superior efficacy of the simple scaling transformation in enhancing targeted transferability, and the redundancies within geometric and color transformations. Based on these, we further design S4ST, an advanced scaling-centered transformation that ingeniously integrates modified scaling with complementary transformations and leverages the benefits of block-wise operations, adhering to a strict black-box manner.
3. Extensive and comprehensive experiments on the ImageNet-Compatible dataset substantiate the effectiveness and efficiency of S$^4$ST. It not only outperforms existing transformation techniques in terms of both effectiveness and efficiency but also surpasses SOTA TTA solutions that depend on 50k to 1.2 million training samples, all without requiring additional data. It also exhibits exceptional transferability to commercial APIs and vision-language models. It is found to significantly improve the exploitation of existing visual elements to produce highly transferable target semantics, circumventing the necessity to learn this capability from extensive multi-class data.


## Acknowledgements

This repository benefits a lot from previous works, including [CFM](https://github.com/dreamflake/CFM), [Targeted-Transfer](https://github.com/ZhengyuZhao/Targeted-Tansfer), [TransferAttackEval
](https://github.com/ZhengyuZhao/TransferAttackEval), [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack). Sincere thanks for their contributions to the adversarial machine learning community.

## Pretrained Weights

Most evaluated models can be automatically downloaded by *torchvision* and *[timm](https://github.com/huggingface/pytorch-image-models)*. 
Please manually download other pretrained weights ([SIN&IN](https://github.com/rgeirhos/texture-vs-shape/blob/master/models/load_pretrained_models.py), [Augmix](https://drive.google.com/file/d/1z-1V3rdFiwqSECz7Wkmn4VJVefJGJGiF/view?usp=sharing), [AT](https://huggingface.co/madrylab/robust-imagenet-models), [Ensemble AT](https://github.com/ylhz/tf_to_pytorch_model)) and drop them to './models/weights/'.

## Generating Targeted Adversarial Examples

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

### Simple Targeted Attack

You can perform simple targeted attacks with various input transformation methods, and using a single or multiple surrogate models, with the following command:

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

## Evaluation

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




