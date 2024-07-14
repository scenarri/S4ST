# S4ST_TTA
:wave: This is a pytorch script to reproduce the experiments in our paper for simple transferable targeted attacks.


## Contents  

1) [Contributions](#Contributions) 
2) [Acknowledgements](#Acknowledgements)
3) [Pretrained Weights](#Pretrained-Weights) 
4) [Generating Targeted Adversarial Examples](#Generating-Targeted-Adversarial-Examples) 
5) [Evaluation](#Evaluation)


## Contributions

1. We contribute a novel and concise self-universal perspective, shedding new light on elucidating the pivotal role of input transformations in TTA. In contrast to the previous consensus, it reveals the feasibility of enhancing targeted transferability by strong transformations.
2. To the best of our knowledge, we are the first to experimentally investigate the impact of a single transformation method on targeted transferability. Our empirical demonstration highlights the significant potential of simple scaling in enhancing TTA. Furthermore, we propose the S4ST to integrate complementary gains from existing methods to achieve superior performance.
3. We conduct comprehensive experiments against diverse and challenging victim models under various settings on the ImageNet-Compatible benchmark dataset. Results demonstrated that our method outperforms SOTA methods by a significant margin in terms of both effectiveness and efficiency.


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




