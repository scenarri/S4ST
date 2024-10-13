# S4ST_TTA
:wave: This is a pytorch script to reproduce the experiments in our paper for simple transferable targeted attacks.


## Contents  

1) [Contributions](#Contributions) 
2) [Acknowledgements](#Acknowledgements)
3) [Pretrained Weights](#Pretrained-Weights) 
4) [Generating Targeted Adversarial Examples](#Generating-Targeted-Adversarial-Examples) 
5) [Evaluation](#Evaluation)


## Contributions

1. We revisit various relevant transformation techniques and the prevailing consensus on designing them for transfer attacks. Our experimental results elucidate the critical role of image transformations in simple TTAs and underscore the current consensus's limitations in understanding these transformations' effectiveness.
2. We introduce effective black-box measures and provide valuable insights. Self-transferability against basic geometric transformations is verified as a reliable indicator of black-box transferability. We further propose surrogate self-alignment to blindly estimate the benefits of attacking basic transformations in enhancing targeted transferability. These two allow for feasible analysis of basic transformations under the black-box setting, significantly diminishing the dependence on empirical or intuitive choices common in current research.
3. To the best of our knowledge, we are the pioneers in demonstrating the uniquely superior efficacy of simple scaling transformations in promoting targeted transferability. An advanced scaling-centric transformation, S4ST, is further devised, ingeniously integrating modified scaling with complementary transformations and leveraging the benefits of block-wise operations.
4. Through extensive and comprehensive experiments on the ImageNet-Compatible dataset, we substantiate our proposed method's effectiveness and efficiency, outperforming existing resource-intensive and simple solutions by considerable margins, thereby establishing a new SOTA for TTAs.


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




