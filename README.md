# SAM4MLLM
This is the implementation of our ECCV'24 "SAM4MLLM: Enhance Multi-Modal Large Language Model for Referring Expression Segmentation"


<img src="./image/SAM4MLLM_PQPP.png" width="60%">


## Dataset Preparation
Download each dataset from website:
- [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [PACO-LVIS](https://github.com/facebookresearch/paco/tree/main)
- [Part-ImageNet](https://github.com/TACJu/PartImageNet)
- [RefCOCO](https://github.com/lichengunc/refer)
- [GRES](https://github.com/henghuiding/ReLA)

You are responsible for checking if the dataset license is fit for the intended purpose.

Put all of them under data directory so you should get:

```
    SAM4MLLM/
    ├──dataset/
    |  ├──ADE20K/
    |  ├──PACO-LVIS/
    |  ├──Part-ImageNet/
    |  ├──RefCOCO/
    |  ├──GRES/
```


## Installation
- pytorch==2.1.2
- transformers==4.42.4
- peft==0.11.1
- lightning==2.3.3
- FlashAttention2(optional)
- LLaVA-NeXT: Follow instruction in https://github.com/LLaVA-VL/LLaVA-NeXT
- EfficientVIT-SAM: Follow instruction in https://github.com/mit-han-lab/efficientvit


## Data pre-process

- Rearrange data

In data, Run each jupyter notebook to generate dataset for training.

- Convert the data into dialouge format:

```
python to_chat_format.ipynb
```

## Traning
```
python sam4mllm_train.py
```

## Inference 
Run simple_infer.ipynb


## Licenses
Copyright © 2024, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](https://github.com/AI-Application-and-Integration-Lab/SAM4MLLM/blob/main/LICENCE) to view a copy of this license.
