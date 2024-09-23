# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import random
import numpy as np
from scipy.stats import qmc

from pycocotools import mask as maskUtils

# convert coco segmentation to binary mask
def seg_to_mask(seg, img_h, img_w):
    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(seg['counts'], list):
        rle = maskUtils.frPyObjects(seg, img_h, img_w)
    else:
        rle = seg
    return maskUtils.decode(rle)

def sample_points_from_bbox(bbox, num_samples=5):
    x1, y1, x2, y2 = bbox
    points = [(random.uniform(x1, x2), random.uniform(y1, y2)) for _ in range(num_samples)]
    return points

class PoissonDiskSampler:
    def __init__(self, num_samples=100, radius=0.1, num_cache=100):
        self.num_samples = num_samples
        self.radius = radius

        self.cache_pattern = []
        for _ in range(num_cache):
            rng = np.random.default_rng()
            engine = qmc.PoissonDisk(d=2, radius=radius, seed=rng)
            sample = engine.random(num_samples)
            self.cache_pattern.append(sample)

    def sample(self, bbox, use_cache=True):
        x1, y1, x2, y2 = bbox
        if use_cache:
            sample = random.choice(self.cache_pattern).copy()
        else:
            rng = np.random.default_rng()
            engine = qmc.PoissonDisk(d=2, radius=self.radius, seed=rng)
            sample = engine.random(self.num_samples)
            
        sample[:, 0] = sample[:, 0] * (x2 - x1) + x1
        sample[:, 1] = sample[:, 1] * (y2 - y1) + y1
        return sample
    
class DotDict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]