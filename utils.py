# -*- coding: utf-8 -*-
# @Author  : dhawal1939
# @File    : utils.py

import cv2
import math
import torch
import numpy as np
from numpy.random import uniform

def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)

def get_antilog_01_vals(val):
    # Color space conversion
    val = (val + 1) / 2
    val = math.e ** val
    tonemapDurand = cv2.createTonemap(2.2)
    ldrDurand = tonemapDurand.process(val[:].detach().cpu().numpy())
    val = np.clip(ldrDurand, 0, 1)
    return val

class PercentileExposure(object):
    def __init__(self, gamma=2.4, low_perc=2, high_perc=98, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            low_perc = uniform(0, 15)
            high_perc = uniform(85, 100)
        self.gamma = gamma
        self.low_perc = low_perc
        self.high_perc = high_perc

    def __call__(self, x):
        low, high = np.percentile(x, (self.low_perc, self.high_perc))
        return map_range(np.clip(x, low, high)) ** (1 / self.gamma)