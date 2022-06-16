# -*- coding: utf-8 -*-
# @Author  : dhawal1939
# @File    : train.py

from model import EquiVarianceIllumination
from config import *


model = EquiVarianceIllumination(in_dim=latent_dim // 3, inter_dim=latent_dim)
