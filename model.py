# -*- coding: utf-8 -*-
# @Author  : dhawal1939
# @File    : model.py

import torch
from torch import nn
from siren_pytorch import Siren


class EquiVarianceIllumination(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim=3, **kwargs):
        super().__init__()

        self.model_1 = nn.Sequential(*[
                                        Siren(in_dim, 128),
                                        Siren(128, 128),
                                      ])

        self.model_2 = nn.Sequential(*[
                                        Siren(128 + inter_dim, 128),
                                        Siren(128, 128),
                                        Siren(128, out_dim)
                                       ])

    def forward(self, d_, z_):
        x = self.model_1(d_)
        x = torch.concat((x, z_), dim=0)
        return self.model_2(x)
