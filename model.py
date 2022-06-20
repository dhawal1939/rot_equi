# -*- coding: utf-8 -*-
# @Author  : dhawal1939
# @File    : model.py

from collections import OrderedDict
import torch
import numpy as np
from torch import nn, var
from config import device
from siren_pytorch import SirenNet, Sine


class EquiVarianceIllumination(nn.Module):
    def __init__(self, in_dim, num_envs, hidden_f=128, out_dim=3, **kwargs):
        super().__init__()

        print(in_dim + 2 + in_dim**2 + in_dim)
        self.model = SirenNet(
                                dim_in=in_dim + 2 + in_dim**2 + in_dim,
                                dim_hidden=hidden_f,
                                num_layers=5,
                                dim_out=out_dim,
                                w0=30.,
                                final_activation=nn.Identity()
                            )
        three_N_val = 3 * in_dim
        multi_var_mean_sampler = torch.distributions.MultivariateNormal(loc=torch.zeros(three_N_val),
                                                                        covariance_matrix=torch.eye(three_N_val))
        multi_var_log_dev_sampler = torch.distributions.MultivariateNormal(loc=torch.ones(three_N_val) * -5,
                                                                           covariance_matrix=torch.eye(three_N_val))
        # initialize with N(0, 1)
        self.mean_parameters = nn.Parameter(multi_var_mean_sampler.sample((num_envs,)))
        # initialize with N(-5, 1)
        self.log_variance_parameters = nn.Parameter(multi_var_log_dev_sampler.sample((num_envs,)))

        self.noise_sampler = torch.distributions.MultivariateNormal(loc=torch.zeros(three_N_val),
                                                                    covariance_matrix=torch.eye(three_N_val))

        del multi_var_log_dev_sampler, multi_var_mean_sampler

    def forward(self, d, idx):
        variance_vec = torch.e ** self.log_variance_parameters[idx][0]
        cov_diag_matrix = torch.diag(variance_vec)
        cov_diag_matrix = torch.diag(variance_vec)
        sampler = torch.distributions.MultivariateNormal(loc=self.mean_parameters[idx],
                                                            covariance_matrix=cov_diag_matrix)

        z = sampler.rsample((1,))[0].to(device)
        z = z.view(3, z.shape[-1] // 3)         # 3 X N

        z_xz = torch.stack((z[0], z[2]))        # projection of z onto xz plane (2 x 3) X (3 x N) ==> 2 X N (z[0] and z[1] components)
        z_y = z[1]                              # projection of z onto y-axis   (1x3) X (3 X N)   ==> 1 X N

        d_x = d[:, :, 1]                        # projection of d onto y ==> y component of d (#d , 1)
        d_z_ = torch.stack((d[:, :, 0], d[:, :, 2])).permute(1, 2, 0)  # d_z' ===> dx and dz components component (1, 512, 2)
        d_z = torch.linalg.norm(d_z_, dim=-1)
        d_y = d_z_ @ z_xz[None, ...]        
        d_ = torch.concat((d_x.view(d_x.shape[1], -1),
                           d_y.view(d_y.shape[1], -1),
                           d_z.view(d_z.shape[1], -1)), dim=-1)  # directions, N + 2

        z_ = torch.concat((z_y.view(-1, 1),
                          (z_xz.permute(1, 0) @ z_xz).view(-1, 1)),
                          dim=0)[:, 0][None, ...]

        x = torch.concat((d_, z_.repeat(d_x.shape[1], 1)), dim=-1)
        x = self.model(x)
        return x, self.mean_parameters[idx].data, variance_vec.data, self.log_variance_parameters[idx][0].data
