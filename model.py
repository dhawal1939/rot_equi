# -*- coding: utf-8 -*-
# @Author  : dhawal1939
# @File    : model.py

import torch
from torch import nn
from config import device
from siren_pytorch import Siren


class EquiVarianceIllumination(nn.Module):
    def __init__(self, in_dim, inter_dim, num_envs, out_dim=3, **kwargs):
        super().__init__()

        self.model_1 = nn.Sequential(*[
            Siren(in_dim + 2, 128),
            Siren(128, 128),
        ])

        self.model_2 = nn.Sequential(*[
            Siren(128 + inter_dim, 128),
            Siren(128, 128),
            Siren(128, out_dim)
        ])
        three_N_val = 3 * in_dim
        multi_var_mean_sampler = torch.distributions.MultivariateNormal(loc=torch.zeros(three_N_val),
                                                                        covariance_matrix=torch.eye(three_N_val))
        multi_var_log_dev_sampler = torch.distributions.MultivariateNormal(loc=torch.ones(three_N_val) * -5,
                                                                           covariance_matrix=torch.eye(three_N_val))
        # initialize with N(0, 1)
        self.mean_parameters = nn.Parameter(multi_var_mean_sampler.rsample((num_envs,)))
        # initialize with N(-5, 1)
        self.log_variance_parameters = nn.Parameter(multi_var_log_dev_sampler.rsample((num_envs,)))

        del multi_var_log_dev_sampler, multi_var_mean_sampler

        self.s_xz = torch.Tensor([[1, 0, 0], [0, 0, 1]]).T.to(device)
        self.s_y = torch.Tensor([[0, 1, 0]]).T.to(device)

    def forward(self, d, idx):
        variance_vec = torch.e ** self.log_variance_parameters[idx][0]
        cov_diag_matrix = torch.diag(variance_vec)
        sampler = torch.distributions.MultivariateNormal(loc=self.mean_parameters[idx],
                                                         covariance_matrix=cov_diag_matrix)
        z = sampler.rsample((1,))[0].to(device)
        z = z.view(1, z.shape[-1] // 3, 3)

        # print(d.shape, self.s_y.shape, self.s_xz.shape, z.shape) 
        z_xz = z @ self.s_xz[None, ...]  # Project latentZ on to XZ-plane
        z_y = z @ self.s_y[None, ...]  # sy @ z

        d_x = d @ self.s_y[None, ...]  # sy @ d
        d_z_ = d @ self.s_xz[None, ...]
        d_z = torch.linalg.norm(d_z_, dim=-1, keepdim=True)
        d_y = d_z_ @ z_xz.permute(0, 2, 1)

        # print(d_x.shape, d_y.shape, d_z.shape, z_y.shape, z_xz.shape)
        d_ = torch.concat((d_x.view(d_x.shape[1], -1),
                           d_y.view(d_y.shape[1], -1),
                           d_z.view(d_z.shape[1], -1)), dim=-1)  # directions, N + 2
        z_ = torch.concat((z_y.view(-1, 1),
                           (z_xz[0] @ z_xz[0].permute(1, 0)).view(-1, 1)),
                          dim=0)[:, 0][None, ...]
        # print(d_.shape, z_.shape)

        x = self.model_1(d_)
        x = torch.concat((x, z_.repeat(d_x.shape[1], 1)), dim=-1)

        x = self.model_2(x)
        return x, self.mean_parameters[idx].data, variance_vec.data
