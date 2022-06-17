# -*- coding: utf-8 -*-
# @Author  : dhawal1939
# @File    : config.py

import torch

latent_dim = 27
beta_kld = 1e-4
epochs = 2400
upscale_interval = 800
learning_rate = 1e-5
initial_resolution = 16
device = 'cpu:0' if not torch.cuda.is_available() else 'cuda:0'
file_dir = 'data/'
log_dir_path = './logs/'
