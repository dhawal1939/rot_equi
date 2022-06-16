# -*- coding: utf-8 -*-
# @Author  : dhawal1939
# @File    : dataloader.py

import math
import torch
import imageio
import numpy as np
from config import *
from torch.utils.data import Dataset


class DirectionColorHDRDataset(Dataset):
    """DirectionColorHDRDataset dataset."""

    def __init__(self, file_list, resolution=initial_resolution):
        """
        Args:
            file_list (tuple/list): list of files to read.
        """
        self.directions, self.env_maps = None, None
        self.resolution = None
        self.file_list = file_list

        self.resize(resolution)

    def get_directions(self):
        width = torch.arange(2 * self.resolution) + 0.5
        height = torch.arange(self.resolution) + 0.5
        xx, yy = torch.meshgrid(width, height, indexing='xy')
        xx = xx / (2 * self.resolution)
        yy = (self.resolution - yy) / self.resolution
        xx *= 2 * math.pi
        yy *= math.pi

        coords_phi_theta = torch.concat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=-1)

        sin_theta = torch.sin(coords_phi_theta[:, 1])
        x = sin_theta * torch.cos(coords_phi_theta[:, 0])
        y = sin_theta * torch.sin(coords_phi_theta[:, 0])
        z = torch.cos(coords_phi_theta[:, 1])

        # Directions
        return torch.concat((x, y, z), dim=-1).reshape(-1, 3)

    def resize(self, resolution):
        self.resolution = resolution
        self.env_maps = [imageio.imread(env_path, format='HDR-FI') for env_path in self.file_list]
        self.env_maps = [torch.from_numpy(np.array(img.resize(self.resolution, 2 * self.resolution)).reshape(-1, 3)) for
                         img in self.env_maps]
        self.directions = self.get_directions()

    def __len__(self):
        return len(self.env_maps)

    def __getitem__(self, idx):
        return self.directions, self.env_maps[idx]
