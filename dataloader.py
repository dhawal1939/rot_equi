# -*- coding: utf-8 -*-
# @Author  : dhawal1939
# @File    : dataloader.py

import cv2
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

        self.sin_theta = torch.sin(coords_phi_theta[:, 1])
        x = self.sin_theta * torch.cos(coords_phi_theta[:, 0])
        y = self.sin_theta * torch.sin(coords_phi_theta[:, 0])
        z = torch.cos(coords_phi_theta[:, 1])

        # Directions
        return torch.concat((x, y, z), dim=-1).reshape(-1, 3)

    def resize(self, resolution):
        self.resolution = resolution
        self.env_maps = [cv2.imread(env_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) for env_path in self.file_list]

        self.env_maps = [np.array(cv2.resize(env_map, (2 * self.resolution, self.resolution)))
                         for env_map in self.env_maps]
        self.env_maps = torch.from_numpy(np.array(self.env_maps).reshape(-1, 1))

        min_val = torch.min(self.env_maps)
        max_val = torch.max(self.env_maps)
        if min_val <= 0.:
            self.env_maps = self.env_maps - min_val + 1e-5  # add epsilon
        if max_val == torch.inf:
            self.env_maps[self.env_maps == torch.inf] = 1e30

        self.env_maps = self.env_maps.view(-1, 2 * self.resolution * self.resolution, 3)

        # log and scale
        self.env_maps = torch.log(self.env_maps)

        # for i, e in enumerate(self.env_maps):
        #     if np.isnan(e).any():
        #         print(e.min(), e.max(), np.isnan(e).sum(), flush=True)
        #         raise Exception(f"Error nan in {i}")
        #     if np.isinf(e).any():
        #         print(e.min(), e.max(), np.isnan(e).sum(), flush=True)
        #         # raise Exception(f"Error inf in {i}")
        # exit(-1)

        for i in range(3):  # RGB
            _max, _min = torch.max(self.env_maps[:, :, i]), torch.min(self.env_maps[:, :, i])
            self.env_maps[:, :, i] = (self.env_maps[:, :, i] - _min) / (_max - _min)
        self.env_maps[:, :, :] = 2 * self.env_maps - 1

        # self.env_maps = [imageio.imread(env_path, format='HDR-FI') for env_path in self.file_list]
        # self.env_maps = [torch.from_numpy(np.array(img.resize(self.resolution, 
        #                                            2 * self.resolution)).reshape(-1, 3)) 
        #                                            for img in self.env_maps]
        self.directions = self.get_directions()

    def __len__(self):
        return len(self.env_maps)

    def __getitem__(self, idx):
        return self.directions, self.env_maps[idx], self.sin_theta, idx
