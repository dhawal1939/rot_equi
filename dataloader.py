# -*- coding: utf-8 -*-
# @Author  : dhawal1939
# @File    : dataloader.py

import cv2
import math
import torch
import imageio
import numpy as np
from torch import nn
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
        coords = torch.permute(torch.stack((xx, yy)), (1, 2, 0))        
        coords = coords / (self.resolution) * math.pi
        coords = coords.reshape(-1, 2)
        self.sin_theta = torch.sin(coords[:, 1])
        x = self.sin_theta * torch.cos(coords[:, 0])
        y = self.sin_theta * torch.sin(coords[:, 0])
        z = torch.cos(coords[:, 1])

        # Directions
        return torch.concat((x, y, z), dim=-1).reshape(-1, 3)

    def resize(self, resolution):
        self.resolution = resolution
        self.env_maps = [cv2.resize(cv2.imread(env_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), 
                                    (2 * self.resolution, self.resolution)) for env_path in self.file_list]
        self.env_maps = [np.clip(img, img[img > 0.0].min(), img[img < np.inf].max()) for img in self.env_maps]
        self.env_maps = torch.from_numpy(np.array(self.env_maps))
        # torch.clamp_(self.env_maps, min = 1e-20, max = 1e25)
        torch.nan_to_num_(self.env_maps, nan=1e-20)

        self.env_maps = self.env_maps.reshape(-1, 2 * self.resolution * self.resolution, 3)

        # log and scale
        self.env_maps = torch.log(self.env_maps)

        for i in range(3):  # RGB
            _max, _min = torch.max(self.env_maps[:, :, i]), torch.min(self.env_maps[:, :, i])
            self.env_maps[:, :, i] = (self.env_maps[:, :, i] - _min) / (_max - _min + 1e-5)
                
        self.env_maps[:, :, :] = 2 * self.env_maps - 1

        self.directions = self.get_directions()

    def __len__(self):
        return len(self.env_maps)

    def __getitem__(self, idx):
        return self.directions, self.env_maps[idx], self.sin_theta, idx
