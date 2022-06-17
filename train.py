# -*- coding: utf-8 -*-
# @Author  : dhawal1939
# @File    : train.py

import glob
from utils import *
from config import *
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from model import EquiVarianceIllumination
from dataloader import DirectionColorHDRDataset
from torch.utils.tensorboard import SummaryWriter

train_file_dir = Path(file_dir) / 'HDRI_Train'
test_file_dir = Path(file_dir) / 'HDRI_Test'
_log_dir = Path(log_dir_path) / datetime.now().strftime("%Y_%m_%d_%H_%M")
_log_dir.absolute().mkdir(exist_ok=True, parents=True)

_save_path = Path(log_dir_path) / datetime.now().strftime("%Y_%m_%d_%H_%M")
Path(_save_path).absolute().mkdir(exist_ok=True, parents=True)

train_files_list = sorted(glob.glob(str(train_file_dir) + '/*.exr'))[:500]
test_files_list = sorted(glob.glob(str(test_file_dir) + '/*.exr'))

writer = SummaryWriter(_log_dir)

N_val = latent_dim // 3
model = EquiVarianceIllumination(in_dim=N_val, inter_dim=N_val ** 2 + N_val, num_envs=len(train_files_list))


current_resolution = initial_resolution

train_dataset = DirectionColorHDRDataset(train_files_list,
                                    resolution=current_resolution)
train_loader = DataLoader(train_dataset,
                        shuffle=True,
                        batch_size=1,
                        num_workers=0)


optim = torch.optim.Adam(params=model.parameters(),
                         lr=learning_rate)
# starts at 1e-5 and ends at 1e-7
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim,
                                                   gamma=(1e-7 / learning_rate) ** (1 / (epochs - 1)))

start_epoch = 0
model = model.to(device)
if load_wt:
    check_point = torch.load('./logs/2022_06_17_17_54/checkpoint_700.pt')
    model.load_state_dict(check_point['model_state_dict'])
    start_epoch = check_point['epoch']
    optim.load_state_dict(check_point['optimizer_state_dict'])
    current_resolution = check_point['current_resolution']

model.train()
for epoch in range(start_epoch, epochs):

    if epoch in [800, 1600]:
        current_resolution *= 2
        train_dataset.resize(current_resolution)

    print(f'Epoch: {epoch}', flush=True)
    # loss accumulation
    recons_loss = torch.Tensor([0]).to(device)
    kld_loss = torch.Tensor([0]).to(device)
    loss = torch.Tensor([0]).to(device)

    pbar = tqdm(total=len(train_loader))

    optim.zero_grad()
    for i, batch in enumerate(train_loader):
        directions, gt_rgb_vals, sin_theta, env_idx = batch
        
        # print('pkb', directions.shape, gt_rgb_vals.shape, sin_theta.shape, 
        # torch.sum(torch.isnan(gt_rgb_vals)), torch.sum(torch.isnan(directions)))
        
        directions = directions.to(device)
        gt_rgb_vals = gt_rgb_vals.to(device)
        sin_theta = sin_theta.to(device)

        pred_vals, mu, var = model(directions, env_idx)

        recons_loss += (1 / sin_theta.shape[1]) * torch.sum(sin_theta.t() * (pred_vals - gt_rgb_vals[0]) ** 2)
        kld_loss += torch.sum(1 + torch.log(var) - var - mu ** 2)
        torch.cuda.empty_cache()
        pbar.update()
    pbar.close()

    loss += recons_loss
    loss -= (beta_kld / latent_dim) * (0.5 * kld_loss)

    loss.backward()
    optim.step()
    scheduler.step()

    print(f'Loss: {loss.item()}', flush=True)
    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_image('pred_env_map', torch.e ** pred_vals.view(current_resolution, 2 * current_resolution, 3),
                     global_step=epoch, dataformats='HWC')   # probably reverse channel
    writer.add_image('gt_env_map', torch.e ** gt_rgb_vals[0].view(current_resolution, 2 * current_resolution, 3),
                     global_step=epoch, dataformats='HWC')  # probably reverse channel

    if epoch % 100 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
            'current_resolution': current_resolution
        }, str(_save_path / f'checkpoint_{epoch}.pt'))
