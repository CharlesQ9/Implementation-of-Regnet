# Aaron Low (2020) RegNet code (Version 1.0) [Source code]. https://github.com/aaronlws95/regnet
"""
Title: RegNet Train code
Author: Aaron Low
Date: 2020
Code version: 1.0
Availability: https://github.com/aaronlws95/regnet
"""
# modified to fit our model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
from pathlib import Path

import src.modelfusion as mod
from src.dataset import Kitti_Dataset
import src.dataset_params as dp

# Setup
os.environ['TORCH_HOME'] = os.path.join('C:\\', 'machine_learning')
start_epoch = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
torch.backends.cudnn.enabled = False

# Config
RUN_ID = 5
SAVE_PATH = str(Path('data_fusionlearn')/'checkpoints_new' /
                'run_{:05d}'.format(RUN_ID))
LOG_PATH = str(Path('data_fusionlearn')/'tensorboard_new' /
               'run_{:05d}'.format(RUN_ID))
Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
Path(LOG_PATH).mkdir(parents=True, exist_ok=True)

# Hyperparameters
LEARNING_RATE = 3e-4
EPOCHS = 200
BATCH_SIZE = 1
SAVE_RATE = 500
LOG_RATE = 10
QUAT_FACTOR = 1

# Dataset
# modified to fit our model
dataset_params = {
    'base_path': dp.TRAIN_SET_2011_09_26['base_path'],
    'date': dp.TRAIN_SET_2011_09_26['date'],
    'drives': dp.TRAIN_SET_2011_09_26['drives'],
    'd_rot': 20,
    'd_trans': 1.5,
    'fixed_decalib': False,
    'resize_w': 1216,
    'resize_h': 352,
}

dataset = Kitti_Dataset(dataset_params)
train_loader = DataLoader(dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

# Model
model = mod.RegNet_v3()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.cuda()

# Load
LOAD_RUN_ID = 5
LOAD_MODEL_ID = 171800
LOAD_PATH = str(Path('data_fusionlearn')/'checkpoints_new' /
                'run_{:05d}'.format(LOAD_RUN_ID)/'model_{:05d}.pth'.format(LOAD_MODEL_ID))
model_load = torch.load(LOAD_PATH)
model.load_state_dict(model_load['model_state_dict'])
start_epoch = model_load['epoch']
model.cuda()

# Tensorboard
writer = SummaryWriter(log_dir=LOG_PATH)

# Train
training_params = {
    'dataset_params': dataset_params,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
}

running_loss = 0.0
model.train()
for epoch in range(start_epoch, EPOCHS):
    for i, data in enumerate(train_loader):
        # Load data
        rgb_img = data['rgb'].cuda()
        depth_img = data['depth'].cuda()
        decalib_quat_real = data['decalib_real_gt'].cuda()
        decalib_quat_dual = data['decalib_dual_gt'].cuda()

        # Forward pass
        out, sx, sq = model(rgb_img, depth_img)  # modified to fit our model

        # Zero optimizer
        optimizer.zero_grad()

        # Calculate loss
        # modified to fit our new loss function
        real_loss = criterion(out[:, :4], decalib_quat_real)
        dual_loss = criterion(out[:, 4:], decalib_quat_dual)
        loss = real_loss*torch.exp(-sx)+sx+dual_loss*torch.exp(-sq)+sq

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Logging
        running_loss += loss.item()
        n_iter = epoch * len(train_loader) + i
        if n_iter % LOG_RATE == 0:
            print('Epoch: {:5d} | Batch: {:5d} | Loss: {:03f}'.format(
                epoch + 1, i + 1, running_loss / LOG_RATE))
            writer.add_scalar('Loss/train', running_loss / LOG_RATE, n_iter)
            running_loss = 0.0

        # Save model
        if n_iter % SAVE_RATE == 0:
            model_save = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_params': training_params,
            }
            torch.save(model_save, SAVE_PATH +
                       '/model_{:05d}.pth'.format(n_iter))

# Save final model
model_save = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'training_params': training_params,
}
torch.save(model_save, SAVE_PATH + '/model_{:05d}.pth'.format(n_iter))
