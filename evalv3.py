# Aaron Low (2020) Evaluation code (Version 1.0) [Source code]. https://github.com/aaronlws95/regnet
"""
Title: Evaluation code
Author: Aaron Low
Date: 2020
Code version: 1.0
Availability: https://github.com/aaronlws95/regnet
"""
# modified for generating the data for plot
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import src.utils as utils
import src.model as mod
from src.dataset import Kitti_Dataset
import src.dataset_params as dp
import numpy as np
from scipy.interpolate import interp1d

# Setup
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
RUN_ID = 5
MODEL_ID = 99000
SAVE_PATH = str(Path('data_v3')/'checkpoints_new' /
                'run_{:05d}'.format(RUN_ID)/'model_{:05d}.pth'.format(MODEL_ID))
print(SAVE_PATH)
# Dataset
dataset_list = []
x = []
for x_temp in range(1, 15, 1):
    dataset_params_temp = {
        'base_path': dp.TEST_SET_2011_09_30['base_path'],
        'date': dp.TEST_SET_2011_09_30['date'],
        'drives': dp.TEST_SET_2011_09_30['drives'],
        'd_rot': x_temp,
        'd_trans': 1.5,
        'fixed_decalib': True,
        'resize_w': 1216,
        'resize_h': 352,
    }
    dataset_list.append(dataset_params_temp)
    x.append(x_temp)

Y = []
for dataset_params in dataset_list:
    dataset = Kitti_Dataset(dataset_params)
    test_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False)

    # Model
    model = mod.RegNet_v3()
    # the following part is nearly not modified
    model_load = torch.load(SAVE_PATH)
    model.load_state_dict(model_load['model_state_dict'])
    model.cuda()

    mean_roll_error = 0
    mean_pitch_error = 0
    mean_yaw_error = 0
    mean_x_error = 0
    mean_y_error = 0
    mean_z_error = 0
    model.eval()
    with torch.no_grad():
        counter = 0
        for data in tqdm(test_loader):
            counter += 1
            rgb_img = data['rgb'].cuda()
            depth_img = data['depth'].cuda()
            out = model(rgb_img, depth_img)
            # print("out", out.shape)
            pred_decalib_quat_real = out[0][:4].cpu().numpy()
            pred_decalib_quat_dual = out[0][4:].cpu().numpy()

            gt_decalib_quat_real = data['decalib_real_gt'][0].numpy()
            gt_decalib_quat_dual = data['decalib_dual_gt'][0].numpy()

            init_extrinsic = data['init_extrinsic'][0]
            # print("pred_decalib_quat_real", pred_decalib_quat_real.shape)
            # print("pred_decalib_quat_dual", pred_decalib_quat_dual.shape)
            pred_decalib_extrinsic = utils.dual_quat_to_extrinsic(
                pred_decalib_quat_real, pred_decalib_quat_dual)
            inv_decalib_extrinsic = utils.inv_extrinsic(pred_decalib_extrinsic)
            pred_extrinsic = utils.mult_extrinsic(
                init_extrinsic, inv_decalib_extrinsic)

            gt_decalib_extrinsic = utils.dual_quat_to_extrinsic(
                gt_decalib_quat_real, gt_decalib_quat_dual)
            inv_decalib_extrinsic = utils.inv_extrinsic(gt_decalib_extrinsic)
            gt_extrinsic = utils.mult_extrinsic(
                init_extrinsic, inv_decalib_extrinsic)

            cur_roll_error, cur_pitch_error, cur_yaw_error, cur_x_error, cur_y_error, cur_z_error = utils.calibration_error(
                pred_extrinsic, gt_extrinsic)

            mean_roll_error += cur_roll_error
            mean_pitch_error += cur_pitch_error
            mean_yaw_error += cur_yaw_error
            mean_x_error += cur_x_error
            mean_y_error += cur_y_error
            mean_z_error += cur_z_error
# the following part is modified
    mean_roll_error /= counter
    mean_pitch_error /= counter
    mean_yaw_error /= counter
    mean_x_error /= counter
    mean_y_error /= counter
    mean_z_error /= counter

    print('Roll Error', mean_roll_error)
    print('Pitch Error', mean_pitch_error)
    print('Yaw Error', mean_yaw_error)
    print('X Error', mean_x_error)
    print('Y Error', mean_y_error)
    print('Z Error', mean_z_error)
    print('Mean Rotational Error', (mean_roll_error +
                                    mean_pitch_error + mean_yaw_error) / 3)
    print('Mean Translational Error',
          (mean_x_error + mean_y_error + mean_z_error) / 3)
    Y.append((mean_roll_error + mean_pitch_error + mean_yaw_error) / 3)

print("x", x)
print("y", Y)
x = np.array(x)
y = np.array(Y)
plt.plot(x, y)
plt.savefig("old_1_5_20.jpg")
