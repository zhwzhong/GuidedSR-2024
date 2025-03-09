# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   nir.py
@Time    :   2023/2/1 20:01
@Desc    :
"""
import os
import cv2
import h5py
import numpy as np
from data import augment
from torch.utils.data import Dataset
from utils.image_resize import imresize

def get_array(x, cached):
    return np.array(x) if cached else x


class NIR(Dataset):
    def __init__(self, args, attr):
        self.args = args
        self.attr = attr

        self.file = h5py.File(f'./Data/gdsr_{attr}.h5', 'r')

        self.file = self.file if attr != 'test' else self.file[f'X{args.scale}']

        cached = (self.args.cached and attr == 'train')

        self.img_names = [key for key in self.file['GT'].keys()]

        self.lr_imgs = [get_array(self.file['LR'].get(key), cached=cached) for key in self.img_names]
        self.gt_imgs = [get_array(self.file['GT'].get(key), cached=cached) for key in self.img_names]
        self.rgb_imgs = [get_array(self.file['RGB'].get(key), cached=cached) for key in self.img_names]

    def __len__(self):
        return int(self.args.show_every * len(self.img_names)) if self.attr == 'train' else len(self.img_names)

    def __getitem__(self, item):
        item = item % len(self.gt_imgs)

        lr_img, gt_img, rgb_img = np.array(self.lr_imgs[item]), np.array(self.gt_imgs[item]), np.array(self.rgb_imgs[item])
        lr_img, gt_img, rgb_img = np.expand_dims(lr_img, 0), np.expand_dims(gt_img, 0), np.transpose(rgb_img, (2, 0, 1))
        # print(np.mean(gt_img))
        if self.attr == 'train':
            lr_img, gt_img, rgb_img = augment.get_patch(lr_img, gt_img, rgb_img, patch_size=self.args.patch_size, scale=self.args.scale)
            lr_img, gt_img, rgb_img = augment.random_rot(lr_img, gt_img, rgb_img, hflip=True, rot=True)


        lr_img = np.expand_dims(imresize(lr_img.astype(float).squeeze(), scalar_scale=self.args.scale) / 255, 0)

        gt_img, rgb_img = gt_img / 255, rgb_img / 255

        lr_img, gt_img, rgb_img = augment.np_to_tensor(lr_img, gt_img, rgb_img, input_data_range=1)

        return {'img_gt': gt_img, 'img_rgb': rgb_img, 'lr_up': lr_img, 'img_name': self.img_names[item]}
