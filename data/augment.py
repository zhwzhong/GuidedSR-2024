# -*- coding: utf-8 -*-

import cv2
import torch
import random
import numpy as np
from utils import imresize


def random_downsample(img, scale):
    pass


def get_patch(*args, patch_size=32, scale=1):
    """
    :param args: (LR, HR, ..)
    :param patch_size: LR Patch Size
    :param scale: HR // LR
    :return: (LR, HR, ..)
    """
    ih, iw = args[0].shape[1:]
    tp = patch_size * scale
    ip = tp // scale

    iy = random.randrange(0, ih - ip + 1)
    ix = random.randrange(0, iw - ip + 1)
    tx, ty = scale * ix, scale * iy
    ret = [
        args[0][:, iy:iy + ip, ix:ix + ip],
        *[a[:, ty:ty + tp, tx:tx + tp] for a in args[1:]]
    ]

    return ret if len(ret) > 1 else ret[0]


def random_rot(*args, hflip=True, rot=True):
    """
    Input: (C, H, W)
    :param args:
    :param hflip:
    :param rot:
    :return:
    """
    hflip = hflip and random.random() < 0.5
    vflip = hflip and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, :, ::-1]
        if vflip: img = img[:, ::-1, :]
        if rot90: img = img.transpose(0, 2, 1)
        return np.ascontiguousarray(img)

    out = [_augment(a) for a in args]
    return out if len(out) > 1 else out[0]


def np_to_tensor(*args, input_data_range=1.0, process_data_range=1.0):
    def _np_to_tensor(img):
        np_transpose = img.astype(np.float32)
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(process_data_range / input_data_range)
        return tensor.float()

    out = [_np_to_tensor(a) for a in args]
    return out if len(out) > 1 else out[0]


def add_jpg_compression(img, quality=90):
    """Add JPG compression artifacts.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        quality (float): JPG compression quality. 0 for lowest quality, 100 for
            best quality. Default: 90.
    Returns:
        (Numpy array): Returned image after JPG, shape (h, w, c), range[0, 1],
            float32.
    """
    img = np.clip(img, 0, 1)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img * 255., encode_param)
    img = np.float32(cv2.imdecode(encimg, 1)) / 255.
    return img