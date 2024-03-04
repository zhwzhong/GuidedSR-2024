# -*- coding: utf-8 -*-

import os
import cv2
import math
import time
import torch
import shutil
import random
import itertools
import numpy as np
from .dist import master_only
from collections import Iterable
from .image_resize import imresize
from loss import pytorch_ssim
from .self_ensemble import self_ensemble, self_ensemble_guided


@master_only
def create_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def torch_psnr(img1, img2, border, data_range):
    if border != 0:
        img1 = img1[:, :, border: -border, border: -border]
        img2 = img2[:, :, border: -border, border: -border]

    img1, img2 = img1 * (255 / data_range), img2 * (255 / data_range)

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    else:
        return 20 * torch.log10_(255.0 / torch.sqrt(mse))


def calc_metrics(img_out, img_gt, args):
    # 必须是tensor 才能支持分布式
    sum_psnr = 0
    sum_ssim = 0

    for index in range(img_gt.size(0)):
        sum_psnr += torch_psnr(img_out[index: index + 1], img_gt[index: index + 1], border=6, data_range=args.data_range)
        sum_ssim += pytorch_ssim.ssim(img_out[index: index + 1], img_gt[index: index + 1])

    metrics = {'PSNR': sum_psnr / img_gt.size(0), 'SSIM': sum_ssim / img_gt.size(0)}
    return metrics


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# def add_jpg_compression(img, quality=90):
#     """Add JPG compression artifacts.
#     Args:
#         img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
#         quality (float): JPG compression quality. 0 for lowest quality, 100 for
#             best quality. Default: 90.
#     Returns:
#         (Numpy array): Returned image after JPG, shape (h, w, c), range[0, 1],
#             float32.
#     """
#     img = np.clip(img, 0, 1)
#     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
#     _, encimg = cv2.imencode('.jpg', img * 255., encode_param)
#     img = np.float32(cv2.imdecode(encimg, 1)) / 255.
#     return img


def down_sample(img, scale, method='cv2', quality=95):
    noisy_image = img + np.random.normal(0, 10 ** 0.5, img.shape)
    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)

    noisy_image = cv2.resize(noisy_image, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_CUBIC)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc_img = cv2.imencode('.jpg', noisy_image, encode_param)
    noisy_image = cv2.imdecode(enc_img, 0)
    #
    return noisy_image


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums, )

    return clever_nums


def to_device(sample, device):
    for key, value in sample.items():
        if key != 'img_name':
            sample[key] = value.to(device, non_blocking=True)
    return sample


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_num, trainable_num = clever_format([total_num, trainable_num])
    return {'Total': total_num, 'Trainable': trainable_num}


def mix_up(samples, alpha, prob=0.7):
    gt_img = samples['img_gt']

    if np.random.rand(1) < prob and alpha > 0:
        lam = np.random.beta(alpha, alpha)
        batch_size = gt_img.size(0)
        index = torch.randperm(batch_size).to(gt_img.device)

        for key, value in samples.items():
            if key != 'img_name':
                samples[key] = lam * value + (1 - lam) * value[index]
    return samples


def ensemble(samples, model, ensemble_mode='mean', dataset='PBVS'):
    if dataset in ['PBVS']:
        return self_ensemble(samples, model, ensemble_mode)
    else:
        return self_ensemble_guided(samples, model, ensemble_mode)


def tensor2uint(*args, data_range):
    def _tensor2uint(img):
        img = img.data.squeeze().float().clamp_(0, data_range).cpu().numpy()
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        return np.uint8((img * (255.0 / data_range)).round())

    out = [_tensor2uint(a) for a in args]
    return out if len(out) > 1 else out[0]


@master_only
def set_checkpoint_dir(args):
    if args.test_only or args.resume:
        return False
    print('Removing Previous Checkpoints and Get New Checkpoints Dir')
    create_dir('./logs/{}/{}'.format(args.dataset, args.file_name))

    create_dir('./checkpoints/{}/{}'.format(args.dataset, args.file_name))


def get_coord(H, W, x=448/3968, y=448/2976):
    x_coord = np.linspace(-x + (x / W), x - (x / W), W)
    x_coord = np.expand_dims(x_coord, axis=0)
    x_coord = np.tile(x_coord, (H, 1))
    x_coord = np.expand_dims(x_coord, axis=0)

    y_coord = np.linspace(-y + (y / H), y - (y / H), H)
    y_coord = np.expand_dims(y_coord, axis=1)
    y_coord = np.tile(y_coord, (1, W))
    y_coord = np.expand_dims(y_coord, axis=0)

    coord = np.ascontiguousarray(np.concatenate([x_coord, y_coord]))
    coord = np.float32(coord)
    return torch.from_numpy(coord)