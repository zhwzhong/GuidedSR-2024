# -*- coding: utf-8 -*-

import torch
import numpy as np
from .misc import time_since
from .dist import master_only


@torch.no_grad()
def torch_psnr(img1, img2, border, data_range):
    if border != 0:
        img1 = img1[:, :, border: -border, border: -border]
        img2 = img2[:, :, border: -border, border: -border]

    img1, img2 = img1 * (255 / data_range), img2* (255 / data_range)

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    else:
        return 20 * torch.log10_(255.0 / torch.sqrt(mse)).detach().cpu().numpy()


def metrics(im_pred, im_true, border, data_range):
    sum_psnr = 0
    for index in range(im_pred.size(0)):
        sum_psnr += torch_psnr(im_pred[index: index + 1], im_true[index: index + 1], border, data_range)

    return {'PSNR': sum_psnr / im_pred.size(0)}


@master_only
def update_summary(epoch, start_time, log_stats, tb_logger):
    try:
        log_info = f"Epoch: [{epoch}], Loss {log_stats['TRAIN/LOSS']:.5f}, "
    except KeyError:
        log_info = f"Epoch: [{epoch}], "
    for k, v in log_stats.items():
        tb_logger.add_scalar(k, v, epoch)
        if k.find('PSNR') != -1:
            log_info += f"{k}: {v:.8f}, "
    # print(log_stats)
    # for name, param in model.named_parameters():
    #     tb_logger.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    log_info += f"Time Spend {time_since(start_time)}"
    print(log_info)


def tensor2uint(*args, data_range):
    def _tensor2uint(img):
        img = img.data.squeeze().float().clamp_(0, data_range).cpu().numpy()
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        return np.uint8((img * (255.0 / data_range)).round())

    out = [_tensor2uint(a) for a in args]
    return out if len(out) > 1 else out[0]