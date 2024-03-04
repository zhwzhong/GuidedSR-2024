# -*- coding: utf-8 -*-

import torch
import itertools

def transform(*args, xflip, yflip, transpose, reverse=False):

    def _transform(img):
        if not reverse:  # forward transform
            if xflip: img = torch.flip(img, [3])
            if yflip: img = torch.flip(img, [2])
            if transpose: img = torch.transpose(img, 2, 3)
        else:  # reverse transform
            if transpose: img = torch.transpose(img, 2, 3)
            if yflip: img = torch.flip(img, [2])
            if xflip: img = torch.flip(img, [3])
        return img
    out = [_transform(a) for a in args]
    return out if len(out) > 1 else out[0]


def self_ensemble(samples, model, ensemble_mode='mean'):
    outputs = []
    # tmp_lr = samples['img_lr'].clone()
    tmp_lr_up = samples['lr_up'].clone()
    opts = itertools.product((False, True), (False, True), (False, True))
    for x_flip, y_flip, transpose in opts:
        # samples['img_lr'] = transform(tmp_lr.clone(), xflip=x_flip, yflip=y_flip, transpose=transpose)
        samples['lr_up'] = transform(tmp_lr_up.clone(), xflip=x_flip, yflip=y_flip, transpose=transpose)
        out_img = model(samples)['img_out']
        outputs.append(transform(out_img, xflip=x_flip, yflip=y_flip, transpose=transpose, reverse=True))

    if ensemble_mode == 'mean':
        out_img = torch.stack(outputs, 0).mean(0)
    elif ensemble_mode == 'median':
        out_img = torch.stack(outputs, 0).median(0)[0]
    else:
        raise ValueError("Unknown ensemble mode %s." % ensemble_mode)
    return {'img_out': out_img}


def transform_guided(*args, xflip, yflip, transpose, reverse=False):
    def _transform(img):
        if not reverse:  # forward transform
            if xflip: img = torch.flip(img, [3])
            if yflip: img = torch.flip(img, [2])
            if transpose: img = torch.transpose(img, 2, 3)
        else:  # reverse transform
            if transpose: img = torch.transpose(img, 2, 3)
            if yflip: img = torch.flip(img, [2])
            if xflip: img = torch.flip(img, [3])
        return img
    out = [_transform(a) for a in args]
    return out if len(out) > 1 else out[0]


def self_ensemble_guided(samples, model, ensemble_mode='mean'):
    outputs = []
    tmp_lr_up = samples['lr_up'].clone()
    tmp_color = samples['img_rgb'].clone()
    opts = itertools.product((False, True), (False, True), (False, True))
    for x_flip, y_flip, transpose in opts:
        samples['lr_up'], samples['img_rgb'] = transform_guided(tmp_lr_up.clone(), tmp_color.clone(), xflip=x_flip, yflip=y_flip,
                                                                            transpose=transpose)
        out_img = model(samples)['img_out']
        outputs.append(transform_guided(out_img, xflip=x_flip, yflip=y_flip, transpose=transpose, reverse=True))

    if ensemble_mode == 'mean':
        out_img = torch.stack(outputs, 0).mean(0)
    elif ensemble_mode == 'median':
        out_img = torch.stack(outputs, 0).median(0)[0]
    else:
        raise ValueError("Unknown ensemble mode %s." % ensemble_mode)
    return {'img_out': out_img}