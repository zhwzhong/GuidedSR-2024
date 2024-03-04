# -*- coding: utf-8 -*-

import os
import cv2
import tqdm
import utils
import torch
import numpy as np
from torch.nn import functional as f


def train_one_epoch(model, criterion, train_data, optimizer, device, epoch, args):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for samples in metric_logger.log_every(train_data, args.print_freq, header):
        samples = utils.to_device(samples, device)
        out = model(utils.mix_up(samples, args.mix_alpha) if args.mix_up else samples)
        loss = criterion(out['img_out'], samples['img_gt'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if args.model_name.lower() in ['base5', 'base6', 'base7', 'base8']:
            metric_logger.update(lr1=optimizer.param_groups[1]["lr"])

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, test_name, val_data, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    sv_path = f'./result/{args.dataset}/x{args.scale}/'
    if args.save_result and args.local_rank == 0:
        utils.create_dir(sv_path)

    nb = len(val_data)

    val_data = enumerate(val_data)
    if args.local_rank in [-1, 0]:
        val_data = tqdm.tqdm(val_data, total=nb)  # 只在主进程打印进度条

    l1_loss = []
    vgg_loss = []
    for _, samples in val_data:
        samples = utils.to_device(samples, device)
        out = utils.ensemble(samples, model, args.ensemble_mode, args.dataset) if args.self_ensemble else model(samples)
        torch.cuda.synchronize()
        loss = criterion(out['img_out'], samples['img_gt'])
        metric_logger.update(loss=loss.item() * 1000)

        l1_loss.append(f.l1_loss(out['img_out'], samples['img_gt']).item())

        if args.save_result:
            for index in range(samples['img_gt'].size(0)):
                save_name = os.path.join(sv_path, samples['img_name'][0])
                img = utils.tensor2uint(out['img_out'][index: index + 1], data_range=args.data_range)
                cv2.imwrite(save_name, img)
                # print('Image Saved to {}'.format(save_name))
        metrics = utils.calc_metrics(out['img_out'], samples['img_gt'], args)
        for metric, value in metrics.items():
            metric_logger.meters[metric].update(value.item(), n=samples['img_gt'].size(0))

    metric_logger.synchronize_between_processes()
    torch.cuda.empty_cache()
    metric_out = {'{}_'.format(test_name) + k: round(meter.global_avg, 5) for k, meter in metric_logger.meters.items()}
    metric_out['L1'] = round(np.mean(l1_loss) * 1000, 5)
    metric_out['VGG'] = round(np.mean(vgg_loss) * 1000, 5)

    print(metric_out)
    return metric_out
