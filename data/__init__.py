# -*- coding: utf-8 -*-

import utils
from data.samplers import RASampler
from importlib import import_module
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DistributedSampler, DataLoader


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_loader(args, attr):
    dataset = getattr(import_module('data.' + args.dataset.lower()), args.dataset)(args, attr)

    if args.distributed and attr == 'train':
        global_rank = utils.get_rank()
        num_task = utils.get_world_size()
        batch_size = int(args.batch_size // args.num_gpus)
        if args.repeated_aug:   # 一个mini-batch中可以包含来自同一个图像的不同增强版本
            data_sampler = RASampler(dataset, num_replicas=num_task, rank=global_rank, shuffle=True)
        else:
            data_sampler = DistributedSampler(dataset, num_replicas=num_task, rank=global_rank, shuffle=True)
    else:
        data_sampler = None
        batch_size = args.batch_size

    shuffle = (data_sampler is None) and (attr == 'train')

    batch_size = batch_size if attr == 'train' else 1
    num_workers = min(args.num_workers, batch_size)
    data_loader = DataLoaderX(dataset, batch_size=batch_size, num_workers=num_workers, sampler=data_sampler,
                             pin_memory=True, drop_last=(attr == 'train'), shuffle=shuffle)
    return data_loader

# from options import args
# args.distributed = True
# data = get_loader(args, 'val')
#
# for _, sample in enumerate(data):
#     print(sample['img_gt'].size())