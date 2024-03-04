# -*- coding: utf-8 -*-


import os
import json
import loss
import utils
import torch
from options import args
from data import get_loader
from models import get_model
from scheduler import create_scheduler
from trainer import train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter
from utils import make_optimizer, set_checkpoint_dir


def main():
    model = get_model(args)
    set_checkpoint_dir(args)
    device = torch.device(args.device)

    writer = SummaryWriter('./logs/{}/{}'.format(args.dataset, args.file_name))
    # model.to(device)
    criterion = loss.Loss(args)
    # vgg_loss = VGGPerceptualLoss().to(device)

    optimizer = make_optimizer(args, model)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    print('===> Parameter Number:', utils.get_parameter_number(model))
    cp_path = f'./checkpoints'
    if args.resume or args.test_only or args.pre_trained:
        model_path = args.load_name if os.path.exists(args.load_name) else f"{cp_path}/{args.load_name}"
        try:
            if len(args.model_path) > 1 and args.pre_trained:
                pass
            else:
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model'])
                if args.resume:
                    args.start_epoch = checkpoint['epoch'] + 1
                    lr_scheduler.step(args.start_epoch)
                    # optimizer.load_state_dict(checkpoint['optimizer'])
        except FileNotFoundError:
            print('===> File {} not exists'.format(model_path))
        else:
            print('===> File {} loaded'.format(model_path))
        evaluate(model.to(device), criterion, args.test_name, device=device, val_data=get_loader(args, args.test_name), args=args)

    model.to(device)
    if args.distributed:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    else:
        model = torch.nn.parallel.DataParallel(model, device_ids=list(range(args.num_gpus)))

    model_without_ddp = model.module
    best_psnr = 0
    if not args.test_only:
        train_data = get_loader(args, 'train')
        for epoch in range(args.start_epoch, num_epochs):
            if args.distributed:
                train_data.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(model, criterion, train_data, optimizer, device, epoch, args)

            log_stats = {**{f'TRAIN_{k}'.upper(): v for k, v in train_stats.items()}}

            test_stats = evaluate(model, criterion, 'val', device=device, val_data=get_loader(args, 'val'), args=args)
            log_stats.update({**{f'{k}'.upper(): v for k, v in test_stats.items()}})

            if args.local_rank == 0:
                [writer.add_scalar(k.replace('_', '/'), v, epoch) for k, v in log_stats.items() if k != 'EPOCH']
                with open("./logs/{}/{}/log.txt".format(args.dataset, args.file_name), 'a') as f:
                    f.write(json.dumps(log_stats) + "\n")

            check_data = {
                    'optimizer': optimizer.state_dict(),
                    'model': model_without_ddp.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
            }

            if best_psnr < test_stats['val_PSNR']:
                best_psnr = test_stats['val_PSNR']
                utils.save_on_master(check_data, '{}/model_best.pth'.format(cp_path))

            utils.save_on_master(check_data, '{}/model_{}.pth'.format(cp_path, str(epoch).zfill(6)))
            utils.check_reserve(cp_path, args.check_history)

            lr_scheduler.step(epoch)
            torch.cuda.empty_cache()

if __name__ == '__main__':

    main()










