# -*- coding: utf-8 -*-

import os
import yaml
import argparse
from set_config import set_config
from utils import init_distributed_mode, set_random_seed, get_gpu_info

config_parser = parser = argparse.ArgumentParser(description='Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch PBVS Challenge')
# Hardware
parser.add_argument('--device', default='cuda')
parser.add_argument('--sync_bn', action='store_true')

parser.add_argument('--seed', type=int, default=60)
parser.add_argument('--num_gpus', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=0)

# DataSet
parser.add_argument('--dataset', type=str, default='NIR')
parser.add_argument('--mean_shift', action='store_true') # 加上均值的差
parser.add_argument('--data_range', type=int, default=1) # LR Patch
parser.add_argument('--cached', action='store_true')
parser.add_argument('--repeated_aug', action='store_true')
parser.add_argument('--show_every', type=float, default=20)
parser.add_argument('--print_freq', type=int, default=1000)

# Optimizer

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--small_lr', type=float, default=1e-5)
parser.add_argument('--opt', type=str, default='Adam')
parser.add_argument('--loss', type=str, default='1*L1')
parser.add_argument('--hdelta', type=float, default=1) # HuberLoss
parser.add_argument('--trade_off', type=float, default=0.5) # HuberLoss

parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--sched', default='multistep', type=str) # cosine multistep
parser.add_argument('--weight_decay', type=float, default=0)

parser.add_argument('--warmup_epochs', type=int, default=0)
parser.add_argument('--cooldown_epochs', type=int, default=10)

parser.add_argument('--min_lr', type=float, default=1e-5)
parser.add_argument('--warmup_lr', type=float, default=1e-5)  # warmup 初始的LR，warmup-epoch以后变为设定的lr
parser.add_argument('--decay_rate',  type=float, default=0.5)
parser.add_argument('--decay_epochs', type=str, default='400')

parser.add_argument('--embed_dim', type=int, default=64)

# Training Stats
parser.add_argument('--resume', action='store_true')
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--pre_trained', action='store_true')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--load_name', type=str, default='model_best.pth')


parser.add_argument('--scale', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--patch_size', type=int, default=256) # LR Patch
parser.add_argument('--val_batch_size', type=int, default=1)

parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--model_name', type=str, default='Net')
parser.add_argument('--test_name', type=str, default='test')

parser.add_argument('--check_history', type=int, default=10)

parser.add_argument('--model_path', type=str, default='')

parser.add_argument('--mix_up', action='store_true')
parser.add_argument('--mix_alpha', type=float, default=0.1)
parser.add_argument('--self_ensemble', action='store_true')
parser.add_argument('--ensemble_mode', type=str, default='mean')

parser.add_argument('--tlc_enhance', action='store_true')

parser.add_argument('--num_moe', type=int, default=4)
parser.add_argument('--freeze_params', action='store_true')
# 分布式训练

parser.add_argument('--dist_url', default='env://')
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--local_rank', type=int, default=0)

parser.add_argument('--save_result', action='store_true')
parser.add_argument('--file_name', type=str, default='')
parser.add_argument('--static_model', action='store_true')

args_config, remaining = config_parser.parse_known_args()

if args_config.config:
    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

args = parser.parse_args(remaining)

# if get_gpu_info.get_memory(num_gpu=args.num_gpus) is False:
#     print('Out of the memory')
#     while True:
#         i = 999 * 9132877
# else:
#     os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(get_gpu_info.get_memory(num_gpu=args.num_gpus))

set_random_seed(args.seed)
init_distributed_mode(args)

set_config(args)





