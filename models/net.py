# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   nafnet.py
@Time    :   2023/2/1 20:08
@Desc    :
"""
import os
import torch
import torch.nn as nn
from models.common import ConvBNReLU2D, Scale
from models.base3 import Net, NAFBlock, NAFNet, NAFNetLocal


class SNet(nn.Module):
    def __init__(self, args):
        super(SNet, self).__init__()
        self.args = args
        self.args = args
        enc_blks = [2, 2, 2, 4]
        middle_blk_num = 6
        dec_blks = [4, 2, 2, 2]
        train_size = (1, 3, args.patch_size, args.patch_size)
        if args.test_only and args.tlc_enhance:
            self.net = NAFNetLocal(img_channel=args.embed_dim, width=args.embed_dim, middle_blk_num=middle_blk_num,
                                   enc_blk_nums=enc_blks,
                                   dec_blk_nums=dec_blks, train_size=train_size)
        else:
            self.net = NAFNet(img_channel=args.embed_dim, width=args.embed_dim, middle_blk_num=middle_blk_num,
                              enc_blk_nums=enc_blks,
                              dec_blk_nums=dec_blks)

    def forward(self, x):
        return self.net(x)


class BasicNet(nn.Module):
    def __init__(self, args):
        super(BasicNet, self).__init__()

        self.basic_model = Net(args)
        if args.pre_trained and os.path.exists(args.model_path):
            load_net = torch.load(args.model_path, map_location=lambda storage, loc: storage)
            self.basic_model.load_state_dict(load_net['model'])
            print('Successfully load pre-trained model ...')
        self.scale = Scale(init_value=1)
        if args.freeze_params:
            for i in self.basic_model.parameters():
                i.requires_grad = False

    def forward(self, samples):
        return self.scale(self.basic_model(samples)['img_out'])


class MoE(nn.Module):
    def __init__(self, args):
        super(MoE, self).__init__()
        self.args = args

        self.layers = nn.ModuleList()

        for _ in range(args.num_moe):
            self.layers.append(BasicNet(args))

        in_channels = args.num_moe + 1 if args.dataset == 'PBVS' else args.num_moe + 4
        self.mix_model = nn.Sequential(
            ConvBNReLU2D(in_channels, args.embed_dim, kernel_size=5, stride=1, padding=2, act='PReLU'),
            SNet(args),
            ConvBNReLU2D(args.embed_dim, 1, kernel_size=5, stride=1, padding=2, act='ReLU'),
        )

    def forward(self, samples):
        out = []
        for i in range(self.args.num_moe):
            if self.args.dataset == 'PBVS':
                inputs = {
                    'lr_up': torch.rot90(samples['lr_up'], i, [2, 3])
                }
            else:
                inputs = {
                    'lr_up': torch.rot90(samples['lr_up'], i, [2, 3]),
                    'img_rgb': torch.rot90(samples['img_rgb'], i, [2, 3])
                }

            out.append(torch.rot90(self.layers[i](inputs), i, [3, 2]))

        add_mean = torch.mean(torch.cat(tuple(out), dim=1), dim=1, keepdim=True)
        out.append(samples['lr_up'])
        if self.args.dataset == 'NIR':
            out.append(samples['img_rgb'])
        out = torch.cat(tuple(out), dim=1)
        # return {'img_out': self.mix_model(out) + samples['lr_up']}
        return {'img_out': self.mix_model(out) + add_mean}


def make_model(args): return MoE(args)
