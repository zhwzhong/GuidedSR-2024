# -*- coding: utf-8 -*-

import torch.optim as optim


def get_group_parameters(model, lr, small_lr):
    params = list(model.named_parameters())
    # mix_model = ['mix_model']
    basic_model = ['basic_model']

    param_group = [
        {'params': [p for n, p in params if not any(nd in n for nd in basic_model)], 'lr': lr},
        {'params': [p for n, p in params if any(nd in n for nd in basic_model)], 'lr': small_lr},
    ]
    num1 = len(param_group[0]['params'])
    num2 = len(param_group[1]['params'])
    print(f'Total :{len(params)}, {num1} with {lr} LR{lr}, and {num2} with {small_lr} as lr')
    return param_group


def optimizer_base5(args, targets):

    if args.opt == 'AMSGrad':
        optimizer = optim.Adam(get_group_parameters(targets, args.lr, args.small_lr), lr=args.lr,
                               weight_decay=args.weight_decay, amsgrad=True)
    elif args.opt == 'AdamW':
        optimizer = optim.AdamW(get_group_parameters(targets, args.lr, args.small_lr), lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(get_group_parameters(targets, args.lr, args.small_lr), lr=args.lr,
                               weight_decay=args.weight_decay)

    return optimizer


def make_optimizer(args, targets):
    if args.model_name.lower() in ['base5', 'base6', 'base7', 'base8']:
        return optimizer_base5(args, targets)

    if args.opt == 'AMSGrad':
        optimizer = optim.Adam(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    elif args.opt == 'AdamW':
        optimizer = optim.AdamW(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return optimizer

