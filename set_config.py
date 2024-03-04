# -*- coding: utf-8 -*-


def set_config(args):

    if args.sched == 'multistep':
        args.decay_epochs = [int(x) for x in args.decay_epochs.split('_')]
    else:
        args.decay_epochs = int(args.decay_epochs)
    args.file_name = args.model_name
