# -*- coding: utf-8 -*-

from importlib import import_module


def get_model(args):
    module = import_module('models.' + args.model_name.lower())
    return module.make_model(args)