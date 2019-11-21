import shutil
from collections import OrderedDict

import torch
import argparse
from torch.autograd import Variable
from os import makedirs
from os.path import exists, join, basename, dirname

from model.cnn_geometric_model import CNNGeometric


class BatchTensorToVars(object):
    """Convert tensors in dict batch to vars
    """

    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda

    def __call__(self, batch):
        batch_var = {}
        for key, value in batch.items():
            batch_var[key] = Variable(value, requires_grad=False)
            if self.use_cuda:
                batch_var[key] = batch_var[key].cuda()

        return batch_var


def save_checkpoint(state, is_best, file):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir != '' and not exists(model_dir):
        makedirs(model_dir)
    torch.save(state, file)
    if is_best:
        shutil.copyfile(file, join(model_dir, 'best_' + model_fn))


def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_torch_model(args, use_cuda):
    model = CNNGeometric(use_cuda=use_cuda, geometric_model='affine',
                         feature_extraction_cnn=args.feature_extraction_cnn)

    # Load trained weights
    print('Loading trained model weights...')
    checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict(
        [(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
    model.load_state_dict(checkpoint['state_dict'])
    return model
