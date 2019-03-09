import random
import math
from PIL import Image
import numpy as np

import torch


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def adjust_learning_rate(epoch):
    lr = 0.1
    if epoch >= 60:
        lr = 0.02
    if epoch >= 120:
        lr = 0.004
    if epoch >= 160:
        lr = 0.0008

    return lr


def standardize(x):
    means = np.array([0.4914009 , 0.48215896, 0.4465308]).reshape(1, 1, 1, 3)
    stds = np.array([0.24703279, 0.24348423, 0.26158753]).reshape(1, 1, 1, 3)

    x -= means
    x /= (stds + 1e-6)

    return x
