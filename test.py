import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict

import keras
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model, load_model
from keras.datasets import cifar10

from utils import *
from wide_resnet import *
from cosine_annealing import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    args = joblib.load('models/%s/args.pkl' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    # create model
    model = load_model('models/%s/model.hdf5'%args.name)

    _, (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32') / 255
    x_test = standardize(x_test)
    y_test = keras.utils.to_categorical(y_test, 10)

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


if __name__ == '__main__':
    main()
