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
from keras.models import Model
from keras.datasets import cifar10

from utils import *
from wide_resnet import *
from cosine_annealing import CosineAnnealingScheduler


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--depth', default=28, type=int)
    parser.add_argument('--width', default=2, type=int)
    parser.add_argument('--scheduler', default=None,
                        choices=['CosineAnnealing'],
                        help='learning rate scheduler')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch-size', default=128, type=int)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.name is None:
        args.name = 'WideResNet%s-%s' %(args.depth, args.width)
        if args.scheduler == 'CosineAnnealing':
            args.name += '_wCosineAnnealing'

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # create model
    model = WideResNet(args.depth, args.width, num_classes=10)
    model.compile(loss='categorical_crossentropy',
            optimizer=SGD(lr=0.1, momentum=0.9),
            metrics=['accuracy'])

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = standardize(x_train)
    x_test = standardize(x_test)

    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    callbacks = [
        ModelCheckpoint('models/%s/model.hdf5'%args.name, verbose=1, save_best_only=True),
        CSVLogger('models/%s/log.csv'%args.name)
    ]

    if args.scheduler == 'CosineAnnealing':
        callbacks.append(CosineAnnealingScheduler(T_max=args.epochs, eta_max=0.05, eta_min=4e-4))
    else:
        callbacks.append(LearningRateScheduler(adjust_learning_rate))

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=args.batch_size),
                        steps_per_epoch=len(x_train)//args.batch_size,
                        validation_data=(x_test, y_test),
                        epochs=args.epochs, verbose=1, workers=4,
                        callbacks=callbacks)

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


if __name__ == '__main__':
    main()
