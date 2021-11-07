import csv
import argparse
from trainer import Trainer
from util import load_dataset, make_iter, Params
import torch
import json
import numpy as np
import utils

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def main(config):
    params = Params('config/params.json')

    if config.mode == 'train':
        vocab, train_data, valid_data = load_dataset(config.mode)
        print("load")
        train_iter, valid_iter = make_iter(params.batch_size, config.mode,
                                           train_data=train_data, valid_data=valid_data)

        print("make_iter")
        trainer = Trainer(params, config.mode, train_iter=train_iter, valid_iter=valid_iter)
        trainer.train(vocab)

    else:
        test_data = load_dataset(config.mode)
        test_iter = make_iter(params.batch_size, config.mode, test_data=test_data)

        trainer = Trainer(params, config.mode, test_iter=test_iter)
        trainer.inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Neural Machine Translation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()
    main(args)
