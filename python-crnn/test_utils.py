#!/usr/bin/python
# encoding: utf-8

import sys
import unittest
import torch
import collections
from lib import dataset
from lib.make_lmdb import check_image_valid
from lib import utils
from lib.make_lmdb import create_dataset


if __name__ == "__main__":
    output_path = '/mnt/data1/data/guyu.gy/DigitDataset/test'
    images_path = '/mnt/data1/data/guyu.gy/DigitDataset/images'
    dataset_file_path = '/mnt/data1/data/guyu.gy/DigitDataset/testset.txt'
    create_dataset(output_path, images_path, dataset_file_path)
#    test_dataset = dataset.lmdbDataset('/mnt/data1/data/guyu.gy/SyntheticChineseStringDataset/train')
#    num = 1
#    img, label = test_dataset[1]
#    img.save('./data/' + str(num) + '.jpg')
#    print('label: {}'.format(label))
