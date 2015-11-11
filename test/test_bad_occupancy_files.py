import unittest
import numpy as np
import theano
from theano.tensor.nnet.conv3d2d import *
import os
import binvox_rw


class TestBadOccupancyFiles(unittest.TestCase):
    def test(self):
        models_dir = '/srv/3d_conv_data/ModelNet10'

        categories = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        examples = []
        subdir = '/' + 'train' + '/'
        for category in categories:
            for file_name in os.listdir(models_dir + '/' + category + subdir):
                if ".binvox" in file_name:
                    examples.append(models_dir + '/' + category + subdir + file_name)

        subdir = '/' + 'test' + '/'
        for category in categories:
            for file_name in os.listdir(models_dir + '/' + category + subdir):
                if ".binvox" in file_name:
                    examples.append(models_dir + '/' + category + subdir + file_name)

        for example in examples:
            with open(example, 'rb') as f:
                model = binvox_rw.read_as_3d_array(f).data

                if model.max() == 0:
                    print example


if __name__ == '__main__':
    unittest.main()
