import random
import math
import unittest
import os

import numpy as np

from datasets import model_net_dataset
from visualization.visualize import *


class TestModelNetDataset(unittest.TestCase):

    def setUp(self):

        self.models_dir = '/srv/3d_conv_data/ModelNet10'
        self.patch_size = 256

        self.dataset = model_net_dataset.ModelNetDataset(self.models_dir,
                                                           self.patch_size)

    def test_iterator(self):

        num_batches = 1
        num_channels = 1

        batch_size = 1

        iterator = self.dataset.iterator(batch_size=batch_size,
                                         num_batches=num_batches,
                                         mode='even_shuffled_sequential')

        batch_x, batch_y = iterator.next()

        visualize_batch_x(batch_x)

        import IPython
        IPython.embed()

        self.assertEqual(batch_x.shape, (batch_size, self.patch_size, num_channels, self.patch_size, self.patch_size))


if __name__ == '__main__':
    unittest.main()
