import random
import math
import unittest
import os
import time
import numpy as np
import visualization.visualize as viz
import matplotlib.pyplot as plt

from datasets import reconstruction_dataset


class TestPointCloudDataset(unittest.TestCase):

    def setUp(self):

        self.dataset = reconstruction_dataset.ReconstructionDataset(patch_size=32)

    def test_iterator(self):

        num_batches = 1
        batch_size = 5


        iterator = self.dataset.iterator(batch_size=batch_size,
                                         num_batches=num_batches)
        s = time.time()
        batch_x, batch_y = iterator.next()
        e = time.time()

        print "total time:" + str(e-s)

        for i in range(batch_size):
            title = str(batch_y[i].argmin())
            viz.visualize_batch_x(batch_x, i, title, 'big_bird_recon/out_' + str(i) + '_x.png')
            viz.visualize_batch_x(batch_y, i, title, 'big_bird_recon/out_' + str(i) + '_y.png')

        import IPython
        IPython.embed()



if __name__ == '__main__':
    unittest.main()