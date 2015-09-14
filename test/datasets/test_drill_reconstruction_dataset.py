import random
import math
import unittest
import os
import time
import numpy as np
import visualization.visualize as viz
import matplotlib.pyplot as plt

from datasets import drill_reconstruction_dataset


class TestPointCloudDataset(unittest.TestCase):

    def setUp(self):

        self.dataset = drill_reconstruction_dataset.DrillReconstructionDataset(patch_size=32)

    def test_iterator(self):

        num_batches = 1
        batch_size = 1


        iterator = self.dataset.iterator(batch_size=batch_size,
                                         num_batches=num_batches)
        s = time.time()
        batch_x, batch_y = iterator.next()
        e = time.time()

        print "total time:" + str(e-s)

        viz.visualize_batch_x(batch_x, 0)
        viz.visualize_batch_x(batch_y, 0)

        import IPython
        IPython.embed()



if __name__ == '__main__':
    unittest.main()