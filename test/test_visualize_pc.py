
import unittest

import numpy as np

from datasets import reconstruction_dataset
from visualization import visualize


class TestVisualizePC(unittest.TestCase):

    def test_viz(self):

        dataset = reconstruction_dataset.ReconstructionDataset()

        num_batches = 10
        batch_size = 5

        iterator = dataset.iterator(batch_size=batch_size, num_batches=num_batches)

        pc = np.load(iterator.dataset.pointclouds[1][0])

        visualize.visualize_pointcloud(pc)

        import IPython
        IPython.embed()


if __name__ == '__main__':

    unittest.main()