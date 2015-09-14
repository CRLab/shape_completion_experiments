import datasets.big_bird_dataset
import unittest
import visualization.visualize
from  layers.layer_utils import downscale_3d

class TestBigbirdDataset(unittest.TestCase):

    def setUp(self):

        self.dataset = datasets.big_bird_dataset.BigBirdDataset()

    def test_iterator(self):

        num_batches = 1
        num_channels = 1

        batch_size = 1

        iterator = self.dataset.iterator(batch_size=batch_size,
                                         num_batches=num_batches)

        batch_x, batch_y = iterator.next()
        batch_x = downscale_3d(batch_x, 8)
        batch_y = downscale_3d(batch_y, 8)
        visualization.visualize.visualize_batch_x(batch_x)
        visualization.visualize.visualize_batch_x(batch_y)

        import IPython
        IPython.embed()

        self.assertEqual(batch_x.shape, (batch_size, self.patch_size, num_channels, self.patch_size, self.patch_size))


if __name__ == '__main__':
    unittest.main()
