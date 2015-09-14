import datasets.geometric_3d_dataset
import unittest
import visualization.visualize
from  layers.layer_utils import downscale_3d

class TestGeometric3dDataset(unittest.TestCase):

    def setUp(self):

        self.dataset = datasets.geometric_3d_dataset.Geometric3DDataset()

    def test_iterator(self):

        num_batches = 1
        num_channels = 1

        batch_size = 1

        iterator = self.dataset.iterator(batch_size=batch_size,
                                         num_batches=num_batches)

        batch_x, batch_y = iterator.next()

        visualization.visualize.visualize_batch_x(batch_x)

        import IPython
        IPython.embed()

        self.assertEqual(batch_x.shape, (batch_size, self.patch_size, num_channels, self.patch_size, self.patch_size))


if __name__ == '__main__':
    unittest.main()
