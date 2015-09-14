
import unittest

from datasets import melonoma_dataset

class TestModelNetDataset(unittest.TestCase):

    def setUp(self):

        self.data_dir = '/srv/3d_conv_data/melanoma/suborder-5/'
        self.dataset = melonoma_dataset.MelonomaDataset(self.data_dir)

    def test_iterator(self):

        num_batches = 1
        num_channels = 1
        batch_size = 1

        iterator = self.dataset.iterator(batch_size=batch_size,
                                         num_batches=num_batches,
                                         mode='even_shuffled_sequential')

        batch_x, batch_y = iterator.next()

        self.assertEqual(batch_x.shape, (batch_size, 256, num_channels, 6, 256))


if __name__ == '__main__':
    unittest.main()
