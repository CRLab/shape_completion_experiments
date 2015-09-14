import random
import math
import unittest
import os

import numpy as np
import visualization.visualize as viz
import matplotlib.pyplot as plt

from datasets import point_cloud_hdf5_dataset


class TestPointCloudDataset(unittest.TestCase):

    def setUp(self):

        #self.hdf5_filepath = '/srv/3d_conv_data/training_data/contact_and_potential_grasps_small.h5'
        self.hdf5_filepath = '/srv/3d_conv_data/training_data/contact_and_potential_grasps-3_23_15_34-3_23_16_35.h5'
        self.topo_view_key = 'rgbd'
        self.y_key = 'grasp_type'
        self.patch_size = 72

        self.dataset = point_cloud_hdf5_dataset.PointCloud_HDF5_Dataset(self.topo_view_key,
                                                           self.y_key,
                                                           self.hdf5_filepath,
                                                           self.patch_size)

    def test_iterator(self):

        num_batches = 4
        num_grasp_types = 8
        num_finger_types = 4
        num_channels = 1

        batch_size = 40

        iterator = self.dataset.iterator(batch_size=batch_size,
                                         num_batches=num_batches,
                                         mode='even_shuffled_sequential',
                                         rgb=True)

        batch_x, batch_y, rgbs = iterator.next()

        for i in range(batch_size):
            title = str(batch_y[i].argmin())
            viz.visualize_batch_x(batch_x, i, title, 'pc_data/out_' + str(i) + '.png')
            plt.clf()
            plt.imshow(rgbs[i])
            plt.savefig('pc_data/out_rgb' + str(i) + '.png')


        import IPython
        IPython.embed()

        self.assertEqual(batch_x.shape, (batch_size, self.patch_size, num_channels, self.patch_size, self.patch_size))
        self.assertEqual(batch_y.shape, (batch_size, num_finger_types * num_grasp_types))


if __name__ == '__main__':
    unittest.main()