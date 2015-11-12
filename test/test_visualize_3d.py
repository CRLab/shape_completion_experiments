import random
import math
import unittest
import os
import numpy as np
from datasets import model_net_dataset, point_cloud_hdf5_dataset
from visualization import visualize
from layers.layer_utils import *
import time


class TestPointCloudDataset(unittest.TestCase):
    # def test_modelnet_visualizaton(self):
    #     models_dir = '/srv/3d_conv_data/ModelNet10'
    #     patch_size = 256
    #
    #     dataset = model_net_dataset.ModelNetDataset(models_dir, patch_size)
    #
    #     num_batches = 1
    #     batch_size = 30
    #
    #     iterator = dataset.iterator(batch_size=batch_size,
    #                                      num_batches=num_batches,
    #                                      mode='even_shuffled_sequential')
    #
    #     for i in range(batch_size):
    #         batch_x, batch_y = iterator.next()
    #
    #         batch_x = downscale_3d(batch_x, 8)
    #
    #         visualize.visualize_batch_x(batch_x, i)
    #
    #         import IPython
    #         IPython.embed()


    def test_grasp_visualizaton(self):
        hdf5_filepath = '/srv/3d_conv_data/training_data/' + \
                        'contact_and_potential_grasps-3_23_15_34-3_23_16_35.h5'
        topo_view_key = 'rgbd'
        y_key = 'grasp_type'
        patch_size = 32

        dataset = point_cloud_hdf5_dataset.PointCloud_HDF5_Dataset(
            topo_view_key,
            y_key,
            hdf5_filepath,
            patch_size)

        num_batches = 10
        batch_size = 5

        iterator = dataset.iterator(batch_size=batch_size,
                                    num_batches=num_batches,
                                    mode='even_shuffled_sequential')

        for i in range(batch_size):

            batch_x, batch_y = iterator.next()
            # batch_x_down_sample = downscale_3d(batch_x, 8)

            for j in range(batch_size):
                visualize.visualize_batch_x(batch_x, j)
                # visualize.visualize_batch_x(batch_x_down_sample, j)
                time.sleep(1)

                import IPython
                IPython.embed()


if __name__ == '__main__':
    unittest.main()
