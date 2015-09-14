
import numpy as np
import os
import collections

import binvox_rw
import visualization.visualize as viz
import tf_conversions
import PyKDL
import h5py
from operator import mul


import math

class ReconstructionDataset():

    def __init__(self,
                 hdf5_filepath='/srv/3d_conv_data/22_model_big_bird_1000_rot_24x24x24.h5'):

        self.dset = h5py.File(hdf5_filepath, 'r')

        self.num_examples = self.dset['x'].shape[0]
        self.patch_size = self.dset['x'].shape[1]

    def get_num_examples(self):
        return self.num_examples

    def iterator(self,
                 batch_size=None,
                 num_batches=None,
                 flatten_y=True):

            return ReconstructionIterator(self,
                                          batch_size=batch_size,
                                          num_batches=num_batches,
                                          flatten_y=flatten_y)


class ReconstructionIterator(collections.Iterator):

    def __init__(self,
                 dataset,
                 batch_size,
                 num_batches,
                 iterator_post_processors=[],
                 flatten_y=True):

        self.dataset = dataset

        self.batch_size = batch_size
        self.num_batches = num_batches
        self.flatten_y = flatten_y

        self.iterator_post_processors = iterator_post_processors

    def __iter__(self):
        return self

    def next(self):

        batch_indices = np.random.random_integers(0, self.dataset.get_num_examples()-1, self.batch_size)

        patch_size = self.dataset.patch_size

        batch_x = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1), dtype=np.float32)
        batch_y = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1), dtype=np.float32)

        for i in range(len(batch_indices)):
            index = batch_indices[i]

            x = self.dataset.dset['x'][index]
            y = self.dataset.dset['y'][index]

            # viz.visualize_3d(x)
            # viz.visualize_3d(y)
            # viz.visualize_pointcloud(pc2_out[0:3, :].T)

            batch_y[i, :, :, :, :] = y
            batch_x[i, :, :, :, :] = x

        #make batch B2C01 rather than B012C
        batch_x = batch_x.transpose(0, 3, 4, 1, 2)
        batch_y = batch_y.transpose(0, 3, 4, 1, 2)

        if self.flatten_y:
            batch_y = batch_y.reshape(batch_y.shape[0], reduce(mul, batch_y.shape[1:]))

        #apply post processors to the patches
        for post_processor in self.iterator_post_processors:
            batch_x, batch_y = post_processor.apply(batch_x, batch_y)

        return batch_x, batch_y

    def batch_size(self):
        return self.batch_size

    def num_batches(self):
        return self.num_batches

    def num_examples(self):
        return self.dataset.get_num_examples()

