import numpy as np
import os
import collections
# import binvox_rw
import visualization.visualize as viz
# import tf_conversions
# import PyKDL

import h5py
from operator import mul
import math


class ShrecHoldoutDataset():
    def __init__(self,
                 h5_dir,
                 model_names):
        self.dset = []
        self.num_examples = []
        self.patch_size = []
        for i, model_name in enumerate(model_names):
            self.dset.append(h5py.File(
                h5_dir + model_name + '/' + model_name + '.h5', 'r'))
            self.num_examples.append(self.dset[i]['x'].shape[0])
            self.patch_size.append(self.dset[i]['x'].shape[1])

    def get_num_examples(self):
        return self.num_examples

    def iterator(self,
                 batch_size=None,
                 num_batches=None,
                 flatten_y=True):
        return ShrecHoldoutIterator(self,
                                         batch_size=batch_size,
                                         num_batches=num_batches,
                                         flatten_y=flatten_y)


class ShrecHoldoutIterator(collections.Iterator):
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

        patch_size = self.dataset.patch_size[
            0]  # Since patch_size is the same for all the models

        batch_x = np.zeros(
            (self.batch_size, patch_size, patch_size, patch_size, 1),
            dtype=np.float32)
        batch_y = np.zeros(
            (self.batch_size, patch_size, patch_size, patch_size, 1),
            dtype=np.float32)

        for i in range(self.batch_size):
            model_no = np.random.random_integers(0, len(self.dataset.dset) - 1)
            index = np.random.random_integers(
                0, self.dataset.dset[model_no]['x'].shape[0] - 1)

            x = self.dataset.dset[model_no]['x'][index]
            y = self.dataset.dset[model_no]['y'][index]

            # viz.visualize_3d(x)
            # viz.visualize_3d(y)

            batch_y[i, :, :, :, :] = y
            batch_x[i, :, :, :, :] = x

        # make batch B2C01 rather than B012C
        batch_x = batch_x.transpose(0, 3, 4, 1, 2)
        batch_y = batch_y.transpose(0, 3, 4, 1, 2)

        if self.flatten_y:
            batch_y = batch_y.reshape(batch_y.shape[0],
                                      reduce(mul, batch_y.shape[1:]))

        # apply post processors to the patches
        for post_processor in self.iterator_post_processors:
            batch_x, batch_y = post_processor.apply(batch_x, batch_y)

        return batch_x, batch_y

    def batch_size(self):
        return self.batch_size

    def num_batches(self):
        return self.num_batches

    def num_examples(self):
        return self.dataset.get_num_examples()


def main():
    h5_dir = '/srv/data/shape_completion_data/shrec/h5/'
    model_name = ['D00003']
    dataset = ShrecHoldoutDataset(h5_dir, model_name)
    it = dataset.iterator(5)
    it.next()

if __name__ == "__main__":
    main()