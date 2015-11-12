import numpy as np
import os
import collections
from operator import mul
import visualization.visualize as viz
from utils.reconstruction_utils import build_test_example_scaled


class TestDataset():
    def __init__(self,
                 data_dir,
                 model_name,
                 patch_size):

        self.pc_dir = data_dir  # + '/' + model_name + '/pointclouds/'
        # self.model_name = model_name
        self.patch_size = patch_size
        filenames = [d for d in os.listdir(self.pc_dir) if
                     not os.path.isdir(self.pc_dir + d)]

        self.examples = []
        for file_name in filenames:
            if ".pcd" in file_name:
                pointcloud_file = self.pc_dir + file_name
                self.examples.append(pointcloud_file)

    def get_num_examples(self):
        return len(self.examples)

    def iterator(self,
                 batch_size=None,
                 num_batches=None):
        return TestDatasetIterator(self,
                                   batch_size=batch_size,
                                   num_batches=num_batches)


class TestDatasetIterator(collections.Iterator):
    def __init__(self,
                 dataset,
                 batch_size,
                 num_batches,
                 iterator_post_processors=[]):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.iterator_post_processors = iterator_post_processors

    def __iter__(self):
        return self

    def next(self):
        batch_indices = \
            np.random.random_integers(0,
                                      self.dataset.get_num_examples() - 1,
                                      self.batch_size)
        patch_size = self.dataset.patch_size
        batch_x = np.zeros(
            (self.batch_size, patch_size, patch_size, patch_size, 1),
            dtype=np.float32)

        for i in range(len(batch_indices)):
            index = batch_indices[i]
            single_view_pointcloud_filepath = self.dataset.examples[index]

            x = build_test_example_scaled(single_view_pointcloud_filepath,
                                          patch_size)

            ############################

            batch_x[i, :, :, :, :] = x

        # make batch B2C01 rather than B012C
        batch_x = batch_x.transpose(0, 3, 4, 1, 2)

        # apply post processors to the patches
        # for post_processor in self.iterator_post_processors:
        #    batch_x, batch_y = post_processor.apply(batch_x, batch_y)

        return batch_x

    def batch_size(self):
        return self.batch_size

    def num_batches(self):
        return self.num_batches

    def num_examples(self):
        return self.dataset.get_num_examples()


if __name__ == "__main__":
    dataset = TestDataset("/srv/data/shape_completion_data/test/", "", 30)
    it = dataset.iterator(5)
    it.next()
