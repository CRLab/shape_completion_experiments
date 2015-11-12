import collections
import numpy as np
from operator import mul
from utils.reconstruction_utils import create_voxel_grid_around_point
from utils.reconstruction_utils import map_pointclouds_to_camera_frame, \
    build_training_example, build_training_example_scaled


class ReconstructionIterator(collections.Iterator):
    def __init__(self,
                 dataset,
                 batch_size,
                 num_batches,
                 flatten_y,
                 is_training,
                 iterator_post_processors=[]):

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.flatten_y = flatten_y
        self.iterator_post_processors = iterator_post_processors
        self.is_training = is_training

    def __iter__(self):
        return self

    def next(self):

        batch_indices = np.random.random_integers(0,
                                                  self.dataset.get_num_examples() - 1,
                                                  self.batch_size)
        patch_size = self.dataset.patch_size
        batch_x = np.zeros(
            (self.batch_size, patch_size, patch_size, patch_size, 1),
            dtype=np.float32)
        batch_y = np.zeros(
            (self.batch_size, patch_size, patch_size, patch_size, 1),
            dtype=np.float32)

        for i in range(len(batch_indices)):
            index = batch_indices[i]
            model_filepath = self.dataset.model_fullfilename
            single_view_pointcloud_filepath = self.dataset.examples[index][0]
            pose_filepath = self.dataset.examples[index][1]

            # x, y = build_training_example(model_filepath, pose_filepath,
            # single_view_pointcloud_filepath, patch_size)
            x, y = \
                build_training_example_scaled(model_filepath, pose_filepath,
                                              single_view_pointcloud_filepath,
                                              patch_size)

            ############################

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


class HDF5ReconstructionIterator(ReconstructionIterator):
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
            if self.dataset.is_training:
                index = np.random.choice(self.dataset.train_set[model_no])
            else:
                index = np.random.choice(self.dataset.test_set[model_no])

            x = self.dataset.dset[model_no]['x'][index]
            y = self.dataset.dset[model_no]['y'][index]

            # viz.visualize_3d(x)
            # viz.visualize_3d(y)
            # viz.visualize_pointcloud(pc2_out[0:3, :].T)

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
