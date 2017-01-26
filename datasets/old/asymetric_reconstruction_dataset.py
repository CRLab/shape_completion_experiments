import numpy as np
import os
import collections
import binvox_rw
import visualization.visualize as viz
import tf_conversions
import PyKDL
import math
#from utils.reconstruction_utils import create_voxel_grid_around_point
from utils.reconstruction_utils import map_pointclouds_to_camera_frame, \
     build_training_example_scaled
from operator import mul


class AsymetricDataset():
    def __init__(self,
                 data_dir,
                 model_name,
                 patch_size):

        self.models_dir = data_dir + '/' + model_name + '/'
        self.pc_dir = data_dir + '/' + model_name + '/pointclouds/'
        self.model_name = model_name
        self.patch_size = patch_size
        self.model_fullfilename = self.models_dir + model_name + '_scaled.binvox'
        filenames = [d for d in os.listdir(self.pc_dir) if
                     not os.path.isdir(self.pc_dir + d)]

        self.examples = []
        for file_name in filenames:
            if "_pc.npy" in file_name:
                pointcloud_file = self.pc_dir + file_name
                pose_file = self.pc_dir + file_name.replace('pc', 'model_pose')
                self.examples.append(
                    (pointcloud_file, pose_file, self.model_fullfilename))

    def get_num_examples(self):
        return len(self.examples)

    def iterator(self,
                 batch_size=None,
                 num_batches=None,
                 flatten_y=True):
        return AsymetricDatasetIterator(self,
                                  batch_size=batch_size,
                                  num_batches=num_batches,
                                  flatten_y=flatten_y)


class AsymetricDatasetIterator(collections.Iterator):
    def __init__(self,
                 dataset,
                 batch_size,
                 num_batches,
                 flatten_y,
                 iterator_post_processors=()):

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.flatten_y = flatten_y
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
        batch_y = np.zeros(
            (self.batch_size, patch_size, patch_size, patch_size, 1),
            dtype=np.float32)

        for i in range(len(batch_indices)):
            index = batch_indices[i]
            model_filepath = self.dataset.model_fullfilename
            single_view_pointcloud_filepath = self.dataset.examples[index][0]
            pose_filepath = self.dataset.examples[index][1]

            # x, y = build_training_example(model_filepath, pose_filepath,
            #                               single_view_pointcloud_filepath,
            #                               patch_size)
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


def main():
    dataset = AsymetricDataset("/srv/data/shape_completion_data/asymetric",
                         "m87", 40)
    it = dataset.iterator(5,flatten_y = False)
    x,y = it.next()
    import IPython
    IPython.embed()


if __name__ == "__main__":
    main()

