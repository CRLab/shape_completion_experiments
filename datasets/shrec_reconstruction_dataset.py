import numpy as np
import os
import collections
import visualization.visualize as viz
from utils.reconstruction_utils import build_training_example


class ReconstructionDataset():
    def __init__(self,
<<<<<<< HEAD
                 models_dir,
                 pc_dir,
                 patch_size,
                 num_models):
=======
                 models_dir="/srv/data/downloaded_mesh_models/shrec/models/",
                 pc_dir="/srv/data/shape_completion_data/shrec/" +
                        "gazebo_reconstruction_data_uniform_rotations_shrec_" +
                        "centered_scaled/",
                 patch_size=72,
                 num_models=None):
>>>>>>> bceed328a330f3ed33460702702cc8339ddfdb2b

        self.models_dir = models_dir
        self.pc_dir = pc_dir
        self.model_names = os.listdir(models_dir)
        if num_models is not None:
            self.model_names = self.model_names[:num_models]

        self.patch_size = patch_size

        filenames = []
        for model_name in self.model_names:
            if os.path.exists(pc_dir + model_name):
                model_files = [pc_dir + model_name + "/" + d for d in
                               os.listdir(pc_dir + model_name) if
                               not os.path.isdir(
                                   os.path.join(pc_dir + model_name, d))]
                filenames.append((model_name, model_files))

        self.examples = []
        for item in filenames:
            model_name, file_names = item
            for file_name in file_names:

                if "_pc.npy" in file_name:
                    pointcloud_file = file_name
                    pose_file = file_name.replace("pc", "pose")
                    binvox_model_file = models_dir + model_name + "/" + \
                                        model_name + ".binvox"
                    scale_file = models_dir + model_name + "/" + \
                                 "offset_and_scale.txt"

                    self.examples.append((pointcloud_file, pose_file,
                                          binvox_model_file, scale_file))

    def get_num_examples(self):
        return len(self.examples)

    def iterator(self,
                 batch_size=10,
                 num_batches=10):

        return ReconstructionIterator(self,
                                      batch_size=batch_size,
                                      num_batches=num_batches)


class ReconstructionIterator(collections.Iterator):
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
        batch_y = np.zeros(
            (self.batch_size, patch_size, patch_size, patch_size, 1),
            dtype=np.float32)

        for i in range(len(batch_indices)):
            index = batch_indices[i]

            single_view_pointcloud_filepath = self.dataset.examples[index][0]
            pose_filepath = self.dataset.examples[index][1]
            model_filepath = self.dataset.examples[index][2]
            scale_filepath = self.dataset.examples[index][3]
            print scale_filepath

            f = open(scale_filepath)
            line_0 = f.readline()
            offset_x, offset_y, offset_z, scale = line_0.split()

            custom_scale = float(scale)
            custom_offset = (float(offset_x), float(offset_y), float(offset_z))

            # print model_filepath
            # print pose_filepath
            # print single_view_pointcloud_filepath
            x, y = build_training_example(model_filepath,
                                          pose_filepath,
                                          single_view_pointcloud_filepath,
                                          patch_size,
                                          custom_scale=custom_scale,
                                          custom_offset=custom_offset)

            viz.visualize_3d(x)
            viz.visualize_3d(y)

            batch_y[i, :, :, :, :] = y
            batch_x[i, :, :, :, :] = x

        # make batch B2C01 rather than B012C
        batch_x = batch_x.transpose(0, 3, 4, 1, 2)
        batch_y = batch_y.transpose(0, 3, 4, 1, 2)

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
