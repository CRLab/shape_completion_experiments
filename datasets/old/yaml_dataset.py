import numpy as np
import collections
import yaml
import pcl
from operator import mul
import random

class YamlDataset:
    def __init__(self,
                 yaml_file):
        self._data = yaml.load(open(yaml_file, "r"))

    def train_iterator(self,
                       batch_size,
                       flatten_y=True):
        return DataIterator(self,
                             batch_size=batch_size,
                             data_key="train_models_train_views",
                             flatten_y=flatten_y)

    def holdout_view_iterator(self,
                             batch_size,
                             flatten_y=True):
        return DataIterator(self,
            batch_size=batch_size,
            data_key="train_models_holdout_views",
            flatten_y=flatten_y)

    def holdout_model_iterator(self,
                               batch_size,
                               flatten_y=True):
        return DataIterator(self,
            batch_size=batch_size,
            data_key="holdout_models_holdout_views",
            flatten_y=flatten_y)

    def verify_dataset(self):
        key_pairs = [("holdout_models_holdout_views","train_models_train_views"),
                    ("holdout_models_holdout_views","train_models_holdout_views"),
                    ("train_models_train_views","train_models_holdout_views")]

        for (k1, k2) in key_pairs:
            print "checking: " + str(k1) + " vs " + str(k2)
            for i,(x0, y0) in enumerate(self._data[k1]):
                for (x1, y1) in self._data[k2] :
                    if x0 == x1:
                        print "Failed: " + str(k1) + " vs " + str(k2) + ": " + str(i)
                        print x0
                        assert False

        print "finished: all good"



class DataIterator():
    def __init__(self,
                 dataset,
                 batch_size,
                 data_key,
                 iterator_post_processors=[],
                 flatten_y=True):

        self.dataset = dataset
        self.batch_size = batch_size
        self.flatten_y = flatten_y
        self.data_key = data_key
        self.patch_size = self.dataset._data["patch_size"]
        self.iterator_post_processors = iterator_post_processors


    def next(self):

        batch_x = np.zeros(
            (self.batch_size, self.patch_size, self.patch_size, self.patch_size, 1),
            dtype=np.float32)
        batch_y = np.zeros(
            (self.batch_size, self.patch_size, self.patch_size, self.patch_size, 1),
            dtype=np.float32)

        for i in range(self.batch_size):

            x_filepath, y_filepath  =  random.choice(self.dataset._data[self.data_key])

            x_np_pts = pcl.load(x_filepath).to_array().astype(int)
            x_mask = (x_np_pts[:, 0], x_np_pts[:, 1], x_np_pts[:, 2])

            y_np_pts = pcl.load(y_filepath).to_array().astype(int)
            y_mask = (y_np_pts[:, 0], y_np_pts[:, 1], y_np_pts[:, 2])

            batch_y[i, :, :, :, :][y_mask] = 1
            batch_x[i, :, :, :, :][x_mask] = 1

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
