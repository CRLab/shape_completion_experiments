import numpy as np
import collections
import yaml
import pcl
from operator import mul
import random

from pool import *

class Dataset:
    def __init__(self,
                 data_directory):
        self.data_directory = data_directory
        self._info = yaml.load(open(self.data_directory + "/info.yaml", "r"))

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

    def verify(self):
        print("this check verifies that all the iterators")
        print("are sampling from text files with 0 overlap.")

        key_pairs = [("holdout_models_holdout_views","train_models_train_views"),
                    ("holdout_models_holdout_views","train_models_holdout_views"),
                    ("train_models_train_views","train_models_holdout_views")]

        for (k0, k1) in key_pairs:
            print "checking: " + str(k0) + " vs " + str(k1)
            k0_datafile = self.data_directory + "/" + k0 + ".txt"
            k0_lines = open(k0_datafile).readlines()

            k1_datafile = self.data_directory + "/" + k1 + ".txt"
            k1_lines = open(k1_datafile).readlines()

            for i,k0_line in enumerate(k0_lines):
                x0, y0  =  k0_line.replace("\n", "").split(", ")
                for k1_line in k1_lines :
                    x1, y1  =  k1_line.replace("\n", "").split(", ")
                    if x0 == x1:
                        print "Failed: " + str(k0) + " vs " + str(k1) + ": " + str(i)
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
        self.data_key = data_key
        self.datafile = self.dataset.data_directory + "/" + self.data_key + ".txt"

        self.batch_size = batch_size
        self.flatten_y = flatten_y
        self.patch_size = self.dataset._info["patch_size"]
        self.iterator_post_processors = iterator_post_processors

        self.lines = open(self.datafile).readlines()

        #LEAVE this at 6, it works better than 4,8,16,32
        self.pool = ThreadPool(6)


    def add_example_to_batch(self, batch_x, batch_y, i):
        random_line = random.choice(self.lines)
        x_filepath, y_filepath  =  random_line.replace("\n", "").split(", ")

        x_np_pts = pcl.load(x_filepath).to_array().astype(int)
        x_mask = (x_np_pts[:, 0], x_np_pts[:, 1], x_np_pts[:, 2])

        y_np_pts = pcl.load(y_filepath).to_array().astype(int)
        y_mask = (y_np_pts[:, 0], y_np_pts[:, 1], y_np_pts[:, 2])

        batch_y[i, :, :, :, :][y_mask] = 1
        batch_x[i, :, :, :, :][x_mask] = 1

    def next(self):

        batch_x = np.zeros(
            (self.batch_size, self.patch_size, self.patch_size, self.patch_size, 1),
            dtype=np.float32)
        batch_y = np.zeros(
            (self.batch_size, self.patch_size, self.patch_size, self.patch_size, 1),
            dtype=np.float32)

        for i in range(self.batch_size):
            self.pool.add_task(self.add_example_to_batch, batch_x, batch_y, i)

        self.pool.wait_completion()

        # single threaded, much slower than using the pool, tested using timeit
        #for i in range(self.batch_size):
        #    self.add_example_to_batch(batch_x, batch_y, i)

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
