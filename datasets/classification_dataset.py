import numpy as np
import collections
import yaml
import pcl
from operator import mul
import random

from pool import ThreadPool

class Dataset:
    def __init__(self,
                 data_directory):
        self.data_directory = data_directory
        self._info = yaml.load(open(self.data_directory + "/info.yaml", "r"))

    def train_iterator(self,
                       batch_size):
        return DataIterator(self,
                             batch_size=batch_size,
                             labels_key="train_model_names",
                             data_key="train_models_train_views")

    def holdout_view_iterator(self,
                             batch_size):
        return DataIterator(self,
            batch_size=batch_size,
            labels_key="train_model_names",
            data_key="train_models_holdout_views")

    def holdout_model_iterator(self,
                               batch_size):
        return DataIterator(self,
            batch_size=batch_size,
            labels_key="holdout_model_names",
            data_key="holdout_models_holdout_views")

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
                 labels_key,
                 data_key,
                 iterator_post_processors=[]):

        self.dataset = dataset

        self.data_key = data_key
        self.labels_key = labels_key

        self.labels_file = self.dataset.data_directory + "/" + self.labels_key + ".txt"
        self.datafile = self.dataset.data_directory + "/" + self.data_key + ".txt"

        self.batch_size = batch_size
        self.patch_size = self.dataset._info["patch_size"]
        self.iterator_post_processors = iterator_post_processors

        label_lines  = open(self.labels_file).readlines()

        #labels to ids example: rubbermaid:0
        self.labels2ids = {}
        for label_line in label_lines:
            label_id, label_name = label_line.replace("\n","").split(", ")
            self.labels2ids[label_name] = int(label_id)

        self.num_labels = len(self.labels2ids.keys())

        data_lines = open(self.datafile).readlines()

        self.data = [None]*len(data_lines)

        for i, data_line in enumerate(data_lines):
            x_filepath, y_filepath = data_line.replace("\n", "").split(", ")
            y_label = y_filepath.split("/")[-3]

            #ex [ (full_pcd_path_forx.pcd, 0), ...]
            self.data[i] =(x_filepath, self.labels2ids[y_label])

        #LEAVE this at 6, it works better than 4,8,16,32
        #self.pool = ThreadPool(6)


    def add_example_to_batch(self, batch_x, batch_y, i):
        x_filepath, y_label = random.choice(self.data)

        batch_y[i, y_label] = 1

        x_np_pts = pcl.load(x_filepath).to_array().astype(int)
        x_mask = (x_np_pts[:, 0], x_np_pts[:, 1], x_np_pts[:, 2])
        batch_x[i, :, :, :, :][x_mask] = 1

    def next(self):

        batch_x = np.zeros(
            (self.batch_size, self.patch_size, self.patch_size, self.patch_size, 1),
            dtype=np.float32)
        batch_y = np.zeros(
            (self.batch_size, self.num_labels),
            dtype=np.float32)

        #for i in range(self.batch_size):
        #    self.pool.add_task(self.add_example_to_batch, batch_x, batch_y, i)

        #self.pool.wait_completion()

        # single threaded, much slower than using the pool, tested using timeit
        for i in range(self.batch_size):
            self.add_example_to_batch(batch_x, batch_y, i)

        # make batch B2C01 rather than B012C
        batch_x = batch_x.transpose(0, 3, 4, 1, 2)

        # apply post processors to the patches
        for post_processor in self.iterator_post_processors:
            batch_x, batch_y = post_processor.apply(batch_x, batch_y)

        return batch_x, batch_y



if __name__ == "__main__":
    dset = Dataset("YCB_Dataset")
    train_iterator = dset.train_iterator(40)
    x_batch, y_batch = train_iterator.next()
    import IPython
    IPython.embed()
