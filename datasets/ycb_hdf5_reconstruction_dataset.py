import numpy as np
import collections
import h5py
from operator import mul


class YcbReconstructionDataset():
    def __init__(self,
                 models_dir,
                 model_names):
        self.dset = []
        self.num_examples = []
        self.patch_size = []
        self.train_set = []
        self.test_set = []
        for i, model_name in enumerate(model_names):
            self.dset.append(h5py.File(models_dir + model_name + '/h5_remesh/' + model_name + '.h5', 'r'))

            self.num_examples.append(self.dset[i]['x'].shape[0])
            self.patch_size.append(self.dset[i]['x'].shape[1])
            np.random.seed = i
            self.train_set.append(np.unique(np.random.random_integers(0, self.num_examples[i] - 1,
                                                                      np.floor(0.9 * self.num_examples[i]))))
            full_set = range(0, self.num_examples[i])
            self.test_set.append(np.setdiff1d(full_set, self.train_set[i]))

    def get_num_examples(self):
        return self.num_examples

    def iterator(self,
                 batch_size=None,
                 num_batches=None,
                 flatten_y=True):
        return YcbReconstructionIterator(self,
                                         batch_size=batch_size,
                                         num_batches=num_batches,
                                         flatten_y=flatten_y)


class YcbReconstructionIterator(collections.Iterator):
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

    def next(self, train):

        patch_size = self.dataset.patch_size[0]  # Since patch_size is the same for all the models

        batch_x = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1), dtype=np.float32)
        batch_y = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1), dtype=np.float32)

        for i in range(self.batch_size):
            model_no = np.random.random_integers(0, len(self.dataset.dset) - 1)
            if train:
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
            batch_y = batch_y.reshape(batch_y.shape[0], reduce(mul, batch_y.shape[1:]))

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
