import numpy as np
import collections
import h5py
from operator import mul


class AsymetricReconstructionDataset:
    def __init__(self,
                 models_dir,
                 training_model_names,
                 holdout_model_names,
                 mesh_folder='/h5_40_v0/'):

        self.training_dset = []
        self.holdout_model_dset = []

        self.num_training_examples = []
        self.num_holdout_model_examples = []

        self.training_patch_size = []
        self.holdout_model_patch_size = []

        self.train_set = []
        self.holdout_view_set = []
        self.holdout_model_set = []

        print "Training and Holdout View Models:"
        for i, model_name in enumerate(training_model_names):
            filename = models_dir + model_name + mesh_folder + model_name + '.h5'
            print filename
            self.training_dset.append(h5py.File(filename, 'r'))

            self.num_training_examples.append(self.training_dset[i]['x'].shape[0])
            self.training_patch_size.append(self.training_dset[i]['x'].shape[1])
            np.random.seed(i)
            self.train_set.append(np.random.choice(
                range(self.num_training_examples[i]),
                int(np.floor(0.9 * self.num_training_examples[i])), replace=False))
            full_set = range(self.num_training_examples[i])
            self.holdout_view_set.append(np.setdiff1d(full_set, self.train_set[i]))

        print "Holdout Models:"
        for i, model_name in enumerate(holdout_model_names):
            filename = models_dir + model_name + mesh_folder + model_name + '.h5'
            print filename
            self.holdout_model_dset.append(h5py.File(filename, 'r'))

            self.num_holdout_model_examples.append(self.holdout_model_dset[i]['x'].shape[0])
            self.holdout_model_patch_size.append(self.holdout_model_dset[i]['x'].shape[1])
            self.holdout_model_set.append(range(self.num_holdout_model_examples[i]))

    def train_iterator(self,
                       batch_size,
                       flatten_y=True):
        return TrainIterator(self,
                             batch_size=batch_size,
                             flatten_y=flatten_y)

    def holdout_view_iterator(self,
                             batch_size,
                             flatten_y=True):
        return HoldoutViewIterator(self,
                                   batch_size=batch_size,
                                   flatten_y=flatten_y)

    def holdout_model_iterator(self,
                               batch_size,
                               flatten_y=True):
        return HoldoutModelIterator(self,
                                    batch_size=batch_size,
                                    flatten_y=flatten_y)


class DataIterator():
    def __init__(self,
                 dataset,
                 batch_size,
                 iterator_post_processors=[],
                 flatten_y=True):

        self.dataset = dataset
        self.batch_size = batch_size
        self.flatten_y = flatten_y

        self.iterator_post_processors = iterator_post_processors

    def get_dset(self):
        raise NotImplementedError

    def get_next_model_idx(self):
        raise NotImplementedError

    def get_next_view_idx(self, model_idx):
        raise NotImplementedError

    def next(self):

        # Since patch_size is the same for all the models
        patch_size = self.dataset.training_patch_size[0]

        batch_x = np.zeros(
            (self.batch_size, patch_size, patch_size, patch_size, 1),
            dtype=np.float32)
        batch_y = np.zeros(
            (self.batch_size, patch_size, patch_size, patch_size, 1),
            dtype=np.float32)

        for i in range(self.batch_size):
            dset = self.get_dset()
            model_idx = self.get_next_model_idx()
            view_idx = self.get_next_view_idx(model_idx)

            x = dset[model_idx]['x'][view_idx]
            y = dset[model_idx]['y'][view_idx]

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


class TrainIterator(DataIterator):

    def get_dset(self):
        return self.dataset.training_dset

    def get_next_model_idx(self):
        return np.random.random_integers(0, len(self.dataset.training_dset) - 1)

    def get_next_view_idx(self, model_idx):
        return np.random.choice(self.dataset.train_set[model_idx])


class HoldoutViewIterator(DataIterator):

    def get_dset(self):
        return self.dataset.training_dset

    def get_next_model_idx(self):
        return np.random.random_integers(0, len(self.dataset.training_dset) - 1)

    def get_next_view_idx(self, model_idx):
        return np.random.choice(self.dataset.holdout_view_set[model_idx])


class HoldoutModelIterator(DataIterator):

    def get_dset(self):
        return self.dataset.holdout_model_dset

    def get_next_model_idx(self):
        return np.random.random_integers(0, len(self.dataset.holdout_model_dset) - 1)

    def get_next_view_idx(self, model_idx):
        return np.random.choice(self.dataset.holdout_model_set[model_idx])

if __name__ == "__main__":
    models_dir = "/srv/data/shape_completion_data/asymetric/"
    training_model_names = ['m87']
    holdout_model_names = ['M000074']
    dset = AsymetricReconstructionDataset(models_dir,training_model_names=training_model_names,holdout_model_names=holdout_model_names)
    import IPython
    IPython.embed()
