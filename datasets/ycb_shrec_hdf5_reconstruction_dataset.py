import numpy as np
import collections
import h5py
from operator import mul


class YcbShrecReconstructionDataset:
    def __init__(self,
                 ycb_models_dir,
                 ycb_model_names,
                 shrec_models_dir,
                 shrec_model_names):
        self.ycb_dset = []
        self.ycb_num_examples = []
        self.ycb_patch_size = []  # PatchSize must be same for both datasets
        self.ycb_train_set = []
        self.ycb_test_set = []
        for i, model_name in enumerate(ycb_model_names):
            self.ycb_dset.append(h5py.File(
                ycb_models_dir + model_name + '/h5_remesh/' + model_name + '.h5',
                'r'))

            self.ycb_num_examples.append(self.ycb_dset[i]['x'].shape[0])
            self.ycb_patch_size.append(self.ycb_dset[i]['x'].shape[1])
            np.random.seed(i)
            self.ycb_train_set.append(np.random.choice(
                range(self.ycb_num_examples[i]),
                int(np.floor(0.9 * self.ycb_num_examples[i])), replace=False))
            full_set = range(self.ycb_num_examples[i])
            self.ycb_test_set.append(np.setdiff1d(full_set, self.ycb_train_set[i]))

        self.shrec_dset = []
        self.shrec_num_examples = []
        self.shrec_patch_size = []  # PatchSize must be same for both datasets
        self.shrec_train_set = []
        self.shrec_test_set = []
        for i, model_name in enumerate(shrec_model_names):
            self.shrec_dset.append(h5py.File(
                shrec_models_dir + model_name + '/' + model_name + '.h5', 'r'))
            self.shrec_num_examples.append(self.shrec_dset[i]['x'].shape[0])
            self.shrec_patch_size.append(self.shrec_dset[i]['x'].shape[1])
            #np.random.seed = i # Doesn't seem to have an effect on the choice function
            self.shrec_train_set.append(np.random.choice(
                range(self.shrec_num_examples[i]),
                int(np.floor(0.9 * self.shrec_num_examples[i])), replace=False))
            full_set = range(self.shrec_num_examples[i])
            self.shrec_test_set.append(np.setdiff1d(full_set, self.shrec_train_set[i]))

        assert self.ycb_patch_size[0] == self.shrec_patch_size[0]

    def iterator(self,
                 batch_size=None,
                 num_batches=None,
                 flatten_y=True):
        return YcbShrecReconstructionIterator(self,
                                         batch_size=batch_size,
                                         num_batches=num_batches,
                                         flatten_y=flatten_y)


class YcbShrecReconstructionIterator(collections.Iterator):
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

        patch_size = self.dataset.ycb_patch_size[
            0]  # Since patch_size is the same for all the models
                #  & both datasets

        batch_x = np.zeros(
            (self.batch_size, patch_size, patch_size, patch_size, 1),
            dtype=np.float32)
        batch_y = np.zeros(
            (self.batch_size, patch_size, patch_size, patch_size, 1),
            dtype=np.float32)

        num_ycb_models = len(self.dataset.ycb_dset)
        num_shrec_models = len(self.dataset.shrec_dset)
        p = [float(num_ycb_models)/(num_ycb_models + num_shrec_models),
             float(num_shrec_models)/(num_ycb_models + num_shrec_models)]
        dataset_no = np.random.choice([0, 1], size=self.batch_size, p=p)

        for i in range(self.batch_size):
            if dataset_no[i] == 0:
                model_no = np.random.random_integers(0, num_ycb_models - 1)
                if train:
                    index = np.random.choice(self.dataset.ycb_train_set[model_no])
                else:
                    index = np.random.choice(self.dataset.ycb_test_set[model_no])

                x = self.dataset.ycb_dset[model_no]['x'][index]
                y = self.dataset.ycb_dset[model_no]['y'][index]
            else:
                model_no = np.random.random_integers(0, num_shrec_models - 1)
                if train:
                    index = np.random.choice(self.dataset.shrec_train_set[model_no])
                else:
                    index = np.random.choice(self.dataset.shrec_test_set[model_no])

                x = self.dataset.shrec_dset[model_no]['x'][index]
                y = self.dataset.shrec_dset[model_no]['y'][index]

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

    def batch_size(self):
        return self.batch_size

    def num_batches(self):
        return self.num_batches

    def num_examples(self):
        return self.dataset.get_num_examples()
