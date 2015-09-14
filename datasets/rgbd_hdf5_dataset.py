

from operator import mul
import h5py
import random
import numpy as np

import pylearn2.datasets.dataset
import pylearn2.utils.rng
from pylearn2.utils.iteration import SubsetIterator, resolve_iterator_class
from pylearn2.utils import safe_izip, wraps



class GaussianNoisePostProcessor():

    def __init__(self, sigma=.2, mu=0, prob_noise_added=.2):
        self.sigma = sigma
        self.mu = mu
        self.prob_noise_added = prob_noise_added

    def apply(self, batch_x, batch_y):

        #we only add noise to the locations where the mask is True
        mask = np.random.rand(*batch_x.shape) < self.prob_noise_added

        #this noise is centered around mu with standard dev sigma
        noise = self.sigma * np.random.randn(*batch_x.shape) + self.mu

        #apply noise where mask is true, zero otherwise
        masked_noise = np.where(mask, noise, 0)

        return batch_x + masked_noise, batch_y


#this dataset has rgbd images, and returns patches centered around
#finger locations within the images
#the hdf5 dataset has the following keys:

#rgbd - shape:(num_images, 640, 480, 4)
#uvd - shape:(num_images, num_finger_types, 3) , 3 is for u,v,d

class RGBD_HDF5_Dataset(pylearn2.datasets.dataset.Dataset):

    def __init__(self, topo_view_key, y_key, hdf5_filepath, patch_size=72):
        self.h5py_dataset = h5py.File(hdf5_filepath, 'r')

        #our topological view is rgbd
        self.topo_view = self.h5py_dataset[topo_view_key]
        self.y = self.h5py_dataset[y_key]

        self.patch_size = patch_size

        if not 'num_grasp_type' in self.h5py_dataset.keys():
            self.h5py_dataset.create_dataset('num_grasp_type', (1,))
            self.h5py_dataset['num_grasp_type'][0] = self.h5py_dataset['grasp_type'][:].max() + 1

        if not 'num_finger_type' in self.h5py_dataset.keys():
            self.h5py_dataset.create_dataset('num_finger_type', (1,))
            self.h5py_dataset['num_finger_type'][0] = self.h5py_dataset['uvd'].shape[1]

    def adjust_for_viewer(self, X):
        return X[:, :, :, 0:3]

    # def get_batch_design(self, batch_size, include_labels=False):
    #
    #     if include_labels:
    #         raise NotImplementedError
    #
    #     topo_batch = self.get_batch_topo(batch_size)
    #     return topo_batch.reshape(topo_batch.shape[-1], reduce(mul, topo_batch.shape[:-1]))

    #def get_batch_topo(self, batch_size):
    #    range_start = 0
    #    range_end = self.topo_view.shape[-1]-batch_size
    #
    #    batch_start = random.randint(range_start, range_end)
    #    batch_end = batch_start + batch_size
    #
    #    return self.topo_view[:, :, :, batch_start:batch_end]
    def get_num_examples(self):
        num_rgbd_images = self.topo_view.shape[0]
        num_finger_types = self.h5py_dataset['uvd'].shape[1]
        return num_rgbd_images * num_finger_types

    def get_topo_batch_axis(self):
        return -1

    def has_targets(self):
        return True

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        return HDF5_Iterator(self,
                             batch_size=batch_size,
                             num_batches=num_batches,
                             mode=mode)


class HDF5_Iterator():

    def __init__(self, dataset,
                 batch_size,
                 num_batches,
                 mode,
                 iterator_post_processors=(GaussianNoisePostProcessor(.01, 0, .5),
                                           GaussianNoisePostProcessor(.1, 0, .001))):

        def _validate_batch_size(batch_size, dataset):
            if not batch_size:
                raise ValueError("batch size is none")

            num_examples = dataset.get_num_examples()
            if batch_size > num_examples:
                raise ValueError("batch size:%i is to large, dataset has %i examples", batch_size, num_examples)

            if batch_size < 0:
                raise ValueError("batch size: %i cannot be negative", batch_size)

            if not isinstance(batch_size, int):
                raise ValueError("batch_size is not an int")

        def _validate_num_batches(num_batches):
            if not num_batches:
                raise ValueError("num_batches is none")

            if num_batches < 0:
                raise ValueError("num_batches: %i cannot be negative", num_batches)

            if not isinstance(num_batches, int):
                raise ValueError("num_batches is not an int")

        self.dataset = dataset
        dataset_size = dataset.get_num_examples()

        _validate_batch_size(batch_size, dataset)
        _validate_num_batches(num_batches)

        subset_iterator_class = resolve_iterator_class(mode)
        self._subset_iterator = subset_iterator_class(dataset_size, batch_size, num_batches)

        self.iterator_post_processors = iterator_post_processors

    def __iter__(self):
        return self

    def next(self):

        #batch_indices = self._subset_iterator.next()
        batch_indices = np.random.random_integers(0, self.dataset.get_num_examples()-1, self.batch_size)

        # if isinstance(batch_indices, slice):
        #     batch_indices = np.array(range(batch_indices.start, batch_indices.stop))

        # if we are using a shuffled sequential subset iterator
        # then next_index will be something like:
        # array([13713, 14644, 30532, 32127, 35746, 44163, 48490, 49363, 52141, 52216])
        # hdf5 can only support this sort of indexing if the array elements are
        # in increasing order
        # batch_size = 0
        # if isinstance(batch_indices, np.ndarray):
        #     batch_indices.sort()
        batch_size = self.batch_size

        num_uvd_per_rgbd = self.dataset.h5py_dataset['uvd'].shape[1]
        num_grasp_types = self.dataset.h5py_dataset['num_grasp_type'][0]

        finger_indices = batch_indices % num_uvd_per_rgbd
        batch_indices = np.floor(batch_indices / num_uvd_per_rgbd)

        patch_size = self.dataset.patch_size

        batch_x = np.zeros((batch_size, patch_size, patch_size, 4))
        batch_y = np.zeros((batch_size, num_uvd_per_rgbd * num_grasp_types))

        #go through and append patches to batch_x, batch_y
        for i in range(len(finger_indices)):
            batch_index = batch_indices[i]
            finger_index = finger_indices[i]
            u, v, d = self.dataset.h5py_dataset['uvd'][batch_index, finger_index, :]
            rgbd = self.dataset.topo_view[batch_index, :,:,:]
            grasp_type = self.dataset.y[batch_index, 0]
            grasp_energy = self.dataset.h5py_dataset['energy'][batch_index]

            patch = rgbd[u-patch_size/2.0: u+patch_size/2.0, v-patch_size/2.0:v+patch_size/2.0, :]
            patch_label = num_uvd_per_rgbd * grasp_type + finger_index

            batch_x[i, :, :, :] = patch
            batch_y[i, patch_label] = grasp_energy

        #make batch C01B rather than B01C
        batch_x = batch_x.transpose(3, 1, 2, 0)

        #apply post processors to the patches
        for post_processor in self.iterator_post_processors:
            batch_x, batch_y = post_processor.apply(batch_x, batch_y)

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        return batch_x, batch_y

    @property
    @wraps(SubsetIterator.batch_size, assigned=(), updated=())
    def batch_size(self):
        return self._subset_iterator.batch_size

    @property
    @wraps(SubsetIterator.num_batches, assigned=(), updated=())
    def num_batches(self):
        return self._subset_iterator.num_batches

    @property
    @wraps(SubsetIterator.num_examples, assigned=(), updated=())
    def num_examples(self):
        return self._subset_iterator.num_examples

    @property
    @wraps(SubsetIterator.uneven, assigned=(), updated=())
    def uneven(self):
        return self._subset_iterator.uneven

    @property
    @wraps(SubsetIterator.stochastic, assigned=(), updated=())
    def stochastic(self):
        return self._subset_iterator.stochastic



