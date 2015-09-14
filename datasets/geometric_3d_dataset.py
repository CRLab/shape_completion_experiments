import numpy as np
import math


class Geometric3DDataset:

    SPHERE_TYPE = 0
    # a cube with vertices touching the axes
    DIAMOND_TYPE = 1
    # an axis-aligned cube
    CUBE_TYPE = 2

    def __init__(self,
                 patch_size=32,
                 centered=True):

        if patch_size <= 10:
            raise NotImplementedError

        self.num_labels = 3  # used only for classification. TODO: find a more elegant way rather than hard-coding this
        self.patch_size = patch_size
        self.centered = centered

    def iterator(self,
                 batch_size=None,
                 num_batches=None,
                 iter_type="CLASSIFICATION"):

        if iter_type == "CLASSIFICATION":
            iter_class = Geometric3dClassificationIterator
        elif iter_type == "KINECT_COMPLETION":
            iter_class = Geometric3dKinectCompletionIterator
        elif iter_type == "HALF_COMPLETION_TASK":
            iter_class = Geometric3dHalfCompletionIterator
        else:
            raise NotImplementedError

        return iter_class(patch_size=self.patch_size,
                          centered=self.centered,
                          num_labels=self.num_labels,
                          batch_size=batch_size,
                          num_batches=num_batches)


class BaseGeometric3dIterator():
    def __init__(self,
                 patch_size,
                 centered,
                 num_labels,
                 batch_size,
                 num_batches):

        self.patch_size = patch_size
        self.centered = centered
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        return self

    def _generate_solid_figures(self, geometry_types):

        if self.centered:
            (x0, y0, z0) = ((self.patch_size-1)/2,)*3
        else:
            if self.task == Geometric3DDataset.HALF_COMPLETION_TASK:
                x0 = (self.patch_size-1)/2
                # generate 2 numbers in the range [0, self.patch_size-1)
                (y0, z0) = np.random.rand(2) * ((self.patch_size-1)-6) + 3
            else:
                # generate 3 numbers in the range [0, self.patch_size-1)
                (x0, y0, z0) = np.random.rand(3) * ((self.patch_size-1)-6) + 3

        solid_figures = np.zeros((self.batch_size, self.patch_size, 1, self.patch_size, self.patch_size),
                                 dtype=np.bool)

        for i in xrange(self.batch_size):
            # radius is a random number in [3, self.patch_size/2)
            radius = (self.patch_size/2 - 3) * np.random.rand() + 3

            # bounding box values for optimization
            x_min = int(max(math.ceil(x0-radius), 0))
            y_min = int(max(math.ceil(y0-radius), 0))
            z_min = int(max(math.ceil(z0-radius), 0))
            x_max = int(min(math.floor(x0+radius), self.patch_size-1))
            y_max = int(min(math.floor(y0+radius), self.patch_size-1))
            z_max = int(min(math.floor(z0+radius), self.patch_size-1))

            if geometry_types[i] == Geometric3DDataset.SPHERE_TYPE:
                # We only iterate through the bounding box of the sphere to check whether voxels are inside the sphere
                radius_squared = radius**2
                for z in xrange(z_min, z_max+1):
                    for x in xrange(x_min, x_max+1):
                        for y in xrange(y_min, y_max+1):
                            if (x-x0)**2 + (y-y0)**2 + (z-z0)**2 <= radius_squared:
                                # inside the sphere
                                solid_figures[i, z, 0, x, y] = 1
            elif geometry_types[i] == Geometric3DDataset.DIAMOND_TYPE:
                # We only iterate through the bounding box of the diamond to check whether voxels are inside the diamond
                for z in xrange(z_min, z_max+1):
                    for x in xrange(x_min, x_max+1):
                        for y in xrange(y_min, y_max+1):
                            if abs(x-x0) + abs(y-y0) + abs(z-z0) <= radius:
                                # inside the diamond
                                solid_figures[i, z, 0, x, y] = 1
            elif geometry_types[i] == Geometric3DDataset.CUBE_TYPE:
                solid_figures[i, z_min:z_max+1, 0, x_min:x_max+1, y_min:y_max+1] = 1
            else:
                raise NotImplementedError

        return solid_figures

    def _kinect_scan(self, solid_figures):
        """
        Takes a 5-d boolean numpy array representing batches of 3-d data in BZCXY format.
        Returns a 5-d array of the same shape, containing only one "on" z value (the one with the lowest index) per each (x, y) pair.
        """
        kinect_result = np.zeros(solid_figures.shape, dtype=np.bool)
        for i in xrange(self.batch_size):
            for x in xrange(self.patch_size):
                for y in xrange(self.patch_size):
                    for z in xrange(self.patch_size):
                        if solid_figures[i, z, 0, x, y] == 1:
                            kinect_result[i, z, 0, x, y] = 1
                            break
        return kinect_result

    def _one_hot(self, labels):
        one_hot_matrix = np.zeros((self.batch_size, self.num_labels), dtype=np.bool)
        for i, label in enumerate(labels):
            one_hot_matrix[i, label] = 1
        return one_hot_matrix

    def next(self):
        raise NotImplementedError


class Geometric3dClassificationIterator(BaseGeometric3dIterator):

    def next(self):
        # TODO: allow users to specify how they want the classes to be distributed.
        # Currently using same probability for each class
        geometry_types = np.random.randint(0, self.num_labels, self.batch_size)
        labels = self.__one_hot(geometry_types)  #self.__one_hot(geometry_types)
        data = self._generate_solid_figures(geometry_types=geometry_types)
        return data, labels


class Geometric3dKinectCompletionIterator(BaseGeometric3dIterator):

    def next(self):
        labels = self._generate_solid_figures(geometry_types=(Geometric3DDataset.SPHERE_TYPE,) * self.batch_size)
        data = self._kinect_scan(labels)
        return data, labels


class Geometric3dHalfCompletionIterator(BaseGeometric3dIterator):

    def next(self):

        geometry_types = np.random.randint(0, self.num_labels, self.batch_size)
        temp = self._generate_solid_figures(geometry_types=geometry_types)
        # split the volume in halves in the x direction
        data = temp[:, :, :, :int(self.patch_size/2), :]
        labels = temp[:, :, :, int(self.patch_size/2):, :]
        return data, labels