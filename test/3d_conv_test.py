import unittest
import numpy as np
import theano
from theano.tensor.nnet.conv3d2d import *


class Test3dConv(unittest.TestCase):
    # simple case, filter is a 3,3,3 cube, all zeroes except in the 8 corners
    # it has ones there.  So, when input is all ones, each one of these corners
    # gets hit once, producing an output of 8
    def test_simple(self):

        n_input_channels = 1
        n_input_samples = 2
        input_x_dim = 3
        input_y_dim = 3
        input_z_dim = 3

        input_shape = (
            n_input_samples, input_z_dim, n_input_channels, input_x_dim,
            input_y_dim)

        dtensor5 = theano.tensor.TensorType('float32', (0,) * 5)
        x = dtensor5()

        n_filter_in_channels = 1
        n_filter_out_channels = 1
        filter_x_dim = 3
        filter_y_dim = 3
        filter_z_dim = 3

        filter_shape = (n_filter_out_channels,
                        filter_z_dim,
                        n_filter_in_channels,
                        filter_x_dim,
                        filter_y_dim)

        filter = np.zeros(filter_shape, dtype=np.float32)

        for i in [0, 2]:
            for j in [0, 2]:
                for k in [0, 2]:
                    filter[0, i, 0, j, k] = 1

        conv_out = conv3d(x, filter, signals_shape=input_shape,
                          filters_shape=filter_shape)
        f = theano.function([x], [conv_out])

        out = f(np.ones(input_shape, dtype=np.float32))
        out = out[0]

        self.assertEqual(out.shape, (2, 1, 1, 1, 1))
        self.assertEqual(out[0, 0, 0, 0, 0], 8)
        self.assertEqual(out[1, 0, 0, 0, 0], 8)

    def test_scale_up(self):

        n_input_channels = 1
        n_input_samples = 2
        input_x_dim = 2
        input_y_dim = 2
        input_z_dim = 4

        input_shape = (
            n_input_samples, input_z_dim, n_input_channels, input_x_dim,
            input_y_dim)

        dtensor5 = theano.tensor.TensorType('float32', (0,) * 5)
        x = dtensor5()

        n_filter_in_channels = 1
        n_filter_out_channels = 1
        filter_x_dim = 2
        filter_y_dim = 1
        filter_z_dim = 3

        filter_shape = (n_filter_out_channels,
                        filter_z_dim,
                        n_filter_in_channels,
                        filter_x_dim,
                        filter_y_dim
                        )

        filter = np.zeros(filter_shape, dtype=np.float32)
        #     [b, z, c, x, y]
        filter[0, 0, 0, 1, 0] = 1
        filter[0, 2, 0, 1, 0] = 2

        conv_out = conv3d(x, filter, signals_shape=input_shape,
                          filters_shape=filter_shape)
        f = theano.function([x], [conv_out])

        input = np.zeros(input_shape, dtype=np.float32)

        fill_value = .1
        for z in range(input_z_dim):
            fill_value *= 10
            for x in range(input_x_dim):
                for y in range(input_y_dim):
                    input[0, z, 0, x, y] = fill_value

        out = f(input)
        out = out[0]

        self.assertEqual(out.shape, (2, 2, 1, 1, 2))
        self.assertEqual(out[0, 0, 0, 0, 0], 102)
        self.assertEqual(out[0, 0, 0, 0, 1], 102)
        self.assertEqual(out[0, 1, 0, 0, 0], 1020)
        self.assertEqual(out[0, 1, 0, 0, 1], 1020)


if __name__ == '__main__':
    unittest.main()
