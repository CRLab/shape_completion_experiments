import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet.conv3d2d import *
from operator import mul
from layers.hidden_layer import *
from layers.conv_layer_3d import *
from layers.layer_utils import *
from layers.recon_layer import *
from layers.flatten_layer import *
from collections import namedtuple
from model_builder import ModelBuilder


class ConvHiddenReconModelConfig():
    def __init__(self,
                 batch_size=3,
                 conv_size=(3, 3),
                 downsample_factor=16,
                 nkerns=(10, 25),
                 nhidden=(1000,),
                 xdim=256 / 2,
                 ydim=256,
                 zdim=256):

        self.batch_size = batch_size
        self.conv_size = conv_size
        self.downsample_factor = downsample_factor
        self.nkerns = nkerns
        self.nhidden = nhidden

        self.input_shape = (
        batch_size, zdim / downsample_factor, 1, xdim / downsample_factor,
        ydim / downsample_factor)
        self.output_shape = (
        self.input_shape[0], reduce(mul, self.input_shape[1:]))

    def build_model(self):

        y = T.matrix('y')  # the labels are presented as 1D vector of
        # [int] labels

        mb = ModelBuilder(self.input_shape)

        # add convolutional layers
        for i in range(len(self.nkerns)):

            nchannels = self.input_shape[2]
            if i > 0:
                nchannels = self.nkerns[i - 1]

            nkerns = self.nkerns[i]
            conv_size = self.conv_size[i]
            filter_shape = (nkerns, conv_size, nchannels, conv_size, conv_size)

            mb.add_conv_layer(filter_shape)

        # add flatten layer to switch to hidden layers
        mb.add_flatten_layer()

        # add hidden layers
        for i in range(len(self.nhidden)):
            nhidden = self.nhidden[i]
            mb.add_hidden_layer(n=nhidden, activation=relu)

        # now add the recon layer
        mb.add_recon_layer(n=self.output_shape[-1], activation=T.nnet.sigmoid)

        return mb.build_model(y)


def main():
    model_config = ConvHiddenReconModelConfig()
    m = model_config.build_model()
    import IPython

    IPython.embed()

if __name__ == "__main__":
    main()
