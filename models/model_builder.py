


import numpy

import theano
import theano.tensor as T

from theano.tensor.nnet.conv3d2d import *

from layers.conv_layer_3d import *
from layers.flatten_layer import *
from layers.hidden_layer import *
from layers.layer_utils import *
from layers.logistic_regression_layer import *
from layers.max_pool_layer_3d import *
from layers.recon_layer import *

from collections import namedtuple

Model = namedtuple('Model', ['train', 'test', 'validate', 'demonstrate', 'layers'], verbose=False)


class ModelBuilder():

    def __init__(self, input_shape, learning_rate=.1):
        self.learning_rate = learning_rate

        self.layers = []

        self.rng = numpy.random.RandomState(23455)
        self.drop = T.iscalar('drop')
        self.input_shape = input_shape

    def add_conv_layer(self, filter_shape):

        if len(self.layers) == 0:
            dtensor5 = theano.tensor.TensorType('float32', (0,)*5)
            input = dtensor5()
            input_shape = self.input_shape
        else:
            input = self.layers[-1].output
            input_shape = self.layers[-1].output_shape

        layer = ConvLayer3D(
            rng=self.rng,
            input=input,
            image_shape=input_shape,
            filter_shape=filter_shape,
            poolsize=(0, 0),
            drop=self.drop
        )

        self.layers.append(layer)

    def add_max_pool_layer(self, downsample_factor=2, ignore_border=False):
        layer = MaxPoolLayer3D(
            input=self.layers[-1].output,
            image_shape=self.layers[-1].output_shape,
            ds=downsample_factor,
            ignore_border=ignore_border
        )

        self.layers.append(layer)

    def add_flatten_layer(self):
        layer = FlattenLayer(
            input=self.layers[-1].output,
            input_shape=self.layers[-1].output_shape,
        )
        self.layers.append(layer)

    def add_hidden_layer(self, n, activation):

        layer = HiddenLayer(
            self.rng,
            input=self.layers[-1].output,
            input_shape=self.layers[-1].output_shape,
            n_out=n,
            activation=activation,
            drop=self.drop
        )

        self.layers.append(layer)

    def add_recon_layer(self, n, activation):
        layer = ReconLayer(
            self.rng,
            input=self.layers[-1].output,
            n_in=self.layers[-1].output_shape[-1],
            n_out=n,
            activation=activation,
        )

        self.layers.append(layer)

    def add_logistic_regression_layer(self, n):
        layer = LogisticRegression(
            input=self.layers[-1].output,
            n_in=self.layers[-1].output_shape[-1],
            n_out=n,rng=self.rng


        )

        self.layers.append(layer)

    def build_model(self, y):

        params = []
        for layer in self.layers:
            params += layer.params

        x = self.layers[0].input

        # the cost we minimize during training is the NLL of the model
        cost = self.layers[-1].negative_log_likelihood(y)
        errors = self.layers[-1].errors(y)

        # create a function to compute the mistakes that are made by the model
        test_model = theano.function(
            [x, y],
            errors,
            givens={
                self.drop: numpy.cast['int32'](0)
            }, allow_input_downcast=True
        )

        validate_model = theano.function(
            [x,y],
            errors,
            givens={

                self.drop: numpy.cast['int32'](0)

            }, allow_input_downcast=True

        )

        demonstrate_model = None
        """theano.function(
            [x, y],
            self.layers[-1].return_output(),
            givens={self.drop: numpy.cast['int32'](0)}, on_unused_input='ignore'
        )"""

        # create a list of gradients for all model parameters
        grads = T.grad(cost, params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.

        #RMSprop
        updates = []
        for p, g in zip(params, grads):
            MeanSquare = theano.shared(p.get_value() * 0.)
            nextMeanSquare = 0.9 * MeanSquare + (1 - 0.9) * g ** 2
            g = g / T.sqrt(nextMeanSquare + 0.000001)
            updates.append((MeanSquare, nextMeanSquare))
            updates.append((p, p - self.learning_rate * g))


        train_model = theano.function(
            [x,y],
            cost,
            updates=updates,
            givens={

                self.drop: numpy.cast['int32'](1)

            }, allow_input_downcast=True
        )

        return Model(train=train_model,
                     test=test_model,
                     validate=validate_model,
                     demonstrate=demonstrate_model,
                     layers=self.layers)


if __name__ == "__main__":
    mb = ModelBuilder()
    m = mb.build_model()
    import IPython
    IPython.embed()