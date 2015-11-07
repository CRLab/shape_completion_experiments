
from datasets import reconstruction_test_dataset
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop
from keras.utils import np_utils, generic_utils

import visualization.visualize as viz
import mcubes
import os
import numpy as np
import cProfile
import pstats
import StringIO
import time
import shutil

PR = cProfile.Profile()

BATCH_SIZE = 3
PATCH_SIZE = 30

NB_TRAIN_BATCHES = 100
NB_TEST_BATCHES = 3
# NB_EPOCH = 2000
NB_EPOCH = 500

RESULTS_DIR = 'results/' + time.strftime("y%y_m%m_d%d_h%H_m%M") + "/"
TEST_OUTPUT_DIR = RESULTS_DIR + "test_output/"
os.makedirs(TEST_OUTPUT_DIR)

LOSS_FILE = RESULTS_DIR +'loss.txt'
ERROR_FILE = RESULTS_DIR +'error.txt'
CURRENT_WEIGHT_FILE = RESULTS_DIR + 'current_weights.h5'
#BEST_WEIGHT_FILE = RESULTS_DIR + 'best_weights.h5'
BEST_WEIGHT_FILE = 'best_weights.h5'
PROFILE_FILE = RESULTS_DIR + 'profile.txt'
RUN_SCRIPT = __file__

def test(model, dataset, weights_filepath):
    """
    Runs the given model on the given dataset using the given weights. Then outputs results into the RESULTS_DIR folder.
    Results include the mesh created by passing the output through the marching cubes algorithm, as well as
    visualizations of the input, output, and reconstructed occupancy voxel grids.

    :param model: a compiled Theano model
    :param dataset: an hdf5 dataset
    :param weights_filepath: the filepath where to find the weights to use
    :return: void
    """

    model.load_weights(weights_filepath)

    train_iterator = dataset.iterator(batch_size=BATCH_SIZE,
                                      num_batches=NB_TEST_BATCHES)

    batch_x = train_iterator.next()

    pred = model._predict(batch_x)
    pred = pred.reshape(BATCH_SIZE, PATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE)

    pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    for i in range(BATCH_SIZE):
        v, t = mcubes.marching_cubes(pred_as_b012c[i, :, :, :, 0], 0.5)
        mcubes.export_mesh(v, t, TEST_OUTPUT_DIR + 'model_' + str(i) + '.dae', 'model')
        viz.visualize_batch_x(pred, i, str(i), TEST_OUTPUT_DIR + "pred_" + str(i))
        viz.visualize_batch_x(batch_x, i, str(i), TEST_OUTPUT_DIR + "input_" + str(i))


def get_model():
    """
    Constructs and compiles a Theano shape completion model using Keras and our custom Convolution3D and MaxPooling3D
    layers.

    :return: a compiled Theano model
    """

    model = Sequential()

    filter_size = 5
    nb_filter_in = 1
    nb_filter_out = 64
    # 30-5+1 = 26
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size,
                            nb_depth=filter_size, border_mode='valid'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(.5))
    # out 13

    filter_size = 4
    nb_filter_in = nb_filter_out
    nb_filter_out = 64
    # 13-4+1 = 10
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size,
                            nb_depth=filter_size, border_mode='valid'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(.5))
    # out 5

    filter_size = 3
    nb_filter_in = nb_filter_out
    nb_filter_out = 64
    # 5-3+1 = 3
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size,
                            nb_depth=filter_size, border_mode='valid'))
    model.add(Dropout(.5))
    # out 3

    dim = 3
    # model.add(Flatten(nb_filter_out*dim*dim*dim))
    model.add(Flatten())
    model.add(Dense(nb_filter_out * dim * dim * dim, 3000, init='normal'))
    model.add(Dense(3000, 4000, init='normal'))
    model.add(Dense(4000, PATCH_SIZE * PATCH_SIZE * PATCH_SIZE, init='normal', activation='sigmoid'))

    # let's train the model using SGD + momentum (how original).
    sgd = RMSprop()
    model.compile(loss='cross_entropy_error', optimizer=sgd)

    return model


def get_dataset():
    """
    :return: an hdf5 dataset
    """

    dataset = reconstruction_test_dataset.TestDataset('/srv/data/shape_completion_data/test/', '', 30)
    return dataset


def main():
    model = get_model()
    dataset = get_dataset()
    test(model, dataset, BEST_WEIGHT_FILE)

    print('Script Completed')
    import IPython

    IPython.embed()


if __name__ == "__main__":
    main()