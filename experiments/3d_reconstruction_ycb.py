from datasets import ycb_reconstruction_dataset
from datasets import hdf5_reconstruction_dataset
from datasets import ycb_hdf5_reconstruction_dataset
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

BATCH_SIZE = 16
PATCH_SIZE = 30

NB_TRAIN_BATCHES = 100
NB_TEST_BATCHES = 10
# NB_EPOCH = 2000
NB_EPOCH = 500

RESULTS_DIR = 'results/' + time.strftime("y%y_m%m_d%d_h%H_m%M") + "/"
TEST_OUTPUT_DIR = RESULTS_DIR + "test_output/"
os.makedirs(TEST_OUTPUT_DIR)

LOSS_FILE = RESULTS_DIR +'loss.txt'
ERROR_FILE = RESULTS_DIR +'error.txt'
CURRENT_WEIGHT_FILE = RESULTS_DIR + 'current_weights.h5'
BEST_WEIGHT_FILE = RESULTS_DIR + 'best_weights.h5'
PROFILE_FILE = RESULTS_DIR + 'profile.txt'
RUN_SCRIPT = __file__


def train(model, dataset):
    """
    Trains the model, while continuously saving the current weights to CURRENT_WEIGHT_FILE and the best weights to
    BEST_WEIGHTS_FILE. After training is done, it saves profiling information to "profile.txt" after printing it to the
    terminal.

    :param model: a compiled Theano model
    :param dataset: an hdf5 dataset
    :return: void
    """

    with open(LOSS_FILE, "w") as loss_file:
        print("logging loss")

    with open(ERROR_FILE, "w") as error_file:
        print("logging error")

    #save this script so we can call load model to get this model again later.
    shutil.copy2(__file__, RESULTS_DIR + __file__)

    lowest_error = 1000000

    for e in range(NB_EPOCH):
        PR.enable()
        train_iterator = dataset.iterator(batch_size=BATCH_SIZE,
                                          num_batches=NB_TEST_BATCHES,
                                          flatten_y=True)

        for b in range(NB_TRAIN_BATCHES):
            X_batch, Y_batch = train_iterator.next(train=1)
            loss = model.train(X_batch, Y_batch)
            print 'loss: ' + str(loss)
            with open(LOSS_FILE, "a") as loss_file:
                loss_file.write(str(loss) + '\n')

        test_iterator = dataset.iterator(batch_size=BATCH_SIZE,
                                         num_batches=NB_TRAIN_BATCHES,
                                         flatten_y=True)

        average_error = 0
        for b in range(NB_TEST_BATCHES):
            X_batch, Y_batch = test_iterator.next(train=0)
            error = model.test(X_batch, Y_batch)
            average_error += error
            print('error: ' + str(error))
            with open(ERROR_FILE, "a") as error_file:
                error_file.write(str(error) + '\n')
        average_error /= NB_TEST_BATCHES

        if e % 4 == 0:
            model.save_weights(CURRENT_WEIGHT_FILE)

        if lowest_error >= average_error:
            lowest_error = average_error
            print('new lowest error ' + str(lowest_error))
            model.save_weights(BEST_WEIGHT_FILE)

        PR.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(PR, stream=s).sort_stats(sortby)
        ps.print_stats()
        stats_str = s.getvalue()
        f = open(PROFILE_FILE, 'w')
        f.write(stats_str)
        f.close()


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
                                      num_batches=NB_TEST_BATCHES,
                                      flatten_y=False)

    batch_x, batch_y = train_iterator.next(train=0)

    pred = model._predict(batch_x)
    pred = pred.reshape(BATCH_SIZE, PATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE)

    pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    for i in range(BATCH_SIZE):
        v, t = mcubes.marching_cubes(pred_as_b012c[i, :, :, :, 0], 0.5)
        mcubes.export_mesh(v, t, TEST_OUTPUT_DIR + 'drill_' + str(i) + '.dae', 'drill')
        viz.visualize_batch_x(pred, i, str(i), TEST_OUTPUT_DIR + "pred_" + str(i))
        viz.visualize_batch_x(batch_x, i, str(i), TEST_OUTPUT_DIR + "input_" + str(i))
        viz.visualize_batch_x(batch_y, i, str(i), TEST_OUTPUT_DIR + "expected_" + str(i))


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

    # dataset = ycb_reconstruction_dataset.YcbDataset("/srv/data/shape_completion_data/ycb/",
    #   "rubbermaid_ice_guard_pitcher_blue", PATCH_SIZE)
    # dataset = hdf5_reconstruction_dataset.ReconstructionDataset(
    #   '/srv/data/shape_completion_data/ycb/wescott_orange_grey_scissors/h5/wescott_orange_grey_scissors.h5')
    model_names = ['black_and_decker_lithium_drill_driver', 'block_of_wood_6in', 'blue_wood_block_1inx1in',
                   'brine_mini_soccer_ball', 'campbells_condensed_tomato_soup', 'cheerios_14oz',
                   'clorox_disinfecting_wipes_35', 'comet_lemon_fresh_bleach', 'domino_sugar_1lb',
                   'frenchs_classic_yellow_mustard_14oz', 'melissa_doug_farm_fresh_fruit_banana',
                   'melissa_doug_farm_fresh_fruit_lemon', 'morton_salt_shaker', 'play_go_rainbow_stakin_cups_1_yellow',
                   'play_go_rainbow_stakin_cups_2_orange', 'pringles_original', 'red_metal_cup_white_speckles',
                   'rubbermaid_ice_guard_pitcher_blue', 'soft_scrub_2lb_4oz', 'sponge_with_textured_cover']
    model_names = ['black_and_decker_lithium_drill_driver', 'block_of_wood_6in', 'blue_wood_block_1inx1in',
                   'brine_mini_soccer_ball', 'campbells_condensed_tomato_soup', 'cheerios_14oz',
                   'clorox_disinfecting_wipes_35', 'comet_lemon_fresh_bleach', 'domino_sugar_1lb',
                   'frenchs_classic_yellow_mustard_14oz', 'melissa_doug_farm_fresh_fruit_banana',
                   'melissa_doug_farm_fresh_fruit_lemon', 'morton_salt_shaker', 'play_go_rainbow_stakin_cups_1_yellow',
                   'play_go_rainbow_stakin_cups_2_orange', 'pringles_original',
                   'rubbermaid_ice_guard_pitcher_blue', 'soft_scrub_2lb_4oz', 'sponge_with_textured_cover']
    dataset = ycb_hdf5_reconstruction_dataset.YcbReconstructionDataset('/srv/data/shape_completion_data/ycb/',
                                                                       model_names)
    return dataset


def main():
    model = get_model()
    dataset = get_dataset()
    train(model, dataset)
    test(model, dataset, BEST_WEIGHT_FILE)

    print('Script Completed')
    import IPython

    IPython.embed()


if __name__ == "__main__":
    main()