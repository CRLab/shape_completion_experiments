#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datasets import ycb_shrec_hdf5_reconstruction_dataset
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop
import visualization.visualize as viz
import mcubes
import os
import cProfile
import pstats
import StringIO
import time
import shutil

PR = cProfile.Profile()

BATCH_SIZE = 64
PATCH_SIZE = 30

NB_TRAIN_BATCHES = 100
NB_TEST_BATCHES = 10
# NB_EPOCH = 2000
NB_EPOCH = 500

RESULTS_DIR = 'results/' + time.strftime("y%y_m%m_d%d_h%H_m%M") + "/"
TEST_OUTPUT_DIR = RESULTS_DIR + "test_output/"
os.makedirs(TEST_OUTPUT_DIR)

LOSS_FILE = RESULTS_DIR + 'loss.txt'
ERROR_FILE = RESULTS_DIR + 'cross_entropy_error.txt'
CURRENT_WEIGHT_FILE = RESULTS_DIR + 'current_weights.h5'
BEST_WEIGHT_FILE = RESULTS_DIR + 'best_weights.h5'
PROFILE_FILE = RESULTS_DIR + 'profile.txt'
RUN_SCRIPT = __file__


def train(model, dataset):
    """
    Trains the model, while continuously saving the current weights to
    CURRENT_WEIGHT_FILE and the best weights to BEST_WEIGHTS_FILE. After
    training is done, it saves profiling information to "profile.txt" after
    printing it to the terminal.

    :param model: a compiled Theano model
    :param dataset: an hdf5 dataset
    :return: void
    """

    with open(LOSS_FILE, "w"):
        print("logging loss")

    with open(ERROR_FILE, "w"):
        print("logging error")

    # save this script so we can call load model to get this model again later.
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
    Runs the given model on the given dataset using the given weights. Then
    outputs results into the RESULTS_DIR folder. Results include the mesh
    created by passing the output through the marching cubes algorithm, as well
    as visualizations of the input, output, and reconstructed occupancy voxel
    grids.

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
    # Prediction comes in format [batch number, z-axis, patch number, x-axis,
    #                             y-axis].
    pred = pred.reshape(BATCH_SIZE, PATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE)
    # Convert prediction to format [batch number, x-axis, y-axis, z-axis,
    #                               patch number].
    pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    for i in range(BATCH_SIZE):
        # The Marching Cubes algorithm [Lorensen and Cline, 1987] in this case
        # will take a 3d occupancy grid with floating point values in the range
        # [0.0, 1.0] and create a mesh at the places where the occupancy value
        # changes from less than to greater than 0.5.
        # A 2D example is shown below for clarity, where each number represents
        # an occupancy value and the lines represent the boundary where the mesh
        # will approximately go. Note that the mesh will not necessarily be
        # axis-aligned.
        # 0.0 0.2|0.7 0.6 0.8 1.0
        # 0.0 0.2|---|0.6 0.8 1.0
        # 0.0 0.2 0.4|0.6 0.8 1.0
        # 0.0 0.3 0.4|0.6 1.0 1.0
        v, t = mcubes.marching_cubes(pred_as_b012c[i, :, :, :, 0], 0.5)
        # Save predicted object mesh.
        mcubes.export_mesh(v, t, TEST_OUTPUT_DIR + 'model_' + str(i) + '.dae',
                           'model')

        # Save visualizations of the predicted, input, and expected occupancy
        # grids.
        viz.visualize_batch_x(pred, i, str(i),
                              TEST_OUTPUT_DIR + "pred_" + str(i))
        viz.visualize_batch_x(batch_x, i, str(i),
                              TEST_OUTPUT_DIR + "input_" + str(i))
        viz.visualize_batch_x(batch_y, i, str(i),
                              TEST_OUTPUT_DIR + "expected_" + str(i))


def get_model():
    """
    Constructs and compiles a Theano shape completion model using Keras and our
    custom Convolution3D and MaxPooling3D
    layers.

    :return: a compiled Theano model
    """

    # ============
    # The network:
    # ============
    # input size: 1 x [30,30,30]
    # [conv 5^3 "valid"]
    # size now: 64 x [26,26,26]
    # [maxpool 2^3] + [dropout 0.5]
    # size now: 64 x [13,13,13]
    # [conv 4^3 "valid"]
    # size now: 64 x [10,10,10]
    # [maxpool 2^3] + [dropout 0.5]
    # size now: 64 x [5,5,5]
    # [conv 3^3 "valid"]
    # size now: 64 x [3,3,3]
    # [flatten]
    # size now: [64*3*3*3] = [1728]
    # [fully connected]
    # size now: [3000]
    # [fully connected]
    # size now: [4000]
    # [fully connected] + [sigmoid activation]
    # output size: [PATCH_SIZE^3] = [30^3] = [9000]

    model = Sequential()

    filter_size = 5
    nb_filter_in = 1
    nb_filter_out = 64

    # input: 1 cube of side length 30
    # output: 64 cubes of side length 30-5+1 = 26
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in,
                            nb_row=filter_size, nb_col=filter_size,
                            nb_depth=filter_size, border_mode='valid', activation='relu', init='he_normal'))
    # output: 64 cubes of side length 26/2 = 13
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(.5))

    filter_size = 4
    nb_filter_in = nb_filter_out
    nb_filter_out = 64

    # output: 64 cubes of side length 13-4+1 = 10
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in,
                            nb_row=filter_size, nb_col=filter_size,
                            nb_depth=filter_size, border_mode='valid', activation='relu', init='he_normal'))
    # output: 64 cubes of size length 10/2 = 5
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # During training: drop (set to zero) each of the current outputs with a 0.5
    # probability.
    # During testing: multiply each of the current outputs by that probability.
    model.add(Dropout(.5))

    filter_size = 3
    nb_filter_in = nb_filter_out
    nb_filter_out = 64

    # output: 64 cubes of side length 5-3+1 = 3
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in,
                            nb_row=filter_size, nb_col=filter_size,
                            nb_depth=filter_size, border_mode='valid', activation='relu', init='he_normal'))
    # During training: drop (set to zero) each of the current outputs with a 0.5
    # probability.
    # During testing: multiply each of the current outputs by that probability.
    model.add(Dropout(.5))

    dim = 3

    # output: a vector of size 64*3*3*3 = 1728
    model.add(Flatten())
    # output: a vector of size 3000
    model.add(Dense(nb_filter_out * dim * dim * dim, 3000, init='he_normal', activation='relu'))
    # output: a vector of size 4000
    model.add(Dense(3000, 4000, init='he_normal', activation='relu'))
    # output: a vector of size PATCH_SIZE*PATCH_SIZE*PATCH_SIZE
    model.add(Dense(4000, PATCH_SIZE * PATCH_SIZE * PATCH_SIZE, init='glorot_normal',
                    activation='sigmoid'))

    optimizer = RMSprop()
    model.compile(loss='cross_entropy_error', optimizer=optimizer)

    return model


def get_dataset():
    """
    :return: an hdf5 dataset
    """

    ycb_models_dir = '/srv/data/shape_completion_data/ycb/'
    ycb_model_names = ['black_and_decker_lithium_drill_driver',
                   'block_of_wood_6in',
                   'blue_wood_block_1inx1in',
                   'brine_mini_soccer_ball',
                   'campbells_condensed_tomato_soup',
                   'cheerios_14oz',
                   'clorox_disinfecting_wipes_35',
                   'comet_lemon_fresh_bleach',
                   'domino_sugar_1lb',
                   'frenchs_classic_yellow_mustard_14oz',
                   'melissa_doug_farm_fresh_fruit_banana',
                   'melissa_doug_farm_fresh_fruit_lemon',
                   'morton_salt_shaker',
                   'play_go_rainbow_stakin_cups_1_yellow',
                   'play_go_rainbow_stakin_cups_2_orange',
                   'pringles_original',
                   # 'red_metal_cup_white_speckles',
                   'rubbermaid_ice_guard_pitcher_blue',
                   'soft_scrub_2lb_4oz',
                   'sponge_with_textured_cover']

    shrec_models_dir = '/srv/data/shape_completion_data/shrec/train_h5/'
    shrec_model_names = ['D00881', 'D00866', 'D01122', 'D00913', 'D00983',
                         'D00069', 'D00094', 'D01199', 'D00804', 'D00345',
                         'D00898', 'D00598', 'D01040', 'D00575', 'D00582',
                         'D00949', 'D00846', 'D00801', 'D00719', 'D00960',
                         'D00510', 'D00924', 'D01158', 'D00117', 'D01101',
                         'D00557', 'D00746', 'D00062', 'D00918', 'D00317',
                         'D00493', 'D00416', 'D00074', 'D00876', 'D01197',
                         'D00848', 'D00335', 'D00935', 'D00858', 'D00798',
                         'D00537', 'D00972', 'D01037', 'D00613', 'D00059',
                         'D00542', 'D00548', 'D00008', 'D01090', 'D00413',
                         'D00694', 'D00696', 'D00915', 'D00578', 'D00580',
                         'D00586', 'D00115', 'D00853', 'D00357', 'D01127',
                         'D00572', 'D00738', 'D00584', 'D00299', 'D00675',
                         'D00128', 'D01103', 'D00667', 'D00734', 'D01104',
                         'D00259', 'D00665', 'D00132', 'D01065', 'D00264',
                         'D01117', 'D00756', 'D00063', 'D00331', 'D00134',
                         'D00241', 'D00499', 'D01178', 'D00168', 'D00071',
                         'D00533', 'D00194', 'D00534', 'D00963', 'D00731',
                         'D00982', 'D01062', 'D00432', 'D00236', 'D00350',
                         'D00985', 'D00205', 'D00943', 'D00447', 'D00230',
                         'D00437', 'D01138', 'D00998', 'D01136', 'D00198',
                         'D00398', 'D00262', 'D01132', 'D00348', 'D00930',
                         'D00911', 'D00101', 'D00171', 'D01157', 'D00895',
                         'D00527', 'D00981', 'D00623', 'D00722', 'D00301',
                         'D01124', 'D00294', 'D00698', 'D00910', 'D01161',
                         'D00785', 'D01017', 'D00260', 'D00373', 'D01196',
                         'D00886', 'D01080', 'D00728', 'D00903', 'D00780',
                         'D01109', 'D00233', 'D01126', 'D00081', 'D01021',
                         'D00822', 'D00714', 'D01026', 'D00593', 'D00302',
                         'D00465', 'D00559', 'D00880', 'D00156', 'D00420',
                         'D00815', 'D00290', 'D00617', 'D00500', 'D00556',
                         'D00084', 'D00878', 'D00468', 'D00513', 'D00321',
                         'D00825', 'D00877', 'D00169', 'D00604', 'D01160',
                         'D01164', 'D00330', 'D00173', 'D00773', 'D00750',
                         'D00999', 'D00214', 'D00573', 'D01181', 'D00003',
                         'D00962', 'D00833', 'D00402', 'D00621', 'D00597',
                         'D00359', 'D00464', 'D00209', 'D00914', 'D00826',
                         'D00576', 'D00177', 'D00902', 'D00323', 'D00454',
                         'D00226', 'D01005', 'D00337', 'D00529', 'D00588',
                         'D00478', 'D00401', 'D00607', 'D00417', 'D00987',
                         'D00717', 'D00774', 'D01188', 'D00211', 'D00847',
                         'D01076', 'D00625', 'D00012', 'D00687', 'D00737',
                         'D00638', 'D00119', 'D00154', 'D00382', 'D01146',
                         'D00776', 'D00310', 'D00354', 'D00371', 'D00887',
                         'D00346', 'D00342', 'D00139', 'D01045', 'D01018',
                         'D00410', 'D00571', 'D00056', 'D00709', 'D00046',
                         'D00532', 'D00802', 'D00516', 'D00411', 'D00060',
                         'D00541', 'D00656', 'D00823', 'D00916', 'D00352',
                         'D00269', 'D00031', 'D00433', 'D00185', 'D00564',
                         'D00247', 'D01027', 'D00642', 'D00797', 'D00587']

    dataset = ycb_shrec_hdf5_reconstruction_dataset.YcbShrecReconstructionDataset(
        ycb_models_dir, ycb_model_names,
        shrec_models_dir, shrec_model_names)
    return dataset


def main():
    print('Step 1/4 -- Compiling Model')
    model = get_model()
    print('Step 2/4 -- Loading Dataset')
    dataset = get_dataset()
    print('Step 3/4 -- Training Model')
    train(model, dataset)
    print('Step 4/4 -- Testing Model')
    test(model, dataset, BEST_WEIGHT_FILE)

    print('Script Completed')
    import IPython

    IPython.embed()


if __name__ == "__main__":
    main()
