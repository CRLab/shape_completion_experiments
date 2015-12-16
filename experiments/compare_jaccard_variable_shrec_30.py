#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datasets import shrec_h5py_reconstruction_dataset
from datasets import shrec_h5py_holdout_dataset
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
import numpy as np
from operator import mul

PR = cProfile.Profile()

BATCH_SIZE = 32
PATCH_SIZE = 30

NB_BATCHES = 100


def numpy_jaccard_similarity(a, b):
    '''
    Returns the number of pixels of the intersection of two voxel grids divided
    by the number of pixels in the union.
    '''
    return np.mean(np.sum(a * b, axis=1) / np.sum((a + b) - a * b, axis=1))


def compute_jaccard_similarities(model, train_dataset, test_dataset, weights_filepath):

    model.load_weights(weights_filepath)

    train_iterator = train_dataset.iterator(batch_size=BATCH_SIZE,
                                      num_batches=NB_BATCHES,
                                      flatten_y=True)
    test_iterator1 = train_dataset.iterator(batch_size=BATCH_SIZE,
                                     num_batches=NB_BATCHES,
                                     flatten_y=True)
    test_iterator2 = test_dataset.iterator(batch_size=BATCH_SIZE,
                                     num_batches=NB_BATCHES,
                                     flatten_y=True)

    jaccard_pred_trained_views = 0.0
    jaccard_base_trained_views = 0.0
    jaccard_pred_holdout_views = 0.0
    jaccard_base_holdout_views = 0.0
    jaccard_pred_holdout_models = 0.0
    jaccard_base_holdout_models = 0.0
    for b in range(NB_BATCHES):
        print('Batch No.: ', b)
        X_batch, Y_batch = test_iterator1.next(train=1)
        error = model.test(X_batch, Y_batch)
        prediction = model._predict(X_batch)
        binarized_prediction = np.array(prediction > 0.5, dtype=int)
        jaccard_pred_trained_views += numpy_jaccard_similarity(Y_batch,
                                                      binarized_prediction)
        X_batch = X_batch.reshape(X_batch.shape[0],
                                      reduce(mul, X_batch.shape[1:]))
        jaccard_base_trained_views += numpy_jaccard_similarity(Y_batch, X_batch)


        X_batch, Y_batch = test_iterator1.next(train=0)
        error = model.test(X_batch, Y_batch)
        prediction = model._predict(X_batch)
        binarized_prediction = np.array(prediction > 0.5, dtype=int)
        jaccard_pred_holdout_views += numpy_jaccard_similarity(Y_batch,
                                                      binarized_prediction)
        X_batch = X_batch.reshape(X_batch.shape[0],
                                      reduce(mul, X_batch.shape[1:]))
        jaccard_base_holdout_views += numpy_jaccard_similarity(Y_batch, X_batch)

        X_batch, Y_batch = test_iterator2.next()
        error = model.test(X_batch, Y_batch)
        prediction = model._predict(X_batch)
        binarized_prediction = np.array(prediction > 0.5, dtype=int)
        jaccard_pred_holdout_models += numpy_jaccard_similarity(Y_batch,
                                                      binarized_prediction)
        X_batch = X_batch.reshape(X_batch.shape[0],
                                      reduce(mul, X_batch.shape[1:]))
        jaccard_base_holdout_models += numpy_jaccard_similarity(Y_batch, X_batch)

    jaccard_pred_trained_views /= NB_BATCHES
    jaccard_base_trained_views /= NB_BATCHES
    jaccard_pred_holdout_views /= NB_BATCHES
    jaccard_base_holdout_views /= NB_BATCHES
    jaccard_pred_holdout_models /= NB_BATCHES
    jaccard_base_holdout_models /= NB_BATCHES

    F.write('Jaccard Predicted Trained Views: ' + str(jaccard_pred_trained_views) + '\n')
    F.write('Jaccard Base Trained Views: ' + str(jaccard_base_trained_views) + '\n')
    F.write('Jaccard Predicted Holdout Views: ' + str(jaccard_pred_holdout_views) + '\n')
    F.write('Jaccard Base Holdout Views: ' + str(jaccard_base_holdout_views) + '\n')
    F.write('Jaccard Predicted Holdout Models: ' + str(jaccard_pred_holdout_models) + '\n')
    F.write('Jaccard Base Holdout Models: ' + str(jaccard_base_holdout_models) + '\n')


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
    model.add(Dense(nb_filter_out * dim * dim * dim, 3000, activation='relu', init='he_normal'))
    # output: a vector of size 4000
    model.add(Dense(3000, 4000, activation='relu', init='he_normal'))
    # output: a vector of size PATCH_SIZE*PATCH_SIZE*PATCH_SIZE
    model.add(Dense(4000, PATCH_SIZE * PATCH_SIZE * PATCH_SIZE, init='glorot_normal',
                    activation='sigmoid'))

    optimizer = RMSprop()
    model.compile(loss='cross_entropy_error', optimizer=optimizer)

    return model


def get_dataset(num_shrec_models):
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

    shrec_model_names = shrec_model_names[0:num_shrec_models]
    train_dataset = shrec_h5py_reconstruction_dataset.ShrecReconstructionDataset(
        shrec_models_dir, shrec_model_names)

    shrec_test_models_dir = '/srv/data/shape_completion_data/shrec/test_h5/'
    shrec_test_model_names = ['D00152', 'D00966', 'D00748', 'D00562', 'D00512',
                              'D00208', 'D00265', 'D01063', 'D00362', 'D00199',
                              'D00842', 'D00857', 'D00551', 'D00218', 'D00800',
                              'D00045', 'D00051', 'D00308', 'D01171', 'D00017',
                              'D00786', 'D00770', 'D00849', 'D01106', 'D00470',
                              'D00220', 'D00712', 'D01047', 'D00681', 'D00400',
                              'D00662', 'D00928', 'D00940', 'D00313', 'D00502']
    test_dataset = shrec_h5py_holdout_dataset.ShrecHoldoutDataset(
        shrec_test_models_dir, shrec_test_model_names)

    return train_dataset, test_dataset


def test_script(num_shrec_models):

    print('Testing on all YCB models + ' + str(num_shrec_models) + ' SHREC models:')
    F.write('All YCB models + ' + str(num_shrec_models) + ' SHREC models\n')

    print('Step 1/3 -- Compiling Model')
    model = get_model()
    print('Step 2/3 -- Loading Dataset')
    train_dataset, test_dataset = get_dataset(num_shrec_models)
    print('Step 3/3 -- Compute Jaccard Similarities')
    compute_jaccard_similarities(model, train_dataset, test_dataset, BEST_WEIGHT_FILE)


if __name__ == "__main__":

    #RESULTS_DIR = 'results/' + time.strftime("y%y_m%m_d%d_h%H_m%M") + "/"
    RESULTS_DIR = 'results/y15_m12_d02_h20_m03_ycb_and_shrec/'
    #os.makedirs(RESULTS_DIR)

    # save this script so we can call load model to get this model again later.
    #shutil.copy2(__file__, RESULTS_DIR + __file__)

    for i in range(6):
        if i == 0:
            num_shrec_models = 25
        else:
            num_shrec_models = i * 50
        SUB_DIR = 'all_ycb_' + '%03d'%(num_shrec_models) + '_shrec/'

        #TEST_OUTPUT_DIR = RESULTS_DIR + SUB_DIR + "test_output/"
        #os.makedirs(TEST_OUTPUT_DIR)
        #LOSS_FILE = RESULTS_DIR + SUB_DIR + 'loss.txt'
        #ERROR_TRAINED_VIEWS = RESULTS_DIR + SUB_DIR + 'cross_entropy_err_trained_views.txt'
        #ERROR_HOLDOUT_VIEWS = RESULTS_DIR + SUB_DIR + 'cross_entropy_err_holdout_views.txt'
        #ERROR_HOLDOUT_MODELS = RESULTS_DIR + SUB_DIR + 'cross_entropy_holdout_models.txt'
        #JACCARD_TRAINED_VIEWS = RESULTS_DIR + SUB_DIR + 'jaccard_err_trained_views.txt'
        #JACCARD_HOLDOUT_VIEWS = RESULTS_DIR + SUB_DIR + 'jaccard_err_holdout_views.txt'
        #JACCARD_HOLDOUT_MODELS = RESULTS_DIR + SUB_DIR + 'jaccard_err_holdout_models.txt'
        CURRENT_WEIGHT_FILE = RESULTS_DIR + SUB_DIR + 'current_weights.h5'
        BEST_WEIGHT_FILE = RESULTS_DIR + SUB_DIR + 'best_weights.h5'
        #PROFILE_FILE = RESULTS_DIR + SUB_DIR + 'profile.txt'
        RESULTS_FILE = RESULTS_DIR + 'jaccard_comparisons.txt'
        F = open(RESULTS_FILE, 'w');
        F.write('Folder: ' + RESULTS_DIR + '\n\n')

        test_script(num_shrec_models)

    print('Script Completed')
    import IPython

    IPython.embed()
