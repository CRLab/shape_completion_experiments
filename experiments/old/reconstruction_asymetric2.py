#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
if not os.environ.has_key('DISPLAY'):
    import matplotlib 
    matplotlib.use('Agg')
from datasets import ycb_reconstruction_dataset
from datasets import hdf5_reconstruction_dataset
from datasets import ycb_hdf5_reconstruction_dataset
from datasets import asymetric_h5py_reconstruction_dataset


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam
import visualization.visualize as viz
import mcubes
import cProfile
import pstats
import StringIO
import time
import shutil
import numpy as np



def numpy_jaccard_similarity(a, b):
    '''
    Returns the number of pixels of the intersection of two voxel grids divided
    by the number of pixels in the union.
    The inputs are expected to be numpy 5D ndarrays in BZCXY format.
    '''
    return np.mean(np.sum(a * b, axis=1) / np.sum((a + b) - a * b, axis=1))


def train(model, dataset):

    with open(LOSS_FILE, "w") as loss_file:
        print("logging loss")

    with open(ERROR_TRAINED_VIEWS, "w"):
        print("logging error for trained views")

    with open(ERROR_HOLDOUT_VIEWS, "w"):
        print("logging error for holdout views")

    with open(ERROR_HOLDOUT_MODELS, "w"):
        print("logging error for holdout models")

    with open(JACCARD_TRAINED_VIEWS, "w"):
        print("logging jaccard_error for trained views")

    with open(JACCARD_HOLDOUT_VIEWS, "w"):
        print("logging jaccard_error for holdout views")

    with open(JACCARD_HOLDOUT_MODELS, "w"):
        print("logging jaccard_error for holdout models")

    lowest_error = 1000000
    highest_jaccard = 0

    for e in range(NB_EPOCH):
        print 'Epoch: ' + str(e)
        PR.enable()

        train_iterator = dataset.train_iterator(batch_size=BATCH_SIZE,
                                                flatten_y=True)

        holdout_view_iterator = dataset.holdout_view_iterator(batch_size=BATCH_SIZE,
                                                              flatten_y=True)

        holdout_model_iterator = dataset.holdout_model_iterator(batch_size=BATCH_SIZE,
                                                                flatten_y=True)

        for b in range(NB_TRAIN_BATCHES):
            X_batch, Y_batch = train_iterator.next()
            loss = model.train(X_batch, Y_batch)
            print 'loss: ' + str(loss)
            with open(LOSS_FILE, "a") as loss_file:
                loss_file.write(str(loss) + '\n')

        average_error = 0
        average_holdout_model_jaccard_similarity = 0
        for b in range(NB_TEST_BATCHES):

            X_batch, Y_batch = holdout_view_iterator.next()
            error = model.test(X_batch, Y_batch)
            prediction = model._predict(X_batch)
            binarized_prediction = np.array(prediction > 0.5, dtype=int)
            train_jaccard_similarity = numpy_jaccard_similarity(Y_batch,
                                                          binarized_prediction)
            print('error: ' + str(error))
            print('jaccard_similarity: ' + str(train_jaccard_similarity))
            with open(ERROR_TRAINED_VIEWS, "a") as error_file:
                error_file.write(str(error) + '\n')
            with open(JACCARD_TRAINED_VIEWS, "a") as jaccard_file:
                jaccard_file.write(str(train_jaccard_similarity) + '\n')

            X_batch, Y_batch = holdout_view_iterator.next()
            error = model.test(X_batch, Y_batch)
            prediction = model._predict(X_batch)
            binarized_prediction = np.array(prediction > 0.5, dtype=int)
            holdout_view_jaccard_similarity = numpy_jaccard_similarity(Y_batch,
                                                          binarized_prediction)
            print('error: ' + str(error))
            print('jaccard_similarity: ' + str(holdout_view_jaccard_similarity))
            with open(ERROR_HOLDOUT_VIEWS, "a") as error_file:
                error_file.write(str(error) + '\n')
            with open(JACCARD_HOLDOUT_VIEWS, "a") as jaccard_file:
                jaccard_file.write(str(holdout_view_jaccard_similarity) + '\n')

            average_error += error

            X_batch, Y_batch = holdout_model_iterator.next()
            error = model.test(X_batch, Y_batch)
            prediction = model._predict(X_batch)
            binarized_prediction = np.array(prediction > 0.5, dtype=int)
            holdout_model_jaccard_similarity = numpy_jaccard_similarity(Y_batch,
                                                          binarized_prediction)

            average_holdout_model_jaccard_similarity += holdout_model_jaccard_similarity

            print('error: ' + str(error))
            print('jaccard_similarity: ' + str(holdout_model_jaccard_similarity))
            with open(ERROR_HOLDOUT_MODELS, "a") as error_file:
                error_file.write(str(error) + '\n')
            with open(JACCARD_HOLDOUT_MODELS, "a") as jaccard_file:
                jaccard_file.write(str(holdout_model_jaccard_similarity) + '\n')

        average_error /= NB_TEST_BATCHES
        average_holdout_model_jaccard_similarity /= NB_TEST_BATCHES

        if e % 4 == 0:
            model.save_weights(CURRENT_WEIGHT_FILE)

        if e > 10 and e % 4 ==0 and lowest_error >= average_error:
            lowest_error = average_error
            print('new lowest error ' + str(lowest_error))
            model.save_weights(BEST_WEIGHT_FILE)

        if e > 10 and e % 4 ==0 and highest_jaccard <= average_holdout_model_jaccard_similarity:
            highest_jaccard = average_holdout_model_jaccard_similarity
            print('new highest highest_jaccard ' + str(highest_jaccard))
            model.save_weights(BEST_WEIGHT_FILE_JACCARD)

        if e % 10 == 0:
            test(model, dataset, e)

        PR.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(PR, stream=s).sort_stats(sortby)
        ps.print_stats()
        stats_str = s.getvalue()
        f = open(PROFILE_FILE, 'w')
        f.write(stats_str)
        f.close()


def test(model, dataset, epoch=-1):

    train_iterator = dataset.train_iterator(batch_size=BATCH_SIZE,
                                            flatten_y=False)

    holdout_view_iterator = dataset.holdout_view_iterator(batch_size=BATCH_SIZE,
                                                          flatten_y=False)

    holdout_model_iterator = dataset.holdout_model_iterator(batch_size=BATCH_SIZE,
                                                            flatten_y=False)

    if epoch == -1:
        base_dir = 'final/'
    else:
        base_dir = 'epoch_' + str(epoch) + '/'

    sub_dir = base_dir + 'trained_views/'
    os.makedirs(TEST_OUTPUT_DIR + sub_dir)

    batch_x, batch_y = train_iterator.next()

    pred = model._predict(batch_x)
    # Prediction comes in format [batch number, z-axis, patch number, x-axis,
    #                             y-axis].
    pred = pred.reshape(BATCH_SIZE, PATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE)
    # Convert prediction to format [batch number, x-axis, y-axis, z-axis,
    #                               patch number].
    pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    for i in range(BATCH_SIZE):

        v, t = mcubes.marching_cubes(pred_as_b012c[i, :, :, :, 0], 0.5)
        # Save predicted object mesh.
        mcubes.export_mesh(v, t, TEST_OUTPUT_DIR + sub_dir + 'model_' + str(i) + '.dae',
                           'model')

        # Save visualizations of the predicted, input, and expected occupancy
        # grids.
        viz.visualize_batch_x(pred, i, str(i),
                              TEST_OUTPUT_DIR + sub_dir + "pred_" + str(i))
        viz.visualize_batch_x(batch_x, i, str(i),
                              TEST_OUTPUT_DIR + sub_dir + "input_" + str(i))
        viz.visualize_batch_x(batch_y, i, str(i),
                              TEST_OUTPUT_DIR + sub_dir + "expected_" + str(i))

    sub_dir = base_dir + 'holdout_views/'
    os.makedirs(TEST_OUTPUT_DIR + sub_dir)

    batch_x, batch_y = holdout_view_iterator.next()
    pred = model._predict(batch_x)
    pred = pred.reshape(BATCH_SIZE, PATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE)
    pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    for i in range(BATCH_SIZE):
        v, t = mcubes.marching_cubes(pred_as_b012c[i, :, :, :, 0], 0.5)
        mcubes.export_mesh(v, t, TEST_OUTPUT_DIR + sub_dir + 'model_' + str(i) + '.dae',
                           'model')
        # Save visualizations of the predicted, input, and expected occupancy
        # grids.
        viz.visualize_batch_x(pred, i, str(i),
                              TEST_OUTPUT_DIR + sub_dir + "pred_" + str(i))
        viz.visualize_batch_x(batch_x, i, str(i),
                              TEST_OUTPUT_DIR + sub_dir + "input_" + str(i))
        viz.visualize_batch_x(batch_y, i, str(i),
                              TEST_OUTPUT_DIR + sub_dir + "expected_" + str(i))

    sub_dir = base_dir + 'holdout_models/'
    os.makedirs(TEST_OUTPUT_DIR + sub_dir)

    batch_x, batch_y = holdout_model_iterator.next()

    pred = model._predict(batch_x)
    pred = pred.reshape(BATCH_SIZE, PATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE)
    pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    for i in range(BATCH_SIZE):
        v, t = mcubes.marching_cubes(pred_as_b012c[i, :, :, :, 0], 0.5)
        mcubes.export_mesh(v, t, TEST_OUTPUT_DIR + sub_dir + 'model_' + str(i) + '.dae',
                           'model')
        # Save visualizations of the predicted, input, and expected occupancy
        # grids.
        viz.visualize_batch_x(pred, i, str(i),
                              TEST_OUTPUT_DIR + sub_dir + "pred_" + str(i))
        viz.visualize_batch_x(batch_x, i, str(i),
                              TEST_OUTPUT_DIR + sub_dir + "input_" + str(i))
        viz.visualize_batch_x(batch_y, i, str(i),
                              TEST_OUTPUT_DIR + sub_dir + "expected_" + str(i))


def get_model():

    model = Sequential()

    
    filter_size = 4
    nb_filter_in = 1
    nb_filter_out = 64

    # input: 1 cube of side length 40
    # output: 64 cubes of side length 40-4+1 = 37
    model.add(Convolution3D(nb_filter=nb_filter_out,
                            stack_size=nb_filter_in,
                            nb_row=filter_size,
                            nb_col=filter_size,
                            nb_depth=filter_size,
                            border_mode='valid',
                            activation='relu', init='he_normal'))
    # output: 64 cubes of side length 36/2 = 18
    #model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(.5))

    filter_size = 4
    nb_filter_in = nb_filter_out
    nb_filter_out = 64

    # output: 64 cubes of side length 37-4+1 = 34
    model.add(Convolution3D(nb_filter=nb_filter_out,
                            stack_size=nb_filter_in,
                            nb_row=filter_size,
                            nb_col=filter_size,
                            nb_depth=filter_size,
                            border_mode='valid',
                            activation='relu', init='he_normal'))

    # output: 64 cubes of size length 15/2 = 7
    #model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # During training: drop (set to zero) each of the current outputs with a 0.5
    # probability.
    # During testing: multiply each of the current outputs by that probability.
    #model.add(Dropout(.5))

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    filter_size = 4
    nb_filter_in = nb_filter_out
    nb_filter_out = 64

    # output: 64 cubes of side length 17-4+1 = 14
    model.add(Convolution3D(nb_filter=nb_filter_out,
                            stack_size=nb_filter_in,
                            nb_row=filter_size,
                            nb_col=filter_size,
                            nb_depth=filter_size,
                            border_mode='valid',
                            activation='relu', init='he_normal'))
    #model.add(Dropout(.5))
    #model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    
    filter_size = 4
    nb_filter_in = nb_filter_out
    nb_filter_out = 64

    # output: 64 cubes of side length 14-4+1 = 11
    model.add(Convolution3D(nb_filter=nb_filter_out,
                            stack_size=nb_filter_in,
                            nb_row=filter_size,
                            nb_col=filter_size,
                            nb_depth=filter_size,
                            border_mode='valid',
                            activation='relu', init='he_normal'))
    

    # During training: drop (set to zero) each of the current outputs with a 0.5
    # probability.
    # During testing: multiply each of the current outputs by that probability.
    #model.add(Dropout(.5))
    
    filter_size = 4
    nb_filter_in = nb_filter_out
    nb_filter_out = 50

    # output: 64 cubes of side length 11-4+1 = 8
        model.add(Convolution3D(nb_filter=nb_filter_out,
                            stack_size=nb_filter_in,
                            nb_row=filter_size,
                            nb_col=filter_size,
                            nb_depth=filter_size,
                            border_mode='valid',
                            activation='relu', init='he_normal'))
    
    dim = 8

    # output: a vector of size 64*5*5*5 = 8000
    model.add(Flatten())
    model.add(Dropout(.5))

    model.add(Dense(nb_filter_out * dim * dim * dim, 3000, init='he_normal', activation='relu'))

    #model.add(Dense(3000, 3000, init='he_normal', activation='relu'))
    model.add(Dropout(.5))


    model.add(Dense(3000, 5000, init='he_normal', activation='relu'))


    model.add(Dense(5000, PATCH_SIZE * PATCH_SIZE * PATCH_SIZE, init='glorot_normal',
                    activation='sigmoid'))

    optimizer = Adam(lr=.0001)
    model.compile(loss='cross_entropy_error', optimizer=optimizer)

    return model


def get_dataset():

    train_model_names = ['32aa23cc38e64c6f4b3c42e318f3affc',

                         '846957e3db7d41aa17443d0b0092f09',
                         'cat_000001418',
                         'cell_phone_000000118',

                         'dog_000001956',
                         'guitar_000000330',
                         'M000010',
                         'M000025',

                         'M000040',
                         'M000043',
                         'M000049',

                         'M000059',
                         'm87',
                         'M000103',
                         'M000106',
                         'M000112',
                         'M000118',
                         'M000144',
                         'M000145',
                         'M000149']

    holdout_model_names = ['dog_000001140',
                           'jar_000000187',
                           'M000035',
                           'M000037',
                           'M000074',
                           'M000097',
                           'M000148']

    dataset = asymetric_h5py_reconstruction_dataset.AsymetricReconstructionDataset(
        models_dir='/srv/data/shape_completion_data/asymetric/',
        training_model_names=train_model_names,
        holdout_model_names=holdout_model_names,
        )

    return dataset


if __name__ == "__main__":

    RESULTS_DIR = 'results/' + time.strftime("y%y_m%m_d%d_h%H_m%M") + "/"

    PR = cProfile.Profile()

    BATCH_SIZE = 32
    PATCH_SIZE = 40

    NB_TRAIN_BATCHES = 100
    NB_TEST_BATCHES = 10
    NB_EPOCH = 500

    TEST_OUTPUT_DIR = RESULTS_DIR  + "test_output/"
    os.makedirs(TEST_OUTPUT_DIR)
    shutil.copy2(__file__, RESULTS_DIR + __file__)

    LOSS_FILE = RESULTS_DIR + 'loss.txt'
    ERROR_TRAINED_VIEWS = RESULTS_DIR + 'cross_entropy_err_trained_views.txt'
    ERROR_HOLDOUT_VIEWS = RESULTS_DIR + 'cross_entropy_err_holdout_views.txt'
    ERROR_HOLDOUT_MODELS = RESULTS_DIR  + 'cross_entropy_holdout_models.txt'
    JACCARD_TRAINED_VIEWS = RESULTS_DIR  + 'jaccard_err_trained_views.txt'
    JACCARD_HOLDOUT_VIEWS = RESULTS_DIR  + 'jaccard_err_holdout_views.txt'
    JACCARD_HOLDOUT_MODELS = RESULTS_DIR + 'jaccard_err_holdout_models.txt'
    CURRENT_WEIGHT_FILE = RESULTS_DIR  + 'current_weights.h5'
    BEST_WEIGHT_FILE = RESULTS_DIR + 'best_weights.h5'
    BEST_WEIGHT_FILE_JACCARD = RESULTS_DIR  + 'best_weights_jaccard.h5'
    PROFILE_FILE = RESULTS_DIR + 'profile.txt'

    print('Step 1/4 -- Loading Dataset')
    dataset = get_dataset()
    print('Step 2/4 -- Compiling Model')
    model = get_model()
    print('Step 3/4 -- Training Model')
    train(model, dataset)
    print('Step 4/4 -- Testing Model')
    test(model, dataset, -1)
