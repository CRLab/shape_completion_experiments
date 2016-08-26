#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
if not os.environ.has_key('DISPLAY'):
    import matplotlib 
    matplotlib.use('Agg')

from datasets import yaml_dataset


# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Flatten
# from keras.layers.convolutional import Convolution3D, MaxPooling3D
# from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam
import visualization.visualize as viz
import mcubes
import cProfile
import pstats
import StringIO
import time
import shutil
import numpy as np
from utils.reconstruction_utils import numpy_jaccard_similarity
import subprocess
import continue_training_parser
import importlib

def prep_for_training(args):

    cmd_str = "cp -r " + args.PREVIOUS_RESULT_DIR + " " + args.RESULTS_DIR
    subprocess.call(cmd_str.split(" "))

    shutil.copy2(__file__, args.RESULTS_DIR + __file__)
    shutil.copy2(continue_training_parser.__name__ + ".py", args.RESULTS_DIR + continue_training_parser.__name__ + ".py")


def get_model(model_python_module):
    return importlib.import_module("model_templates." + model_python_module).get_model(args.PATCH_SIZE)


def get_dataset(dataset_filepath):
    return yaml_dataset.YamlDataset(dataset_filepath)


def train(model, dataset, start_epoch):
    PR = cProfile.Profile()

    lowest_error = 1000000
    highest_jaccard = 0

    for e in range(args.NB_EPOCH):
        e += start_epoch
        print 'Epoch: ' + str(e)
        PR.enable()

        train_iterator = dataset.train_iterator(batch_size=args.BATCH_SIZE,
                                                flatten_y=True)

        holdout_view_iterator = dataset.holdout_view_iterator(batch_size=args.BATCH_SIZE,
                                                              flatten_y=True)

        holdout_model_iterator = dataset.holdout_model_iterator(batch_size=args.BATCH_SIZE,
                                                                flatten_y=True)

        for b in range(args.NB_TRAIN_BATCHES):
            X_batch, Y_batch = train_iterator.next()
            loss = model.train(X_batch, Y_batch)
            print 'loss: ' + str(loss)
            with open(args.LOSS_FILE, "a") as loss_file:
                loss_file.write(str(loss) + '\n')

        average_error = 0
        average_holdout_model_jaccard_similarity = 0
        for b in range(args.NB_TEST_BATCHES):

            X_batch, Y_batch = holdout_view_iterator.next()
            error = model.test(X_batch, Y_batch)
            prediction = model._predict(X_batch)
            binarized_prediction = np.array(prediction > 0.5, dtype=int)
            train_jaccard_similarity = numpy_jaccard_similarity(Y_batch,
                                                          binarized_prediction)
            print('error: ' + str(error))
            print('jaccard_similarity: ' + str(train_jaccard_similarity))
            with open(args.ERROR_TRAINED_VIEWS, "a") as error_file:
                error_file.write(str(error) + '\n')
            with open(args.JACCARD_TRAINED_VIEWS, "a") as jaccard_file:
                jaccard_file.write(str(train_jaccard_similarity) + '\n')

            X_batch, Y_batch = holdout_view_iterator.next()
            error = model.test(X_batch, Y_batch)
            prediction = model._predict(X_batch)
            binarized_prediction = np.array(prediction > 0.5, dtype=int)
            holdout_view_jaccard_similarity = numpy_jaccard_similarity(Y_batch,
                                                          binarized_prediction)
            print('error: ' + str(error))
            print('jaccard_similarity: ' + str(holdout_view_jaccard_similarity))
            with open(args.ERROR_HOLDOUT_VIEWS, "a") as error_file:
                error_file.write(str(error) + '\n')
            with open(args.JACCARD_HOLDOUT_VIEWS, "a") as jaccard_file:
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
            with open(args.ERROR_HOLDOUT_MODELS, "a") as error_file:
                error_file.write(str(error) + '\n')
            with open(args.JACCARD_HOLDOUT_MODELS, "a") as jaccard_file:
                jaccard_file.write(str(holdout_model_jaccard_similarity) + '\n')

        average_error /= args.NB_TEST_BATCHES
        average_holdout_model_jaccard_similarity /= args.NB_TEST_BATCHES

        if e % 4 == 0:
            model.save_weights(args.CURRENT_WEIGHT_FILE)

        if e > 10 and e % 4 ==0 and lowest_error >= average_error:
            lowest_error = average_error
            print('new lowest error ' + str(lowest_error))
            model.save_weights(args.BEST_WEIGHT_FILE)

        if e > 10 and e % 4 ==0 and highest_jaccard <= average_holdout_model_jaccard_similarity:
            highest_jaccard = average_holdout_model_jaccard_similarity
            print('new highest highest_jaccard ' + str(highest_jaccard))
            model.save_weights(args.BEST_WEIGHT_FILE_JACCARD)

        if e % 10 == 0:
            test(model, dataset, e)

        PR.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(PR, stream=s).sort_stats(sortby)
        ps.print_stats()
        stats_str = s.getvalue()
        f = open(args.PROFILE_FILE, 'w')
        f.write(stats_str)
        f.close()


def test(model, dataset, epoch=-1):

    train_iterator = dataset.train_iterator(batch_size=args.BATCH_SIZE,
                                            flatten_y=False)

    holdout_view_iterator = dataset.holdout_view_iterator(batch_size=args.BATCH_SIZE,
                                                          flatten_y=False)

    holdout_model_iterator = dataset.holdout_model_iterator(batch_size=args.BATCH_SIZE,
                                                            flatten_y=False)

    if epoch == -1:
        base_dir = 'final/'
    else:
        base_dir = 'epoch_' + str(epoch) + '/'

    sub_dir = base_dir + 'trained_views/'
    os.makedirs(args.TEST_OUTPUT_DIR + sub_dir)

    batch_x, batch_y = train_iterator.next()

    pred = model._predict(batch_x)
    # Prediction comes in format [batch number, z-axis, patch number, x-axis,
    #                             y-axis].
    pred = pred.reshape(args.BATCH_SIZE, args.PATCH_SIZE, 1, args.PATCH_SIZE, args.PATCH_SIZE)
    # Convert prediction to format [batch number, x-axis, y-axis, z-axis,
    #                               patch number].
    pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    for i in range(args.BATCH_SIZE):

        v, t = mcubes.marching_cubes(pred_as_b012c[i, :, :, :, 0], 0.5)
        # Save predicted object mesh.
        mcubes.export_mesh(v, t, args.TEST_OUTPUT_DIR + sub_dir + 'model_' + str(i) + '.dae',
                           'model')

        # Save visualizations of the predicted, input, and expected occupancy
        # grids.
        viz.visualize_batch_x(pred, i, str(i),
                              args.TEST_OUTPUT_DIR + sub_dir + "pred_" + str(i))
        viz.visualize_batch_x(batch_x, i, str(i),
                              args.TEST_OUTPUT_DIR + sub_dir + "input_" + str(i))
        viz.visualize_batch_x(batch_y, i, str(i),
                              args.TEST_OUTPUT_DIR + sub_dir + "expected_" + str(i))

    sub_dir = base_dir + 'holdout_views/'
    os.makedirs(args.TEST_OUTPUT_DIR + sub_dir)

    batch_x, batch_y = holdout_view_iterator.next()
    pred = model._predict(batch_x)
    pred = pred.reshape(args.BATCH_SIZE, args.PATCH_SIZE, 1, args.PATCH_SIZE, args.PATCH_SIZE)
    pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    for i in range(args.BATCH_SIZE):
        v, t = mcubes.marching_cubes(pred_as_b012c[i, :, :, :, 0], 0.5)
        mcubes.export_mesh(v, t, args.TEST_OUTPUT_DIR + sub_dir + 'model_' + str(i) + '.dae',
                           'model')
        # Save visualizations of the predicted, input, and expected occupancy
        # grids.
        viz.visualize_batch_x(pred, i, str(i),
                              args.TEST_OUTPUT_DIR + sub_dir + "pred_" + str(i))
        viz.visualize_batch_x(batch_x, i, str(i),
                              args.TEST_OUTPUT_DIR + sub_dir + "input_" + str(i))
        viz.visualize_batch_x(batch_y, i, str(i),
                              args.TEST_OUTPUT_DIR + sub_dir + "expected_" + str(i))

    sub_dir = base_dir + 'holdout_models/'
    os.makedirs(args.TEST_OUTPUT_DIR + sub_dir)

    batch_x, batch_y = holdout_model_iterator.next()

    pred = model._predict(batch_x)
    pred = pred.reshape(args.BATCH_SIZE, args.PATCH_SIZE, 1, args.PATCH_SIZE, args.PATCH_SIZE)
    pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    for i in range(args.BATCH_SIZE):
        v, t = mcubes.marching_cubes(pred_as_b012c[i, :, :, :, 0], 0.5)
        mcubes.export_mesh(v, t, args.TEST_OUTPUT_DIR + sub_dir + 'model_' + str(i) + '.dae',
                           'model')
        # Save visualizations of the predicted, input, and expected occupancy
        # grids.
        viz.visualize_batch_x(pred, i, str(i),
                              args.TEST_OUTPUT_DIR + sub_dir + "pred_" + str(i))
        viz.visualize_batch_x(batch_x, i, str(i),
                              args.TEST_OUTPUT_DIR + sub_dir + "input_" + str(i))
        viz.visualize_batch_x(batch_y, i, str(i),
                              args.TEST_OUTPUT_DIR + sub_dir + "expected_" + str(i))




if __name__ == "__main__":

    #want to change anything, change the parser file!!!!!!!!
    args = continue_training_parser.get_args()

    print('Step 1/5 -- Prepping For Training')
    prep_for_training(args)
    print('Step 2/5 -- Loading Dataset')
    dataset = get_dataset(args.DATASET_FILEPATH)
    print('Step 3/5 -- Compiling Model')
    model = get_model(args.MODEL_PYTHON_MODULE)
    model.load_weights(args.CURRENT_WEIGHT_FILE)
    print('Step 4/5 -- Training Model')
    train(model, dataset, args.START_EPOCH)
    print('Step 5/5 -- Testing Model')
    test(model, dataset, -1)
