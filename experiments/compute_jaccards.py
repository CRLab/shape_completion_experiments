#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
if not os.environ.has_key('DISPLAY'):
    import matplotlib 
    matplotlib.use('Agg')

from datasets import yaml_dataset

import visualization.visualize as viz
import mcubes
import cProfile
import pstats
import StringIO
import time
import shutil
import numpy as np
from utils.reconstruction_utils import numpy_jaccard_similarity

import compute_jaccards_parser
import importlib


def prep(args):

    os.makedirs(args.RESULTS_DIR)

    shutil.copy2(__file__, args.RESULTS_DIR + __file__)
    shutil.copy2(compute_jaccards_parser.__name__ + ".py", args.RESULTS_DIR + compute_jaccards_parser.__name__ + ".py")
    shutil.copy2("model_templates/" + args.MODEL_PYTHON_MODULE + ".py", args.RESULTS_DIR + args.MODEL_PYTHON_MODULE.split(".")[-1] + ".py")

    with open(args.JACCARD_TRAINED_VIEWS, "w"):
        print("logging jaccard_error for trained views")

    with open(args.JACCARD_HOLDOUT_VIEWS, "w"):
        print("logging jaccard_error for holdout views")

    with open(args.JACCARD_HOLDOUT_MODELS, "w"):
        print("logging jaccard_error for holdout models")


def get_model(model_python_module, weights_filepath):
    model= importlib.import_module("model_templates." + model_python_module).get_model(args.PATCH_SIZE)
    model.load_weights(weights_filepath)
    return model

def get_dataset(dataset_filepath):
    return yaml_dataset.YamlDataset(dataset_filepath)

def log_jaccards(model, iterator, error_log_file, jaccard_log_file):
    X_batch, Y_batch = iterator.next()

    prediction = model._predict(X_batch)
    binarized_prediction = np.array(prediction > 0.5, dtype=int)
    jaccard_similarity = numpy_jaccard_similarity(Y_batch,
                                                  binarized_prediction)

    print('jaccard_similarity: ' + str(jaccard_similarity))

    with open(jaccard_log_file, "a") as jaccard_file:
        jaccard_file.write(str(jaccard_similarity) + '\n')

    return jaccard_similarity

def compute(model, dataset):

        train_iterator = dataset.train_iterator(batch_size=args.BATCH_SIZE,
                                                flatten_y=True)

        holdout_view_iterator = dataset.holdout_view_iterator(batch_size=args.BATCH_SIZE,
                                                              flatten_y=True)

        holdout_model_iterator = dataset.holdout_model_iterator(batch_size=args.BATCH_SIZE,
                                                                flatten_y=True)

        for b in range(args.NB_TEST_BATCHES):

            train_jaccard_similarity = log_jaccards(model,
                 train_iterator, args.ERROR_TRAINED_VIEWS, args.JACCARD_TRAINED_VIEWS)

            holdout_view_jaccard_similarity = log_jaccards(model,
                 holdout_view_iterator, args.ERROR_HOLDOUT_VIEWS, args.JACCARD_HOLDOUT_VIEWS)

            holdout_model_jaccard_similarity = log_jaccards(model,
                 holdout_model_iterator, args.ERROR_HOLDOUT_MODELS, args.JACCARD_HOLDOUT_MODELS)



if __name__ == "__main__":

    #want to change anything, change the parser file!!!!!!!!
    args = compute_jaccards_parser.get_args()

    print('Step 1/4 -- Prepping Results Dir')
    prep(args)

    print('Step 2/4 -- Loading Dataset')
    dataset = get_dataset(args.DATASET)
    
    print('Step 3/4 -- Compiling Model')
    model = get_model(args.MODEL_PYTHON_MODULE, args.WEIGHT_FILE)
    
    print('Step 4/4 -- Computing')
    compute(model, dataset)



