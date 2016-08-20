from datasets import ycb_hdf5_reconstruction_dataset

import numpy as np
import theano
import theano.tensor as T
import time
import os
import shutil
from experiments.results.y16_m07_d26_h18_m46 import \
   reconstruction_ycb_40 as trained_model_file

from utils import reconstruction_utils

def numpy_jaccard_similarity(x, y):


    intersection = np.sum(x * y, axis=1)#x
    intersection = np.sum(intersection, axis=1)#y
    intersection = np.sum(intersection, axis=1) #z

    union = np.sum((x + y) - x * y, axis=1)#x
    union = np.sum(union, axis=1)#y
    union = np.sum(union, axis=1)#z

    jaccard = np.mean(intersection / union, axis=0)

    return jaccard

def build_theano_jaccard():
    dtensor5 = T.TensorType('float32',  broadcastable=(True, False, False, False, False))
    dtensor5broad = T.TensorType('float32',  broadcastable=(False, False, False, False, False))
    x = dtensor5('x')
    y = dtensor5broad('y')

    intersection = T.sum(x * y, axis=1)#x
    intersection = T.sum(intersection, axis=1)#y
    intersection = T.sum(intersection, axis=1) #z

    union = T.sum((x + y) - x * y, axis=1)#x
    union = T.sum(union, axis=1)#y
    union = T.sum(union, axis=1)#z

    jaccard = intersection / union
    jaccard_fn = theano.function([x,y], jaccard)
    intersection_fn = theano.function([x,y], intersection)
    union_fn = theano.function([x,y], union)
    return jaccard_fn, intersection_fn, union_fn


def get_completion(model, example_in ):
    example = example_in[:]
    batch_x = example.transpose(0, 3, 4, 1, 2)

    pred = model._predict(batch_x)
    pred = pred.reshape(1, 40, 1, 40, 40)

    pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    batch_x = batch_x.transpose(0, 3, 4, 1, 2)
    mask = reconstruction_utils.get_occluded_voxel_grid(batch_x[0, :, :, :, 0])
    completed_region = pred_as_b012c[0, :, :, :, 0]

    completed_region *= mask

    completed_region = completed_region.reshape((1, 40, 40, 40, 1))
    completed_region[completed_region < 0.5] = 0
    completed_region[completed_region >= 0.5] = 1

    completed_region += example_in
    completed_region[completed_region >= 0.5] = 1

    return completed_region

if __name__ == "__main__":

    train_model_names = ['black_and_decker_lithium_drill_driver',

                         'blue_wood_block_1inx1in',
                         'brine_mini_soccer_ball',
                         'campbells_condensed_tomato_soup',

                         'clorox_disinfecting_wipes_35',
                         'comet_lemon_fresh_bleach',
                         'domino_sugar_1lb',
                         'frenchs_classic_yellow_mustard_14oz',

                         'melissa_doug_farm_fresh_fruit_lemon',
                         'morton_salt_shaker',
                         'play_go_rainbow_stakin_cups_1_yellow',

                         'pringles_original',
                         'rubbermaid_ice_guard_pitcher_blue',
                         'soft_scrub_2lb_4oz',
                         'sponge_with_textured_cover']

    holdout_model_names = ['block_of_wood_6in',
                           'cheerios_14oz',
                           'melissa_doug_farm_fresh_fruit_banana',
                           'play_go_rainbow_stakin_cups_2_orange']

    dataset = ycb_hdf5_reconstruction_dataset.YcbReconstructionDataset(
        models_dir='/srv/data/shape_completion_data/ycb/',
        training_model_names=train_model_names,
        holdout_model_names=holdout_model_names,
        mesh_folder="/h5_remesh_40_recentered/")

    #40x040x40 weights trained on ~8 different ycb objects
    weights_filepath = '/home/jvarley/shape_completion/train/shape_completion_experiments/experiments/results/y16_m07_d28_h00_m04/best_weights.h5'
    model = trained_model_file.get_model()
    model.load_weights(weights_filepath)

    RESULTS_DIR = time.strftime("y%y_m%m_d%d_h%H_m%M") + "/"
    os.mkdir(RESULTS_DIR)
    shutil.copy2(__file__, RESULTS_DIR + __file__)
    OUTFILE = RESULTS_DIR + "jaccard.csv"
    f = open(OUTFILE, 'w')
    f.write("query_model_name,query_indice,gt_to_completion_jaccard,gt_to_partial_jaccard,completion_to_closest_training_example_jaccard,holdout_model_filename,single_view_pointcloud_filepath\n")
    f.close()
    jaccard_fn, intersection_fn, union_fn = build_theano_jaccard()

    #holdout_views
    query_view_set = dataset.holdout_view_set
    query_model_names = train_model_names
    query_model_dset = dataset.training_dset

    #holdout_models
    # query_view_set = dataset.holdout_model_set
    # query_model_names = holdout_model_names
    # query_model_dset = dataset.holdout_model_dset


    for i, holdout_model_view_indices in enumerate(query_view_set):
        holdout_model_name = query_model_names[i]
        holdout_model_dset = query_model_dset[i]

        #In [23]: dataset.holdout_model_dset[0]['y'].shape
        #Out[23]: (726, 40, 40, 40, 1)
        holdout_model_examples = holdout_model_dset['y']
        holdout_model_partials = holdout_model_dset['x']
        print holdout_model_dset.keys()
        holdout_model_filenames = holdout_model_dset['pose_filepath']
        single_view_pointcloud_filepaths  = holdout_model_dset['single_view_pointcloud_filepath']

        holdout_indices = dataset.holdout_view_set[i]
        #holdout_indices.sort()

        num_views = 0
        for holdout_idx in holdout_indices:
            num_views += 1
            if num_views == 10:
                break
            holdout_example = holdout_model_examples[holdout_idx]
            holdout_example = holdout_example.reshape((1, 40, 40, 40, 1))
            holdout_model_filename = holdout_model_filenames[holdout_idx]
            single_view_pointcloud_filepath = single_view_pointcloud_filepaths[holdout_idx]


            holdout_partial = holdout_model_partials[holdout_idx]
            holdout_partial = holdout_partial.reshape((1, 40, 40, 40, 1))

            gt_to_partial_jaccard = jaccard_fn(holdout_example, holdout_partial)[0,0]

            completed_region = get_completion(model, holdout_partial)
            gt_to_completion_jaccard = jaccard_fn(completed_region, holdout_example)[0,0]

            smallest_jaccard = 10000000
            largest_jaccard = 0

            for j, training_model_view_indices  in enumerate(dataset.train_set):
                training_model_dset = dataset.training_dset[j]
                training_model_name = train_model_names[j]
                print "comparing: " + str(holdout_model_name) + " to: " + str(training_model_name)

                training_examples = training_model_dset['y']
                indices = dataset.train_set[j]

                indices.sort()

                training_examples = np.take(training_examples, indices, axis=0)

                jaccards = jaccard_fn(holdout_example, training_examples)

                if jaccards.min() < smallest_jaccard:
                    smallest_jaccard = jaccards.min()

                if jaccards.max() > largest_jaccard:
                    largest_jaccard = jaccards.max()

            completion_to_closest_training_example_jaccard = largest_jaccard

            f = open(OUTFILE, 'a')

            #f.write("query_model_name, query_indice, gt_to_completion_jaccard, gt_to_partial_jaccard, completion_to_closest_training_example_jaccard \n")
            f.write(holdout_model_name + ",")
            f.write(str(holdout_idx) + ",")
            f.write(str(gt_to_completion_jaccard) + ",")
            f.write(str(gt_to_partial_jaccard) + ",")
            f.write(str(completion_to_closest_training_example_jaccard) + ",")
            f.write(str(holdout_model_filename) + ",")
            f.write(str(single_view_pointcloud_filepath) + "\n")
            f.close()
            print holdout_model_name
            print holdout_idx
            print smallest_jaccard
            print largest_jaccard



    import IPython
    IPython.embed()