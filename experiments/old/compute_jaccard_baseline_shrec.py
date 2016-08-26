#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datasets import shrec_h5py_reconstruction_dataset, ycb_hdf5_reconstruction_dataset
import visualization.visualize as viz
import cProfile
import numpy as np
from operator import mul

PR = cProfile.Profile()

BATCH_SIZE = 32
PATCH_SIZE = 30

NB_TRAIN_BATCHES = 100


def numpy_jaccard_similarity(a, b):
    '''
    Returns the number of pixels of the intersection of two voxel grids divided
    by the number of pixels in the union.
    The inputs are expected to be numpy 5D ndarrays in BZCXY format.
    '''
    return np.mean(np.sum(a * b, axis=1) / np.sum((a + b) - a * b, axis=1))


def compute_average_jaccard_similarity(dataset):
    """
    Trains the model, while continuously saving the current weights to
    CURRENT_WEIGHT_FILE and the best weights to BEST_WEIGHTS_FILE. After
    training is done, it saves profiling information to "profile.txt" after
    printing it to the terminal.

    :param model: a compiled Theano model
    :param dataset: an hdf5 dataset
    :return: void
    """

    iterator = dataset.iterator(batch_size=BATCH_SIZE,
                                      num_batches=NB_TRAIN_BATCHES,
                                      flatten_y=True)

    jaccard_similarity = 0
    for b in range(NB_TRAIN_BATCHES):
        print 'Batch No.:', b
        X_batch, Y_batch = iterator.next(train=1)
        X_batch = X_batch.reshape(X_batch.shape[0],
                                      reduce(mul, X_batch.shape[1:]))
        jaccard_similarity += numpy_jaccard_similarity(Y_batch, X_batch)
    jaccard_similarity /= NB_TRAIN_BATCHES

    return jaccard_similarity


def get_dataset():
    """
    :return: an hdf5 dataset
    """

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

    dataset = shrec_h5py_reconstruction_dataset.ShrecReconstructionDataset(
        shrec_models_dir, shrec_model_names)

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
    dataset = ycb_hdf5_reconstruction_dataset.YcbReconstructionDataset(
        ycb_models_dir,
        ycb_model_names)

    return dataset


def main():

    dataset = get_dataset()
    print 'Computing the average jaccard similarity'
    avg_jac_sim = compute_average_jaccard_similarity(dataset)
    print 'Avg. Jaccard Similarity:', avg_jac_sim


if __name__ == "__main__":

    main()
    print('Script Completed')
    import IPython
    IPython.embed()
