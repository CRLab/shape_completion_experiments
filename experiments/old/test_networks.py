import os
import subprocess

from datasets import ycb_hdf5_reconstruction_dataset
from experiments.old_train_scripts.reconstruction_3d_ycb_variable_shrec_30_LP2_LD2_final_4000_dense import get_model

if not os.environ.has_key('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')

import visualization.visualize as viz
import mcubes
import time
import numpy as np

PATCH_SIZE = 30
BATCH_SIZE = 50
OUTDIR = "completion_comparison/" + time.strftime("y%y_m%m_d%d_h%H_m%M") + "/"

WEIGHT_FILES = [
    'results/y16_m01_d23_h10_m02/all_ycb_025_shrec/best_weights.h5',
    'results/y16_m01_d23_h10_m02/all_ycb_050_shrec/best_weights.h5',
    'results/y16_m01_d23_h10_m02/all_ycb_100_shrec/best_weights.h5',
    'results/y16_m01_d23_h10_m02/all_ycb_150_shrec/best_weights.h5',
    'results/y16_m01_d23_h10_m02/all_ycb_200_shrec/best_weights.h5',
    'results/y16_m01_d23_h10_m02/all_ycb_250_shrec/best_weights.h5',

    'results/y16_m01_d22_h11_m57/no_ycb_025_shrec/best_weights.h5',
    'results/y16_m01_d22_h11_m57/no_ycb_050_shrec/best_weights.h5',
    'results/y16_m01_d22_h11_m57/no_ycb_100_shrec/best_weights.h5',
    'results/y16_m01_d22_h11_m57/no_ycb_150_shrec/best_weights.h5',
    'results/y16_m01_d22_h11_m57/no_ycb_200_shrec/best_weights.h5',
    'results/y16_m01_d22_h11_m57/no_ycb_250_shrec/best_weights.h5',
]
YCB_MODELS_DIR = '/srv/data/shape_completion_data/ycb_30_h5_dir/'
YCB_MODEL_NAMES = ['black_and_decker_lithium_drill_driver',
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
               'rubbermaid_ice_guard_pitcher_blue',
               'soft_scrub_2lb_4oz',
               'sponge_with_textured_cover']

#YCB_MODEL_NAMES = [ 'frenchs_classic_yellow_mustard_14oz']

def numpy_jaccard_similarity(a, b):
    '''
    Returns the number of pixels of the intersection of two voxel grids divided
    by the number of pixels in the union.
    The inputs are expected to be numpy 5D ndarrays in BZCXY format.
    '''
    #return np.sum(a * b, axis=1) / np.sum((a + b) - a * b, axis=1)
    return (np.sum(a * b, axis=(1,2,3,4)) / np.sum((a + b) - a * b, axis=(1,2,3,4)))




def mesh_voxels(voxels, save_path, mesh_name='mesh'):

    coarse_mesh = save_path + 'coarse_' + mesh_name.replace('.ply', '.dae')
    smooth_mesh = save_path + 'smooth_' + mesh_name

    v, t = mcubes.marching_cubes(voxels, 0.5)
    mcubes.export_mesh(v, t,  coarse_mesh, mesh_name)

    script_file = '/srv/data/temp/' + 'poisson_remesh.mlx'

    cmd_string = 'meshlabserver -i ' + coarse_mesh
    cmd_string = cmd_string + ' -o ' + smooth_mesh
    cmd_string = cmd_string + ' -s ' + script_file

    process = subprocess.call(cmd_string, shell=True)


def test_and_save(batch_x, batch_y, model, type='train_views'):

    sub_dir = OUTDIR + type + '/models/objects/'
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
        print "created sub directory: " + str(sub_dir)

    print "Getting Predictions ..."
    pred = model._predict(batch_x)
    pred = pred.reshape(BATCH_SIZE, PATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE)
    pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    batch_y_as_b012c = batch_y.reshape(BATCH_SIZE, PATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE)
    batch_y_as_b012c = batch_y_as_b012c.transpose(0, 3, 4, 1, 2)

    batch_x_as_b012c = batch_x.reshape(BATCH_SIZE, PATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE)
    batch_x_as_b012c = batch_x_as_b012c.transpose(0, 3, 4, 1, 2)

    np.save(sub_dir + 'batch_x_as_b012c.npy', batch_x_as_b012c)
    np.save(sub_dir + 'batch_y_as_b012c.npy',batch_y_as_b012c)
    np.save(sub_dir + 'pred_as_b012c.npy', pred_as_b012c)

    jaccard_errors = numpy_jaccard_similarity(pred, batch_y)

    jaccard_filepath = sub_dir + "jaccard.csv"

    with open(jaccard_filepath, "w") as jaccard_file:
        jaccard_file.write("index, jaccard\n")

        for i in range(BATCH_SIZE):
            jaccard = jaccard_errors[i]
            jaccard_file.write(str(i) + ',' + str(jaccard) + '\n')

            print "Meshing input, prediction and ground truth for example " + str(i) + '/' + str(BATCH_SIZE)

            mesh_voxels(batch_x_as_b012c[i, :, :, :, 0], sub_dir, 'input_' + str(i) + '.ply')
            viz.visualize_batch_x(batch_x, i, str(i),sub_dir + "input_" + str(i))

            mesh_voxels(batch_y_as_b012c[i, :, :, :, 0], sub_dir, 'gt_' + str(i) + '.ply')
            viz.visualize_batch_x(batch_y, i, str(i), sub_dir + "expected_" + str(i))

            mesh_voxels(pred_as_b012c[i, :, :, :, 0], sub_dir, 'pred_' + str(i) + '.ply')
            viz.visualize_batch_x(pred, i, str(i), sub_dir + "pred_" + str(i))

        jaccard_file.close()







if __name__ == "__main__":


    print "Initializing Dataset..."
    train_dataset = ycb_hdf5_reconstruction_dataset.YcbReconstructionDataset(
        YCB_MODELS_DIR, YCB_MODEL_NAMES)

    print "Initializing Iterator..."
    train_iterator = train_dataset.iterator(batch_size=BATCH_SIZE,
                                      num_batches=1,
                                      flatten_y=False)
    print "Getting Batch..."
    batch_x, batch_y = train_iterator.next(train=1)

    print "Getting Model..."
    model = get_model()

    for weight_file in WEIGHT_FILES:
        print "Running for weight_file: " + str(weight_file) + str('...')
        model.load_weights(weight_file)
        type = weight_file.replace('.h5', "").replace('results/', "").replace('/', '_')
        test_and_save(batch_x, batch_y, model, type=type)

