#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import time
import shutil
import numpy as np
from utils.reconstruction_utils import *

import test_reconstruction_parser
import importlib
import pcl
import binvox_rw

def prep_for_testing(args):

    # os.makedirs(args.TEST_OUTPUT_DIR)
    # for model_name in args.MODEL_NAMES:
    #     for view_name in args.VIEW_NAMES:
    #         os.makedirs(args.TEST_OUTPUT_DIR + model_name + "/" + view_name)

    shutil.copy2(__file__, args.SCRIPTS_HISTORY_DIR + __file__)
    shutil.copy2(test_reconstruction_parser.__name__ + ".py", args.SCRIPTS_HISTORY_DIR + test_reconstruction_parser.__name__ + ".py")


def get_model(model_python_module, weights_filepath):
    model= importlib.import_module(model_python_module).get_model(args.PATCH_SIZE)
    model.load_weights(weights_filepath)
    return model


def test(model,
    model_pose_filepath,
    partial_view_filepath,
    completion_filepath,
    gt_center_to_upright_filepath):

    pc = pcl.load(partial_view_filepath).to_array()
    model_pose = np.load(model_pose_filepath)
    gt_center_to_upright = np.load(gt_center_to_upright_filepath)

    batch_x = np.zeros((1, args.PATCH_SIZE, args.PATCH_SIZE, args.PATCH_SIZE, 1), dtype=np.float32)

    batch_x[0, :, :, :, :], voxel_resolution, offset = build_test_from_pc_scaled(pc, args.PATCH_SIZE)

    mask = get_occluded_voxel_grid(batch_x[0, :, :, :, 0])

    #make batch B2C01 rather than B012C
    batch_x = batch_x.transpose(0, 3, 4, 1, 2)

    pred = model._predict(batch_x)

    # Prediction comes in format [batch number, z-axis, patch number, x-axis,
    #                             y-axis].
    pred = pred.reshape(1, args.PATCH_SIZE, 1, args.PATCH_SIZE, args.PATCH_SIZE)

    # Convert prediction to format [batch number, x-axis, y-axis, z-axis,
    #                               patch number].
    pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    completed_region = pred_as_b012c[0, :, :, :, 0]

    #output is voxel grid from camera_frame of reference.
    output = completed_region * mask

    output_vox = output > 0.5

    """
    This is the voxel grid in the camera frame of reference.
    vox = binvox_rw.Voxels(output > 0.5,
               (args.PATCH_SIZE, args.PATCH_SIZE, args.PATCH_SIZE),
               (offset[0], offset[1], offset[2]-1),
               voxel_resolution * args.PATCH_SIZE,
               "xyz")
    """



    #go from voxel grid back to list of points. 
    #4xn
    completion = np.zeros((4, len(output_vox.nonzero()[0])))
    completion[0] = (output_vox.nonzero()[0] ) * voxel_resolution + offset[0]
    completion[1] = (output_vox.nonzero()[1] ) * voxel_resolution + offset[1]
    completion[2] = (output_vox.nonzero()[2] ) * voxel_resolution + offset[2]
    completion[3] = 1.0


    world_to_camera_transform = np.array([[0, 0, 1, -1],[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    transform = np.dot(model_pose.T, world_to_camera_transform)
    # 4xn array                                                                 
    completion_rot = np.dot(transform, completion).T

    for i in range(3):
        completion_rot[:,i] += gt_center_to_upright[i]

    center = get_center(completion_rot[:,0:3])
    pc_center_in_voxel_grid = (20, 20, 20)
    offset = np.array(center) - np.array(pc_center_in_voxel_grid) * voxel_resolution

    patch_center_z = (completion_rot[:, 2].max() + completion_rot[:, 2].min())/2.0
    patch_center = (0,0,patch_center_z)

    voxel_grid = create_voxel_grid_around_point_scaled(completion_rot[:,0:3], patch_center,
                                          voxel_resolution, args.PATCH_SIZE,
                                          (20,20,20))

    vox = binvox_rw.Voxels(voxel_grid[:,:,:,0],
                   (args.PATCH_SIZE, args.PATCH_SIZE, args.PATCH_SIZE),
                   (offset[0], offset[1], offset[2]),
                   voxel_resolution * args.PATCH_SIZE,
                   "xyz")


    #if not os.path.exists(completion_filepath):
    binvox_rw.write(vox, open(completion_filepath, 'w'))


if __name__ == "__main__":

    #want to change anything, change the parser file!!!!!!!!
    args = test_reconstruction_parser.get_args()

    print('Step 1/3 -- Prepping For Testing')
    prep_for_testing(args)

    print('Step 2/3 -- Compiling Model')
    model = get_model(args.MODEL_PYTHON_MODULE, args.WEIGHT_FILE)

    print('Step 3/3 -- Testing Model')
    for model_name in args.MODEL_NAMES:
        for view_name in args.VIEW_NAMES:

            partial_view_filepath = args.INPUT_DATA_DIR + args.INPUT_DATASET + model_name + "/pointclouds/_" + view_name + "_pc.pcd"
            model_pose_filepath = args.INPUT_DATA_DIR + args.INPUT_DATASET + model_name + "/pointclouds/_" + view_name + "_model_pose.npy"
            gt_center_to_upright_filepath = args.INPUT_DATA_DIR + args.INPUT_DATASET + model_name + "/meshes/gt_center_to_upright.npy"
            
            completion_filepath = args.TEST_OUTPUT_DIR + model_name + "/" + view_name + "/completion_of.binvox"

            print completion_filepath
            test(model,model_pose_filepath, partial_view_filepath, completion_filepath, gt_center_to_upright_filepath)
