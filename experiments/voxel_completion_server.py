#!/usr/bin/env python

from utils import reconstruction_utils
from shape_completion_server.srv import *
import rospy
import pcl
import numpy as np
import importlib
import scipy.ndimage

class VoxelCompletionServer(object):

    def __init__(self, patch_size, model_python_module, weight_filepath, debug):

        rospy.loginfo("Starting Voxel Completion Server")

        self.patch_size = patch_size
        self.model_python_module = model_python_module
        self.weight_filepath = weight_filepath
        self.debug = debug
        self.mask = False
        self.save_pcd = True
        s = rospy.Service('complete_voxel_grid', VoxelCompletion, self.complete_voxel_grid)

        if self.debug:
            self.model = None
        else:
            self.model= importlib.import_module(model_python_module).get_model(self.patch_size)
            self.model.load_weights(self.weight_filepath)

        rospy.loginfo("debug: " + str(self.debug))
        rospy.loginfo("save_pcd: " + str(self.save_pcd))
        rospy.loginfo("mask: " + str(self.mask))
        rospy.loginfo("weight_filepath: " + str(self.weight_filepath))
        rospy.loginfo("model_python_module: " + str(self.model_python_module))
        rospy.loginfo("Started Voxel Completion Server")

    def save_pc(self, data,filename):
        x = data.nonzero()
        x_np = np.zeros((x[0].shape[0], 3))
        x_np[:,0] = x[0]
        x_np[:,1] = x[1]
        x_np[:,2] = x[2]
        cloud = pcl.PointCloud(np.array(x_np, np.float32))
        pcl.save(cloud, filename)

    def complete_voxel_grid(self, goal):
        batch_x_B012C_flat = np.array(goal.batch_x_B012C_flat)
        
        if self.debug:
          y_filepath = "/srv/data/shape_completion_data/ycb/rubbermaid_ice_guard_pitcher_blue/pointclouds/_6_1_7_y.pcd"
          y_np_pts = pcl.load(y_filepath).to_array().astype(int)
          y_mask = (y_np_pts[:, 0], y_np_pts[:, 1], y_np_pts[:, 2])

          y = np.zeros((1,self.patch_size, self.patch_size, self.patch_size, 1))
          y[0, :, :, :, :][y_mask] = 1
          return y.flatten()
            
        else:
            batch_x_B012C = batch_x_B012C_flat.reshape((1,self.patch_size,self.patch_size,self.patch_size,1))
            #make batch B2C01 rather than B012C                                                      
            batch_x = batch_x_B012C.transpose(0, 3, 4, 1, 2)
            
            pred = self.model._predict(batch_x)
            pred = pred.reshape(1, self.patch_size, 1, self.patch_size, self.patch_size)
            
            pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)
            
            if self.mask:
                mask = reconstruction_utils.get_occluded_voxel_grid(batch_x_B012C[0, :, :, :, 0])
                #mask = scipy.ndimage.binary_dilation(mask)

            completed_region = pred_as_b012c[0, :, :, :, 0]
            
            batch_x = batch_x.transpose(0,3,4,1,2)
            batch_x = batch_x[0,:,:,:,0]
            
            if self.save_pcd:
                self.save_pc(batch_x, "x.pcd")
                self.save_pc(completed_region > 0.5, "y.pcd")
                np.save(open("batch_x.npy", 'w'), batch_x)
                np.save(open("completed_region.npy", 'w'), completed_region > 0.5)
            if self.mask:
                completed_region *= mask
        
        return completed_region.flatten()


if __name__ == "__main__":
    
    patch_size = 40
    model_python_module = "results.y16_m08_d24_h18_m45.conv3_dense2"
    weight_filepath = "/home/jvarley/shape_completion/train/shape_completion_experiments/experiments/results/y16_m08_d24_h18_m45/best_weights.h5"
    debug = False
    
    #model_python_modeul = "results.y16_m08_d19_h18_m"
    rospy.init_node("voxel_completion_node")
    vcs = VoxelCompletionServer(
        patch_size, 
    	model_python_module,
    	weight_filepath,
        debug)

    rospy.spin()
