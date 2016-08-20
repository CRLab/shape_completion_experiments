import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv3d2d import conv3d
import binvox_rw
import visualization.visualize as viz
import tf_conversions
import PyKDL
from off_utils.off_handler import OffHandler
import math
import pcl
import plyfile

import rospy
import actionlib
import shape_msgs.msg
import geometry_msgs.msg
import graspit_shape_completion.msg

import subprocess
import os
import binvox_rw

PATCH_SIZE = 40

assert False
if __name__ == "__main__":

    models_dir = "/srv/data/shape_completion_data/asymetric/"
    
    models = os.listdir(models_dir)

    for model_name in models:

        #for model_name in ["rubbermaid_ice_guard_pitcher_blue"]:
        view_per_model_count = 0
        
        model_path  = models_dir + model_name
        meshes_path = model_path + "/meshes/"
        if "." in model_name:
            continue
        if not os.path.exists(meshes_path):
            os.mkdir(meshes_path)
        
        for item in os.listdir(model_path):
            if not os.path.isdir(model_path + "/" + item):
                pass
                #print model_path + "/" + item
                #print meshes_path + item
                #os.rename(model_path + "/" + item, meshes_path + item)
                
        cmd_str = "meshlabserver -i "  + meshes_path + model_name + "_scaled.off" + " -o " + meshes_path + model_name + "_scaled.ply"
        #subprocess.call(cmd_str.split(" "))
         
        cmd_str = "pcl_ply2pcd " + meshes_path + model_name + "_scaled.ply " + meshes_path + model_name + "_scaled.pcd"
        print cmd_str
        subprocess.call(cmd_str.split(" "))
        """
        gt = models_dir + model_name + "/meshes/" + model_name + ".pcd"
        gt_mesh = models_dir + model_name + "/meshes/" + model_name + ".ply"
        
        if os.path.exists(gt):
            pose_dir = models_dir + model_name + "/pointclouds/"
            #pose_dir = "/home/jvarley/shape_completion_data2/" + model_name + "/"
            for datafile in os.listdir(pose_dir):
                #if "model_pose.npy" in datafile:
                if "pose.npy" in datafile:
                    view_per_model_count += 1
                    if view_per_model_count > 10:
                        view_per_model_count = 0
                        break
                    print "working on: "+ pose_dir + datafile
                    pose = pose_dir + datafile
                    pc = pose_dir + datafile.replace("pose.npy", "pc.npy")
                    partial_name = datafile.strip("_pose.npy")
                    align_gt_and_partial(gt,gt_mesh, pose, pc, model_name, partial_name)
                    

        """
