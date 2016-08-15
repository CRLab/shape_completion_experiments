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

def load_partial_np_pc(filepath) :
    partial_np_pc = np.load(filepath)
    
    if partial_np_pc.shape[1] > 4:
        partial_np_pc = partial_np_pc.T
    partial_np_pc = partial_np_pc[:, 0:3]
    return partial_np_pc


def load_gt_np_pc(filepath):
    gt_pcd = pcl.load(filepath)
    # gt_np shape is (#pts, 3)
    gt_np_temp = gt_pcd.to_array()
    gt_np_pc = np.ones((gt_np_temp.shape[0], 4))
    gt_np_pc[:, 0:3] = gt_np_temp
    return gt_np_pc


def map_to_camera_frame(pc, gt_np, gt_mesh_vertices, model_pose):

    gt_np = np.copy(gt_np)
    pc = np.copy(pc)
    gt_mesh_vertices = np.copy(gt_mesh_vertices)
    model_pose = np.copy(model_pose)

    gt_np = gt_np.transpose()
    gt_mesh_vertices = gt_mesh_vertices.transpose()

    dist_to_camera = -1
    trans_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0),
                              PyKDL.Vector(0, 0, dist_to_camera))
    trans_matrix = tf_conversions.posemath.toMatrix(trans_frame)

    # go from camera coords to world coords
    rot_frame = PyKDL.Frame(PyKDL.Rotation.RPY(-math.pi / 2, 0, -math.pi / 2),
                            PyKDL.Vector(0, 0, 0))
    rot_matrix = tf_conversions.posemath.toMatrix(rot_frame)

    gt_np = np.dot(model_pose, gt_np)
    gt_np = np.dot(rot_matrix.T, gt_np)
    gt_np = np.dot(trans_matrix.T, gt_np)

    gt_mesh_vertices = np.dot(model_pose, gt_mesh_vertices)
    gt_mesh_vertices = np.dot(rot_matrix.T, gt_mesh_vertices)
    gt_mesh_vertices = np.dot(trans_matrix.T, gt_mesh_vertices)

    if pc.shape[0] < 5:
        pc = pc.transpose()
    assert pc.shape[0] > 4
    pc2_out = np.ones((pc.shape[0], 4))
    pc2_out[:, 0:3] = pc
    trans_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0),
                              PyKDL.Vector(0, 0, -1))
    trans_matrix = tf_conversions.posemath.toMatrix(trans_frame)
    pc2_out = np.dot(trans_matrix, pc2_out.T)

    return pc2_out, gt_np, gt_mesh_vertices.transpose()


def map_to_object_frame(pc, model_pose):

    dist_to_camera = -1

    rot_frame = PyKDL.Frame(PyKDL.Rotation.RPY(-math.pi / 2, 0, -math.pi / 2),
                            PyKDL.Vector(0, 0, 0))
    rot_matrix = tf_conversions.posemath.toMatrix(rot_frame)

    trans_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0),
                              PyKDL.Vector(0, 0, dist_to_camera))
    trans_matrix = tf_conversions.posemath.toMatrix(trans_frame)

    transform = np.dot(model_pose.T, np.dot(rot_matrix, trans_matrix))

    assert pc.shape[0] > 4
    pc2_out = np.ones((pc.shape[0], 4))
    pc2_out[:, 0:3] = pc

    pc2_out = np.dot(transform, pc2_out.T)

    return pc2_out, transform

def get_completion(partial_np_pc, transform):

    goal = graspit_shape_completion.msg.CompleteMeshGoal()
    assert partial_np_pc.shape[0] > 3
    assert partial_np_pc.shape[1] == 3
    for i in range(partial_np_pc.shape[0]):
        point = partial_np_pc[i, :]
        goal.partial_mesh.vertices.append(geometry_msgs.msg.Point(*point))

    client = actionlib.SimpleActionClient('complete_mesh', graspit_shape_completion.msg.CompleteMeshAction)

    client.wait_for_server()
    client.send_goal(goal)
    client.wait_for_result()
    result = client.get_result()

    data = np.ones((len(result.completed_mesh.vertices), 4))
    for i, point in enumerate(result.completed_mesh.vertices):
        x = point.x
        y = point.y
        z = point.z
        data[i] = np.array((x, y, z, 1))

    data_of = data[:]
    trans_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0),
                              PyKDL.Vector(0, 0, -1))
    trans_matrix = tf_conversions.posemath.toMatrix(trans_frame)

    data = np.dot(trans_matrix, data.T).T
    data_of = np.dot(transform, data_of.T).T

    data = data[:, 0:3]
    data_of = data_of[:, 0:3]
    
    completed_pcd_cf = pcl.PointCloud(np.array(data, np.float32))
    completed_pcd_of = pcl.PointCloud(np.array(data_of, np.float32))

    vg = np.array(result.voxel_grid).reshape((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE))
    vox = binvox_rw.Voxels(vg,
                           (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
                           (result.voxel_offset_x, result.voxel_offset_y, result.voxel_offset_z -1 ),
                           result.voxel_resolution * PATCH_SIZE,
                           "xyz")
    return completed_pcd_cf, completed_pcd_of, vox


def align_gt_and_partial(gt_file_path,
                         gt_mesh_file_path,
                         model_pose_filepath,
                         single_view_pointcloud_filepath,
                         model_name,
                         partial_name ):

    model_pose = np.load(model_pose_filepath)
    partial_np_pc = load_partial_np_pc(single_view_pointcloud_filepath)
    gt_np_pc = load_gt_np_pc(gt_file_path)

    gt_mesh = plyfile.PlyData.read(gt_mesh_file_path)
    gt_mesh_vertices = np.zeros((gt_mesh['vertex']['x'].shape[0], 4))
    gt_mesh_vertices[:, 0] = gt_mesh['vertex']['x']
    gt_mesh_vertices[:, 1] = gt_mesh['vertex']['y']
    gt_mesh_vertices[:, 2] = gt_mesh['vertex']['z']

    partial_np_pc_cf, gt_np_pc_cf, gt_mesh_vertices_cf = map_to_camera_frame(
        partial_np_pc, gt_np_pc, gt_mesh_vertices, model_pose)

    partial_np_pc_of, camera_to_object_transform = map_to_object_frame(
        partial_np_pc, model_pose)
    
    gt_mesh['vertex']['x'] = gt_mesh_vertices_cf[:, 0]
    gt_mesh['vertex']['y'] = gt_mesh_vertices_cf[:, 1]
    gt_mesh['vertex']['z'] = gt_mesh_vertices_cf[:, 2]

    partial_pcd_pc_cf = pcl.PointCloud(np.array(partial_np_pc_cf.transpose()[:, 0:3], np.float32))
    gt_pcd_pc_cf = pcl.PointCloud(np.array(gt_np_pc_cf.transpose()[:, 0:3], np.float32))

    partial_pcd_pc_of = pcl.PointCloud(np.array(partial_np_pc_of.transpose()[:, 0:3], np.float32))
    gt_pcd_pc_of = pcl.PointCloud(np.array(gt_np_pc[:, 0:3], np.float32))
    
    result_folder = "/home/jvarley/shape_completion_data_aligned_all_ycb/" + model_name + "/" + partial_name + "/"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    partial_filepath = result_folder + "c_partial.pcd"
    if not os.path.exists(partial_filepath):
        pcl.save(partial_pcd_pc_cf, partial_filepath)

    gt_filepath = result_folder + "c_gt.pcd"
    if not os.path.exists(gt_filepath):
        pcl.save(gt_pcd_pc_cf, gt_filepath)

    partial_filepath_of = result_folder + "c_partial_of.pcd"
    if not os.path.exists(partial_filepath_of):
        pcl.save(partial_pcd_pc_of, partial_filepath_of)

    gt_filepath_of = result_folder + "c_gt_of.pcd"
    if not os.path.exists(gt_filepath_of):
        pcl.save(gt_pcd_pc_of, gt_filepath_of)

    gt_mesh_filepath = result_folder + "c_gt.ply"
    if not os.path.exists(gt_mesh_filepath):
        gt_mesh.text = True
        gt_mesh.write(open(gt_mesh_filepath, "w"))

    completed_filepath = result_folder + "c_completed.pcd"
    if not os.path.exists(completed_filepath):
       completed_pcd_cf, completed_pcd_of, completed_binvox = get_completion(partial_np_pc, camera_to_object_transform)
       pcl.save(completed_pcd_cf, completed_filepath)
       pcl.save(completed_pcd_of, completed_filepath.replace(".pcd", "_of.pcd"))

       binvox_filepath = result_folder + "c_completed.binvox"
       if not os.path.exists(binvox_filepath):
           binvox_rw.write(completed_binvox, open(binvox_filepath, 'w'))


    camera_to_object_transform_filepath = result_folder + "camera_to_object.npy"
    if not os.path.exists(camera_to_object_transform_filepath):
        np.save(camera_to_object_transform_filepath, camera_to_object_transform.T)


if __name__ == "__main__":

    rospy.init_node('gen_smoothing_data')

    models_dir = "/srv/data/shape_completion_data/ycb/"
    models = os.listdir(models_dir)

    for model_name in models:

        #for model_name in ["rubbermaid_ice_guard_pitcher_blue"]:
        view_per_model_count = 0

        gt = models_dir + model_name + "/meshes/" + model_name + ".pcd"
        gt_mesh = models_dir + model_name + "/meshes/" + model_name + ".ply"
        
        if os.path.exists(gt):
            pose_dir = "/srv/data/shape_completion_data/ycb/" + model_name + "/pointclouds/"
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
                    

