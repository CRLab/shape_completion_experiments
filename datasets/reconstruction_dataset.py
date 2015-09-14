
import numpy as np
import os
import collections

import binvox_rw
import visualization.visualize as viz
import tf_conversions
import PyKDL

import math

class ReconstructionDataset():

    def __init__(self,
                 models_dir="/srv/3d_conv_data/22_model_reconstruction_1000_rand_rot/models/",
                 pc_dir="/srv/3d_conv_data/22_model_reconstruction_1000_rand_rot/pointclouds/",
                 patch_size=72):

        self.models_dir = models_dir
        self.pc_dir = pc_dir
        self.model_names = os.listdir(models_dir)
        self.patch_size = patch_size

        filenames = []
        for model_name in self.model_names:
            model_files = [pc_dir + model_name + "/" + d for d in os.listdir(pc_dir + model_name) if not os.path.isdir(os.path.join(pc_dir + model_name, d))]
            filenames.append((model_name, model_files))

        self.examples = []
        for item in filenames:
            model_name, file_names = item
            for file_name in file_names:

                if "_pc.npy" in file_name:

                    pointcloud_file = file_name
                    pose_file = file_name.replace("pc", "pose")
                    binvox_model_file = models_dir + model_name + "/optimized_tsdf_texture_mapped_mesh.binvox"

                    self.examples.append((pointcloud_file, pose_file, binvox_model_file))

    def get_num_examples(self):
        return len(self.examples)

    def iterator(self,
                 batch_size=None,
                 num_batches=None,
                 flatten_y=False):

            return ReconstructionIterator(self,
                                          batch_size=batch_size,
                                          num_batches=num_batches,
                                          flatten_y=flatten_y)


def create_voxel_grid_around_point(points, patch_center, voxel_resolution=0.001, num_voxels_per_dim=72):

    voxel_grid = np.zeros((num_voxels_per_dim,
                           num_voxels_per_dim,
                           num_voxels_per_dim,
                           1))

    centered_scaled_points = np.floor((points-patch_center + num_voxels_per_dim/2*voxel_resolution) / voxel_resolution)

    x_valid = [centered_scaled_points[:, 0] < num_voxels_per_dim]
    y_valid = [centered_scaled_points[:, 1] < num_voxels_per_dim]
    z_valid = [centered_scaled_points[:, 2] < num_voxels_per_dim]

    centered_scaled_points = centered_scaled_points[x_valid]
    centered_scaled_points = centered_scaled_points[y_valid]
    centered_scaled_points = centered_scaled_points[z_valid]

    csp_int = centered_scaled_points.astype(int)

    mask = (csp_int[:, 0], csp_int[:, 1], csp_int[:, 2], np.zeros((csp_int.shape[0]), dtype=int))

    voxel_grid[mask] = 1

    return voxel_grid


def map_pointclouds_to_world(pc, non_zero_arr, model_pose):
    #this works, to reorient pointcloud
    #apply the model_pose transform, this is the rotation
    #that was applied to the model in gazebo
    #non_zero_arr1 = np.dot(model_pose, non_zero_arr)
    non_zero_arr1 = non_zero_arr

    #from camera to world
    #the -1 is the fact that the model is 1 meter away from the camera
    dist_to_camera = -1
    trans_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0), PyKDL.Vector(0, 0, dist_to_camera))
    trans_matrix = tf_conversions.posemath.toMatrix(trans_frame)

    #go from camera coords to world coords
    rot_frame = PyKDL.Frame(PyKDL.Rotation.RPY(-math.pi/2, 0, -math.pi/2), PyKDL.Vector(0, 0, 0))
    rot_matrix = tf_conversions.posemath.toMatrix(rot_frame)

    pc2 = np.ones((pc.shape[0], 4))
    pc2[:, 0:3] = pc

    #put point cloud in world frame at origin of world
    pc2_out = np.dot(trans_matrix, pc2.T)
    pc2_out = np.dot(rot_matrix, pc2_out)

    #rotate point cloud by same rotation that model went through
    pc2_out = np.dot(model_pose.T, pc2_out)
    return pc2_out, non_zero_arr1


def map_pointclouds_to_camera_frame(pc, non_zero_arr, model_pose):
    #apply the model_pose transform, this is the rotation
    #that was applied to the model in gazebo
    #non_zero_arr1 = np.dot(model_pose, non_zero_arr)
    non_zero_arr1 = non_zero_arr

    #from camera to world
    #the -1 is the fact that the model is 1 meter away from the camera
    dist_to_camera = -2
    trans_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0), PyKDL.Vector(0, 0, dist_to_camera))
    trans_matrix = tf_conversions.posemath.toMatrix(trans_frame)

    #go from camera coords to world coords
    rot_frame = PyKDL.Frame(PyKDL.Rotation.RPY(-math.pi/2, 0, -math.pi/2), PyKDL.Vector(0, 0, 0))
    rot_matrix = tf_conversions.posemath.toMatrix(rot_frame)

    non_zero_arr1 = np.dot(model_pose, non_zero_arr1)
    non_zero_arr1 = np.dot(rot_matrix.T, non_zero_arr1)
    non_zero_arr1 = np.dot(trans_matrix.T, non_zero_arr1)

    pc2_out = np.ones((pc.shape[0], 4))
    pc2_out[:, 0:3] = pc
    trans_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0), PyKDL.Vector(0, 0, -1))
    trans_matrix = tf_conversions.posemath.toMatrix(trans_frame)
    pc2_out = np.dot(trans_matrix, pc2_out.T)

    return pc2_out, non_zero_arr1


def build_training_example(binvox_file_path, model_pose_filepath, single_view_pointcloud_filepath, patch_size):

    pc = np.load(single_view_pointcloud_filepath)
    pc = pc[:, 0:3]
    model_pose = np.load(model_pose_filepath)
    with open(binvox_file_path, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)

    points = model.data
    scale = model.scale
    translate = model.translate
    dims = model.dims

    non_zero_points = points.nonzero()

    #get numpy array of nonzero points
    num_points = len(non_zero_points[0])
    non_zero_arr = np.zeros((4, num_points))

    non_zero_arr[0] = non_zero_points[0]
    non_zero_arr[1] = non_zero_points[1]
    non_zero_arr[2] = non_zero_points[2]
    non_zero_arr[3] = 1.0

    translate_arr = np.array(translate).reshape(3, 1)

    #meters to centimeters
    scale /= 100
    #inches to meters
    scale /= 2.54

    non_zero_arr[0:3, :] = non_zero_arr[0:3, :] + translate_arr * 1.0/scale

    non_zero_arr[0:3, :] = non_zero_arr[0:3, :] * scale

    #this is an easier task, the y value is always the same. i.e the model standing
    #up at the origin.
    #pc2_out, non_zero_arr1 = map_pointclouds_to_world(pc, non_zero_arr, model_pose)
    pc2_out, non_zero_arr1 = map_pointclouds_to_camera_frame(pc, non_zero_arr, model_pose)

    min_x = pc2_out[0, :].min()
    min_y = pc2_out[1, :].min()
    min_z = pc2_out[2, :].min()

    max_x = pc2_out[0, :].max()
    max_y = pc2_out[1, :].max()
    max_z = pc2_out[2, :].max()

    center = (min_x + (max_x-min_x)/2.0, min_y + (max_y-min_y)/2.0, min_z + (max_z-min_z)/2.0)

    # viz.visualize_pointclouds(pc2_out.T, non_zero_arr1.T[:, 0:3], False, True)
    # import IPython
    # IPython.embed()

    #now non_zero_arr and pc points are in the same frame of reference.
    #since the images were captured with the model at the origin
    #we can just compute an occupancy grid centered around the origin.
    x = create_voxel_grid_around_point(pc2_out[0:3, :].T, center, voxel_resolution=.02, num_voxels_per_dim=patch_size)
    y = create_voxel_grid_around_point(non_zero_arr1.T[:, 0:3], center, voxel_resolution=.02, num_voxels_per_dim=patch_size)

    return x, y



class ReconstructionIterator(collections.Iterator):

    def __init__(self,
                 dataset,
                 batch_size,
                 num_batches,
                 flatten_y,
                 iterator_post_processors=[]):

        self.dataset = dataset

        self.batch_size = batch_size
        self.num_batches = num_batches
        self.flatten_y = flatten_y

        self.iterator_post_processors = iterator_post_processors

    def __iter__(self):
        return self

    def next(self):

        batch_indices = np.random.random_integers(0, self.dataset.get_num_examples()-1, self.batch_size)

        patch_size = self.dataset.patch_size

        batch_x = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1), dtype=np.float32)
        batch_y = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1), dtype=np.float32)

        for i in range(len(batch_indices)):
            index = batch_indices[i]

            single_view_pointcloud_filepath = self.dataset.examples[index][0]
            pose_filepath = self.dataset.examples[index][1]
            model_filepath = self.dataset.examples[index][2]

            #print model_filepath
            #print pose_filepath
            #print single_view_pointcloud_filepath
            x, y = build_training_example(model_filepath, pose_filepath, single_view_pointcloud_filepath, patch_size)

            # viz.visualize_3d(x)
            # viz.visualize_3d(y)
            # viz.visualize_pointcloud(pc2_out[0:3, :].T)

            batch_y[i, :, :, :, :] = y
            batch_x[i, :, :, :, :] = x

        #make batch B2C01 rather than B012C
        batch_x = batch_x.transpose(0, 3, 4, 1, 2)
        batch_y = batch_y.transpose(0, 3, 4, 1, 2)

        #apply post processors to the patches
        for post_processor in self.iterator_post_processors:
            batch_x, batch_y = post_processor.apply(batch_x, batch_y)

        if self.flatten_y:
            batch_y = batch_y.reshape(self.batch_size, patch_size**3)

        return batch_x, batch_y

    def batch_size(self):
        return self.batch_size

    def num_batches(self):
        return self.num_batches

    def num_examples(self):
        return self.dataset.get_num_examples()

