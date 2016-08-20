import tf_conversions
import pcl
import PyKDL
import numpy as np
import binvox_rw
import math

USE_THEANO=False
OLD_FIT = False
PERCENT_PATCH_SIZE = (4.0/5.0)

if OLD_FIT:
    PERCENT_PATCH_SIZE = (2.0/3.0)

PERCENT_X = 0.5
PERCENT_Y = 0.5
PERCENT_Z = 0.45

class PrecomputeWorker():

    def __init__(self):
        # if USE_THEANO:
        #     self.map_partial_to_camera_frame_fn = self.get_map_partial_to_camera_frame_fn()
        #     self.map_gt_to_camera_frame_fn = self.get_map_gt_to_camera_frame_fn()


        self.current_binvox_filepath = None
        self.current_binvox_np = None

    def get_voxel_resolution(self, pc, patch_size):
        min_x = pc[0, :].min()
        min_y = pc[1, :].min()
        min_z = pc[2, :].min()
        max_x = pc[0, :].max()
        max_y = pc[1, :].max()
        max_z = pc[2, :].max()

        max_dim = max((max_x - min_x),
                      (max_y - min_y),
                      (max_z - min_z))

        voxel_resolution = (1.0*max_dim) / (PERCENT_PATCH_SIZE * patch_size)
        return voxel_resolution


    def get_center(self, pc):

        min_x = pc[0, :].min()
        min_y = pc[1, :].min()
        min_z = pc[2, :].min()
        max_x = pc[0, :].max()
        max_y = pc[1, :].max()
        max_z = pc[2, :].max()

        center = np.array((min_x + (max_x - min_x) / 2.0,
                           min_y + (max_y - min_y) / 2.0,
                           min_z + (max_z - min_z) / 2.0))

        return center

    # def get_map_partial_to_camera_frame_fn(self):
    #     import theano.tensor as T
    #     import theano
    #
    #     pc = T.fmatrix('pc')
    #     trans_matrix = T.fmatrix('trans_matrix')
    #
    #     pc2_out = T.dot(trans_matrix, pc)
    #
    #     f = theano.function([trans_matrix,pc], pc2_out, allow_input_downcast=True)
    #     return f


    def map_partial_to_camera_frame(self, pc):
        trans_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0),
                                  PyKDL.Vector(0, 0, -1))
        trans_matrix = tf_conversions.posemath.toMatrix(trans_frame)

        if USE_THEANO:
            pc2_out = self.map_partial_to_camera_frame_fn(trans_matrix, pc)
        else:
            pc2_out = np.dot(trans_matrix, pc)

        return pc2_out

    # def get_map_gt_to_camera_frame_fn(self):
    #     import theano.tensor as T
    #     import theano
    #
    #     gt_pc = T.fmatrix('gt_pc')
    #     trans_matrix = T.fmatrix('trans_matrix')
    #     rot_matrix = T.fmatrix('rot_matrix')
    #     model_pose = T.fmatrix('model_pose')
    #
    #     temp0 = T.dot(model_pose, gt_pc)
    #     temp1 = T.dot(rot_matrix.T, temp0)
    #     gt_pc_cf = T.dot(trans_matrix.T, temp1)
    #
    #     f = theano.function([gt_pc, trans_matrix, rot_matrix, model_pose], gt_pc_cf, allow_input_downcast=True)
    #     return f

    def map_gt_to_camera_frame(self, gt_pc, model_pose):

        trans_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0),
                                  PyKDL.Vector(0, 0, -1))
        trans_matrix = tf_conversions.posemath.toMatrix(trans_frame)

        # go from camera coords to world coords
        rot_frame = PyKDL.Frame(PyKDL.Rotation.RPY(-math.pi / 2, 0, -math.pi / 2),
                                PyKDL.Vector(0, 0, 0))
        rot_matrix = tf_conversions.posemath.toMatrix(rot_frame)
        #
        # if USE_THEANO:
        #     gt_pc_cf = self.map_gt_to_camera_frame_fn(gt_pc, trans_matrix, rot_matrix, model_pose)
        # else:
        temp = np.dot(model_pose, gt_pc)
        temp = np.dot(rot_matrix.T, temp)
        gt_pc_cf = np.dot(trans_matrix.T, temp)

        return gt_pc_cf

    # def get_create_voxel_grid_around_point_scaled_fn(self):
    #     import theano.tensor as T
    #     import theano
    #
    #     patch_center = T.fvector('patch_center')
    #     points = T.fmatrix('points')
    #     voxel_grid = T.btensor4('voxel_grid')
    #     pc_center_in_voxel_grid = T.fvector('pc_center_in_voxel_grid')
    #     voxel_resolution = T.fscalar('voxel_resolution')
    #     num_voxels_per_dim = T.fscalar('num_voxels_per_dim')
    #
    #     centered_scaled_points = T.floor(
    #         (points - patch_center +
    #             pc_center_in_voxel_grid * voxel_resolution) / voxel_resolution)
    #
    #     mask = centered_scaled_points.max(axis=1) < num_voxels_per_dim
    #     centered_scaled_points = centered_scaled_points[mask]
    #
    #     mask = centered_scaled_points.min(axis=1) > 0
    #     centered_scaled_points = centered_scaled_points[mask]
    #
    #     #csp_int = centered_scaled_points.astype(int)
    #     mask = T.cast(centered_scaled_points, "int32")
    #
    #     #mask = (csp_int[:, 0], csp_int[:, 1], csp_int[:, 2],
    #     #        T.zeros((csp_int.shape[0]), dtype=int))
    #
    #     voxel_grid[(mask.nonzero())] = 1
    #
    #     f = theano.function([points,
    #                          patch_center,
    #                          voxel_resolution,
    #                          num_voxels_per_dim,
    #                          pc_center_in_voxel_grid],
    #                         voxel_grid, allow_input_downcast=True)
    #     return f

    def create_voxel_grid_around_point_scaled(self, points, patch_center,
                                              voxel_resolution, num_voxels_per_dim,
                                              pc_center_in_voxel_grid):
        voxel_grid = np.zeros((num_voxels_per_dim,
                               num_voxels_per_dim,
                               num_voxels_per_dim,
                               1), dtype=np.bool)

        # if USE_THEANO:
        #     voxel_grid = self.create_voxel_grid_around_point_scaled_fn(points,
        #                                                                patch_center,
        #                                                                voxel_resolution,
        #                                                                num_voxels_per_dim,
        #                                                                pc_center_in_voxel_grid)
        #
        #     return voxel_grid

        centered_scaled_points = np.floor(
            (points - np.array(patch_center) + np.array(
                pc_center_in_voxel_grid) * voxel_resolution) / voxel_resolution)

        mask = centered_scaled_points.max(axis=1) < num_voxels_per_dim
        centered_scaled_points = centered_scaled_points[mask]

        if centered_scaled_points.shape[0] == 0:
            return voxel_grid

        mask = centered_scaled_points.min(axis=1) > 0
        centered_scaled_points = centered_scaled_points[mask]

        if centered_scaled_points.shape[0] == 0:
            return voxel_grid

        csp_int = centered_scaled_points.astype(int)

        mask = (csp_int[:, 0], csp_int[:, 1], csp_int[:, 2],
                np.zeros((csp_int.shape[0]), dtype=int))

        voxel_grid[mask] = 1

        return voxel_grid


    def get_gt_pc(self,binvox_file_path):

        if self.current_binvox_filepath == binvox_file_path:
            return np.copy(self.current_binvox_np)

        with open(binvox_file_path, 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)

        points = model.data
        binvox_scale = model.scale
        binvox_offset = model.translate
        dims = model.dims

        non_zero_points = points.nonzero()

        # get numpy array of nonzero points
        num_points = len(non_zero_points[0])
        non_zero_arr = np.zeros((4, num_points))

        non_zero_arr[0] = non_zero_points[0]
        non_zero_arr[1] = non_zero_points[1]
        non_zero_arr[2] = non_zero_points[2]
        non_zero_arr[3] = 1.0

        binvox_offset = np.array(binvox_offset).reshape(3, 1)

        # go from binvox to off original mesh
        non_zero_arr[0:3, :] = non_zero_arr[0:3, :] / \
                               np.array(dims).reshape(3, 1) * binvox_scale
        non_zero_arr[0:3, :] = non_zero_arr[0:3, :] + binvox_offset

        # # go from off to mesh
        # non_zero_arr[0:3, :] = non_zero_arr[0:3, :] * custom_scale
        # non_zero_arr[0:3, :] = non_zero_arr[0:3, :] - np.array(custom_offset).reshape(3,1)
        self.current_binvox_filepath = binvox_file_path
        self.current_binvox_np = np.copy(non_zero_arr)
        return non_zero_arr

    def voxelgrid2pc(self, voxels):
        non_zero_points = voxels.nonzero()

        # get numpy array of nonzero points
        num_points = len(non_zero_points[0])
        non_zero_arr = np.zeros((4, num_points))

        non_zero_arr[0] = non_zero_points[0]
        non_zero_arr[1] = non_zero_points[1]
        non_zero_arr[2] = non_zero_points[2]
        non_zero_arr[3] = 1.0

        return np.array(non_zero_arr, np.float32)

    def load_pcd(self, filepath):
        pcd = pcl.load(filepath)
        # gt_np shape is (#pts, 3)
        temp = pcd.to_array()
        pcd_np = np.ones((temp.shape[0], 4))
        pcd_np[:, 0:3] = temp
        return pcd_np.T

    def build_training_example_scaled(self,
                                      binvox_file_path,
                                      model_pose_filepath,
                                      single_view_pointcloud_filepath,
                                      x_filepath,
                                      y_filepath,
                                      patch_size):

        #4xn
        partial_pc = self.load_pcd(single_view_pointcloud_filepath)
        #4x4
        model_pose = np.load(model_pose_filepath)
        #4xn
        gt_pc = self.get_gt_pc(binvox_file_path)
        #4xn
        pc2_cf = self.map_partial_to_camera_frame(partial_pc)
        #4xn
        gt_pc_cf = self.map_gt_to_camera_frame(gt_pc, model_pose)

        center = self.get_center(pc2_cf)
        voxel_resolution = self.get_voxel_resolution(pc2_cf, patch_size)

        pc_center_in_voxel_grid = (patch_size*PERCENT_X, patch_size*PERCENT_Y, patch_size*PERCENT_Z)

        x = self.create_voxel_grid_around_point_scaled(pc2_cf[0:3, :].T, center,
                                                  voxel_resolution,
                                                  num_voxels_per_dim=patch_size,
                                                  pc_center_in_voxel_grid=pc_center_in_voxel_grid)

        y = self.create_voxel_grid_around_point_scaled(gt_pc_cf[0:3,:].T,
                                                  center,
                                                  voxel_resolution,
                                                  num_voxels_per_dim=patch_size,
                                                  pc_center_in_voxel_grid=pc_center_in_voxel_grid)

        #4xn
        x_pc = self.voxelgrid2pc(x)
        y_pc = self.voxelgrid2pc(y)

        try:
            pcl.save(pcl.PointCloud(x_pc[0:3, :].T), x_filepath, binary=True)
            pcl.save(pcl.PointCloud(y_pc[0:3, :].T), y_filepath, binary=True)
        except Exception as e:
            print "Failed to save: " + str(x_filepath)

if __name__ == "__main__":

    binvox_file_path = "/srv/data/shape_completion_data/ycb/black_and_decker_lithium_drill_driver/models/black_and_decker_lithium_drill_driver.binvox"
    model_pose_filepath = "/home/jvarley/8_18_11_40/black_and_decker_lithium_drill_driver/pointclouds/_2_3_6_model_pose.npy"
    single_view_pointcloud_filepath = "/home/jvarley/8_18_11_40/black_and_decker_lithium_drill_driver/pointclouds/_2_3_6_pc.pcd"
    x_filepath = "/home/jvarley/x.pcd"
    y_filepath = "/home/jvarley/y.pcd"
    patch_size = 40
    custom_scale=1
    custom_offset=(0, 0, 0)

    worker = PrecomputeWorker()
    import time
    import cProfile
    import StringIO
    import pstats
    start_time = time.time()
    PROFILE_FILE =   'profile.txt'
    PR = cProfile.Profile()
    PR.enable()
    for i in range(3):
        print i
        worker.build_training_example_scaled(binvox_file_path,
            model_pose_filepath,
            single_view_pointcloud_filepath,
            x_filepath,
            y_filepath,
            patch_size)

    PR.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(PR, stream=s).sort_stats(sortby)
    ps.print_stats()
    stats_str = s.getvalue()
    f = open(PROFILE_FILE, 'w')
    f.write(stats_str)
    f.close()
    end_time = time.time()
    print end_time - start_time