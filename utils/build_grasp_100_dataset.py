# coding: utf-8

import os
import yaml
import random
import pcl
import numpy as np

def build(yaml_file):
    #adding these to grasp_25_database.yaml
    train_model_names = ['box_poisson_016',
                         'violin_poisson_010',
                         'toaster_poisson_001',
                         'pliers_poisson_002',
                         'flashlight_poisson_004',
                         'violin_poisson_012',
                         'toy_poisson_024',
                         'cellphone_poisson_035',
                         'hammer_poisson_017',
                         'can_poisson_002',
                         'tetra_pak_poisson_023',
                         'cellphone_poisson_015',
                         'donut_poisson_006',
                         'screwdriver_poisson_023',
                         'hammer_poisson_025',
                         'drill_poisson_000',
                         'screwdriver_poisson_011',
                         'toy_poisson_004',
                         'spray_can_poisson_004',
                         'donut_poisson_004',
                         'watering_can_poisson_006',
                         'mushroom_poisson_004',
                         'cellphone_poisson_004',
                         'screwdriver_poisson_013',
                         'spray_bottle_poisson_001',
                         'bottle_new_poisson_000',
                         'trash_can_poisson_009',
                         'tetra_pak_poisson_017',
                         'bowl_poisson_022',
                         'box_poisson_024',
                         'jar_poisson_019',
                         'pliers_poisson_010',
                         'banana_poisson_002',
                         'jar_poisson_011',
                         'wrench_poisson_022',
                         'pitcher_poisson_001',
                         'box_poisson_017',
                         'box_poisson_022',
                         'camera_poisson_013',
                         'light_bulb_poisson_011',
                         'tape_poisson_003',
                         'horseshoe_poisson_001',
                         'wrench_poisson_020',
                         'shampoo_new_poisson_001',
                         'can_poisson_011',
                         'pumpkin_poisson_000',
                         'cellphone_poisson_025',
                         'box_new_poisson_000',
                         'trash_can_poisson_001',
                         'egg_poisson_002',
                         'can_poisson_000',
                         'toaster_poisson_010',
                         'egg_poisson_000',
                         'trash_can_poisson_010',
                         'remote_poisson_018',
                         'dumpbell_poisson_000',
                         'knife_poisson_007',
                         'tetra_pak_poisson_004',
                         'remote_poisson_008',
                         'egg_poisson_012']
                        
    holdout_model_names = [ 'banana_poisson_005',
                         'camera_poisson_014',
                         'trash_can_poisson_021',
                         'flashlight_poisson_001',
                         'jar_poisson_013',

                         'box_poisson_002',
                         'book_poisson_002',
                         'bowl_poisson_015',
                         'can_poisson_020',
                         'flashlight_poisson_005',

                         'mushroom_poisson_013',
                         'hammer_poisson_001',
                         'soccer_ball_poisson_007',
                         'mushroom_poisson_000',
                         'banjo_poisson_002']
    
    patch_size = 40
    dataset = {}
    dataset["train_model_names"] = train_model_names
    dataset["holdout_model_names"] = holdout_model_names
    dataset["patch_size"] = patch_size

    train_models_train_views = []
    train_models_holdout_views = []
    holdout_models_holdout_views = []

    for model in holdout_model_names:
        pointclouds_dir = root_path + "/" + model + "/pointclouds"
        if os.path.exists(pointclouds_dir):
            for mfile in os.listdir(pointclouds_dir):
                if "x.pcd" in mfile:
                    x = pointclouds_dir + "/" + mfile
                    y = x.replace("x.pcd","y.pcd")
                    if verify_example(x, y, patch_size):
                        holdout_models_holdout_views.append((x,y))
                    

    for model in train_model_names:
        pointclouds_dir = root_path + "/" + model + "/pointclouds"
        if os.path.exists(pointclouds_dir):
            for mfile in os.listdir(pointclouds_dir):
                if "x.pcd" in mfile:
                    x = pointclouds_dir + "/" + mfile
                    y = x.replace("x.pcd","y.pcd")
                    if random.random() < .8:
                        if verify_example(x, y, patch_size):
                            train_models_train_views.append((x,y))
                    else:
                        if verify_example(x, y, patch_size):
                            train_models_holdout_views.append((x,y))
                        
    dataset['train_models_train_views'] = train_models_train_views
    dataset['train_models_holdout_views'] = train_models_holdout_views
    dataset['holdout_models_holdout_views'] = holdout_models_holdout_views

    dataset.keys()
    for key in dataset.keys():
        print key
        
    with open(yaml_file, "w") as outfile:
        yaml.dump(dataset, outfile, default_flow_style=True)
     

def verify_example(x_filepath, y_filepath, patch_size):

    success = True

    x = pcl.load(x_filepath).to_array().astype(int)
    x_mask = (x[:, 0], x[:, 1], x[:, 2])
    y = pcl.load(y_filepath).to_array().astype(int)
    y_mask = (y[:, 0], y[:, 1], y[:, 2])

    voxel_x = np.zeros(
        (patch_size, patch_size, patch_size, 1),
        dtype=np.float32)
    voxel_y = np.zeros(
        (patch_size, patch_size, patch_size, 1),
        dtype=np.float32)

    voxel_y[y_mask] = 1
    voxel_x[x_mask] = 1

    x_count = np.count_nonzero(voxel_x)

    overlap = voxel_x*voxel_y
    overlap_count = np.count_nonzero(overlap)

    #this should be high, almost all points should be in gt
    percent_overlap = float(overlap_count) / float(x_count)

    overlap_threshold = 0.8
    if percent_overlap < overlap_threshold:
        print "pcl_viewer " + x_filepath + " has overlap < " + str(overlap_threshold)
        success = False

    pts_threshold = 100
    if x_count < pts_threshold:
        print "pcl_viewer " + x_filepath + " has less than " + str(pts_threshold) + " points !!"
        success = False

    return success


def verify(yaml_file):

    Success = True
    problems = []

    dataset = yaml.load(open(yaml_file, 'r'))
    for key in ['train_models_train_views','train_models_holdout_views','holdout_models_holdout_views']:
        for x_filepath, y_filepath in dataset[key]:

            patch_size = dataset["patch_size"]
            if not verify_example(x_filepath, y_filepath, patch_size):
                Success = False
                problems.append(x_filepath)

    return Success, problems

if __name__ == "__main__":

    root_path = "/srv/data/shape_completion_data/grasp_database"
    yaml_file = "GRASP_25_100_Dataset.yaml"

    print "BUILDING DATASET"
    build(yaml_file)

    print "VERIFYING DATASET" 
    Success, problems = verify(yaml_file)
    import IPython
    IPython.embed()
