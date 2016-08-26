# coding: utf-8

import os
import yaml
import random
import pcl
import numpy as np

def build(yaml_file):

    train_model_names = ['32aa23cc38e64c6f4b3c42e318f3affc',
                         '846957e3db7d41aa17443d0b0092f09',
                         'cat_000001418',
                         'cell_phone_000000118',
                         'dog_000001956',
                         'guitar_000000330',
                         'M000010',
                         'M000025',
                         'M000040',
                         'M000043',
                         'M000049',
                         'M000059',
                         'm87',
                         'M000103',
                         'M000106',
                         'M000112',
                         'M000118',
                         'M000144',
                         'M000145',
                         'M000149']

    holdout_model_names = ['dog_000001140',
                           'jar_000000187',
                           'M000035',
                           'M000037',
                           'M000074',
                           'M000097',
                           'M000148']

    patch_size = 40
    dataset = {}
    dataset["train_model_names"] = train_model_names
    dataset["holdout_model_names"] = holdout_model_names
    dataset["patch_size"] = patch_size

    train_models_train_views = []
    train_models_holdout_views = []
    holdout_models_holdout_views = []

    root_path = "/srv/data/shape_completion_data/asymetric"

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
    yaml_file = "ASYMETRIC_Dataset.yaml"
    print "BUILDING DATASET"
    build(yaml_file)

    print "VERIFYING DATASET" 
    Success, problems = verify(yaml_file)
    import IPython
    IPython.embed()
