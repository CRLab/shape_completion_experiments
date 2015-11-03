import numpy as np
import binvox_rw
import mcubes
import os

from reconstruction_utils import create_voxel_grid_around_point

def remesh_binvox_model(model_dir, model_name, results_dir):

    with open(model_dir + model_name + '.binvox', 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)

    points = model.data
    binvox_scale = model.scale
    binvox_offset = model.translate
    dims = model.dims

    binvox_offset = np.array(binvox_offset).reshape(1, 3)
    num_voxels_per_dim = max(dims)

    voxel_grid = np.zeros((num_voxels_per_dim + 2,
                           num_voxels_per_dim + 2,
                           num_voxels_per_dim + 2))

    voxel_grid[1:num_voxels_per_dim+1, 1:num_voxels_per_dim+1, 1:num_voxels_per_dim+1] = points

    v, t = mcubes.marching_cubes(voxel_grid, 0.5)
    v = v * binvox_scale / num_voxels_per_dim + binvox_offset;
    mcubes.export_mesh(v, t, results_dir + model_name + '.dae', model_name)

if __name__=='__main__':

    model_names = ['black_and_decker_lithium_drill_driver', 'block_of_wood_6in', 'block_of_wood_12in', 'blue_wood_block_1inx1in',
                   'brine_mini_soccer_ball', 'campbells_condensed_tomato_soup', 'champion_sports_official_softball', 'cheerios_14oz',
                   'cheeze-it_388g', 'clorox_disinfecting_wipes_35', 'comet_lemon_fresh_bleach', 'domino_sugar_1lb',
                   'frenchs_classic_yellow_mustard_14oz', 'jell-o_chocolate_flavor_pudding', 'jell-o_strawberry_gelatin_dessert',
                   'large_black_spring_clamp', 'master_chef_ground_coffee_297g', 'medium_black_spring_clamp',
                   'melissa_doug_farm_fresh_fruit_apple', 'melissa_doug_farm_fresh_fruit_banana', 'melissa_doug_farm_fresh_fruit_lemon',
                   'melissa_doug_farm_fresh_fruit_orange', 'melissa_doug_farm_fresh_fruit_pear', 'melissa_doug_farm_fresh_fruit_strawberry',
                   'morton_salt_shaker', 'orange_wood_block_1inx1in', 'penn_raquet_ball', 'play_go_rainbow_stakin_cups_1_yellow',
                   'play_go_rainbow_stakin_cups_2_orange', 'play_go_rainbow_stakin_cups_3_red', 'play_go_rainbow_stakin_cups_5_green',
                   'pringles_original', 'purple_wood_block_1inx1in', 'red_metal_bowl_white_speckles', 'red_metal_cup_white_speckles',
                   'red_metal_plate_white_speckles', 'rubbermaid_ice_guard_pitcher_blue', 'soft_scrub_2lb_4oz', 'spam_12oz',
                   'sponge_with_textured_cover']

    for model_name in model_names:
        data_dir = '/srv/data/shape_completion_data/ycb/'
        models_dir = data_dir + model_name + '/models/'
        mesh_dir = data_dir + model_name + '/meshes/'

        remesh_binvox_model(models_dir, model_name, mesh_dir)
