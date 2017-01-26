
import time
import argparse
	
#IF you want to test a new model, go ahead and modify the default 
#args in this file. Then go ahead and run it, this file will be copied into the
#results directory so it is ok to modify.

def get_args():

	#training_dir = 'y16_m08_d19_h12_m00'
        training_dir = 'y16_m08_d24_h18_m45'

	training_results_dir = "/home/jvarley/shape_completion/train/shape_completion_experiments/experiments/results/"
	scripts_history_dir = "/home/jvarley/mnt_curacao/post_processing/data/scripts_history/"
	test_results_dir = "/home/jvarley/mnt_curacao/post_processing/data/shape_completion_results/"
	input_data_dir = "/srv/data/shape_completion_data/"
	input_dataset = "grasp_database/"

	parser = argparse.ArgumentParser(description='Push examples through a trained model.')

	parser.add_argument('--PATCH_SIZE', type=int, 
		default=40)

	parser.add_argument('--INPUT_DATA_DIR', type=str, 
		default=input_data_dir)

	parser.add_argument('--INPUT_DATASET', type=str, 
		default=input_dataset)

	parser.add_argument('--SCRIPTS_HISTORY_DIR', type=str, 
		default=scripts_history_dir)

	parser.add_argument('--TEST_OUTPUT_DIR', type=str, 
		default=test_results_dir)

	parser.add_argument('--WEIGHT_FILE', type=str,
	 	default=training_results_dir + training_dir + '/best_weights.h5')

	parser.add_argument('--MODEL_PYTHON_MODULE', type=str,
	 	#default="results.y16_m08_d19_h12_m00.reconstruction_ycb_40_5")
                        default="results.y16_m08_d24_h18_m45.conv3_dense2")

	args = parser.parse_args()
        """
	args.MODEL_NAMES = ['black_and_decker_lithium_drill_driver',
						'brine_mini_soccer_ball',
						'campbells_condensed_tomato_soup',
						'clorox_disinfecting_wipes_35',
						'comet_lemon_fresh_bleach',
						'domino_sugar_1lb',
						'frenchs_classic_yellow_mustard_14oz',
						'melissa_doug_farm_fresh_fruit_lemon',
						'morton_salt_shaker',
						'play_go_rainbow_stakin_cups_1_yellow',
						'pringles_original',
						'rubbermaid_ice_guard_pitcher_blue',
						'soft_scrub_2lb_4oz',
						'sponge_with_textured_cover',
						'block_of_wood_6in',
						'cheerios_14oz',
						'melissa_doug_farm_fresh_fruit_banana',
						'play_go_rainbow_stakin_cups_2_orange']
        """
	args.VIEW_NAMES = ['5_1_10',
						'3_5_9',
						'4_4_0',
						'6_1_2',
						'8_4_0',
						'0_5_2',
						'6_3_2',
						'1_4_0',
						'0_1_6',
						'3_0_1']

        
        args.MODEL_NAMES=['flashlight_poisson_002',
         'notebook_poisson_035',
         'box_poisson_019',
         'spray_can_poisson_001',
         'cellphone_poisson_016',
         'banana_poisson_005',
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
         'banjo_poisson_002',
         'flashlight_poisson_006',
         'hammer_poisson_031',
         'screwdriver_poisson_021',
         'lime_poisson_001',
         'watering_can_poisson_000',
         'guitar_poisson_001',
         'boltcutter_poisson_000',
         'trash_can_poisson_041',
         'toilet_paper_poisson_000',
         'wrench_poisson_021',
         'box_poisson_018',
         'cellphone_poisson_029',
         'hammer_poisson_006',
         'can_poisson_005',
         'remote_poisson_016',
         'tape_poisson_005',
         'camera_poisson_006',
         'hammer_poisson_023',
         'book_poisson_004',
         'toy_poisson_001',
         'box_poisson_021',
         'banjo_poisson_001',
         'tape_poisson_002',
         'notebook_poisson_041',
         'mushroom_poisson_007',
         'can_poisson_019',
         'trash_can_poisson_005',
         'book_poisson_008',
         'pliers_poisson_017',
         'can_poisson_001',
         'egg_poisson_011',
         'notebook_poisson_010',
         'cellphone_poisson_031',
         'mushroom_poisson_005',
         'bowl_poisson_005',
         'soccer_ball_poisson_006',
         'light_bulb_poisson_003',
         'box_poisson_023',
         'spray_can_poisson_002',
         'trash_can_poisson_037',
         'remote_poisson_007',
         'detergent_bottle_poisson_002',
         'trash_can_poisson_047',
         'soccer_ball_poisson_003',
         'pliers_poisson_000',
         'cellphone_poisson_009',
         'tetra_pak_poisson_020',
         'trash_can_poisson_011',
         'figurine_poisson_005',
         'knife_poisson_004',
         'remote_poisson_012',
         'book_poisson_011',
         'violin_poisson_013',
         'toy_poisson_019',
         'jar_poisson_002',
         'trash_can_poisson_035',
         'pitcher_poisson_003',
         'notebook_poisson_043',
         'can_poisson_010',
         'kettle_poisson_000',
         'stapler_poisson_002',
         'jar_poisson_008',
         'light_bulb_poisson_007',
         'remote_poisson_004',
         'toy_poisson_011',
         'wrench_poisson_010',
         'knife_poisson_030',
         'cellphone_poisson_013',
         'horseshoe_poisson_000',
         'banana_poisson_004',
         'remote_poisson_013',
         'tetra_pak_poisson_024',
         'flashlight_poisson_014',
         'can_poisson_014',
         'jar_poisson_007',
         'kettle_poisson_006',
         'stapler_poisson_023',
         'toilet_paper_poisson_008',
         'camera_poisson_004',
         'watering_can_poisson_003',
         'camera_poisson_015',
         'toaster_poisson_009',
         'stapler_poisson_007',
         'stapler_poisson_026',
         'toaster_poisson_006',
         'knife_poisson_032',
         'watermelon_poisson_001',
         'knife_poisson_011',
         'cellphone_poisson_040',
         'soccer_ball_poisson_005',
         'egg_poisson_007',
         'can_poisson_003',
         'donut_poisson_005',
         'cellphone_poisson_008',
         'book_poisson_003',
         'violin_poisson_016',
         'wrench_poisson_002',
         'book_poisson_015',
         'block_of_wood_6in',
         'cheerios_14oz',
         'melissa_doug_farm_fresh_fruit_banana',
         'play_go_rainbow_stakin_cups_2_orange']
        
	return args
