
import time
import argparse
	
#IF you want to test a new model, go ahead and modify the default 
#args in this file. Then go ahead and run it, this file will be copied into the
#results directory so it is ok to modify.

def get_args():

	training_dir = 'y16_m08_d19_h12_m00'

	training_results_dir = "/home/jvarley/shape_completion/train/shape_completion_experiments/experiments/results/"
	scripts_history_dir = "/home/jvarley/mnt_curacao/post_processing/data/scripts_history/"
	test_results_dir = "/home/jvarley/mnt_curacao/post_processing/data/shape_completion_results/"
	input_data_dir = "/srv/data/shape_completion_data/"
	input_dataset = "ycb/"

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
	 	default="results.y16_m08_d19_h12_m00.reconstruction_ycb_40_5")

	args = parser.parse_args()

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


	return args
