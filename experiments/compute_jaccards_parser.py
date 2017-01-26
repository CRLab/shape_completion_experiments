
import time
import argparse
	
#IF you want to train a new model, go ahead and modify the default 
#args in this file. Then go ahead and run it, this file will be copied into the
#results directory so it is ok to modify.

def get_args():

	RESULTS_DIR = 'results/' + time.strftime("y%y_m%m_d%d_h%H_m%M") + "/"

	parser = argparse.ArgumentParser(description='Train model for shape completion')

	parser.add_argument('--DATASET', type=str, 
		default="/home/jvarley/shape_completion/train/shape_completion_experiments/datasets/YCB_GRASP_590_Dataset_bad.yaml")
		#default="/home/jvarley/shape_completion/train/shape_completion_experiments/datasets/YCB_GRASP_590_Dataset_good.yaml")

	parser.add_argument('--BATCH_SIZE', type=int, 
		default=32)

	parser.add_argument('--PATCH_SIZE', type=int, 
		default=40)

	parser.add_argument('--NB_TEST_BATCHES', type=int, 
		default=100)

	parser.add_argument('--RESULTS_DIR', type=str, 
		default=RESULTS_DIR)

	parser.add_argument('--JACCARD_TRAINED_VIEWS', type=str, 
		default=RESULTS_DIR  + 'jaccard_err_trained_views.txt')

	parser.add_argument('--JACCARD_HOLDOUT_VIEWS', type=str, 
		default=RESULTS_DIR  + 'jaccard_err_holdout_views.txt')

	parser.add_argument('--JACCARD_HOLDOUT_MODELS', type=str, 
		default=RESULTS_DIR + 'jaccard_err_holdout_models.txt')

	parser.add_argument('--WEIGHT_FILE', type=str,
	 	default="/home/jvarley/shape_completion/train/shape_completion_experiments/experiments/results/y16_m08_d24_h18_m45/best_weights_jaccard.h5")

	parser.add_argument('--PROFILE_FILE', type=str, 
		default=RESULTS_DIR + 'profile.txt')

	parser.add_argument('--MODEL_PYTHON_MODULE', type=str,
	 	default="conv3_dense2")

	args = parser.parse_args()

	return args
