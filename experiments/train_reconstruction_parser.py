
import time
import argparse
	
#IF you want to train a new model, go ahead and modify the default 
#args in this file. Then go ahead and run it, this file will be copied into the
#results directory so it is ok to modify.

def get_args():

	RESULTS_DIR = 'results/' + time.strftime("y%y_m%m_d%d_h%H_m%M") + "/"

	parser = argparse.ArgumentParser(description='Train model for shape completion')

	parser.add_argument('--BATCH_SIZE', type=int, 
		default=32)

	parser.add_argument('--PATCH_SIZE', type=int, 
		default=40)

	parser.add_argument('--NB_TRAIN_BATCHES', type=int, 
		default=100)

	parser.add_argument('--NB_TEST_BATCHES', type=int, 
		default=10)

	parser.add_argument('--NB_EPOCH', type=int,
	 	default=300)

	parser.add_argument('--RESULTS_DIR', type=str, 
		default=RESULTS_DIR)

	parser.add_argument('--TEST_OUTPUT_DIR', type=str, 
		default=RESULTS_DIR  + "test_output/")

	parser.add_argument('--LOSS_FILE', type=str, 
		default=RESULTS_DIR + 'loss.txt')

	parser.add_argument('--ERROR_TRAINED_VIEWS', type=str, 
		default=RESULTS_DIR + 'cross_entropy_err_trained_views.txt')

	parser.add_argument('--ERROR_HOLDOUT_VIEWS', type=str, 
		default=RESULTS_DIR + 'cross_entropy_err_holdout_views.txt')

	parser.add_argument('--ERROR_HOLDOUT_MODELS', type=str, 
		default=RESULTS_DIR  + 'cross_entropy_holdout_models.txt')

	parser.add_argument('--JACCARD_TRAINED_VIEWS', type=str, 
		default=RESULTS_DIR  + 'jaccard_err_trained_views.txt')

	parser.add_argument('--JACCARD_HOLDOUT_VIEWS', type=str, 
		default=RESULTS_DIR  + 'jaccard_err_holdout_views.txt')

	parser.add_argument('--JACCARD_HOLDOUT_MODELS', type=str, 
		default=RESULTS_DIR + 'jaccard_err_holdout_models.txt')

	parser.add_argument('--CURRENT_WEIGHT_FILE', type=str,
	 	default=RESULTS_DIR  + 'current_weights.h5')

	parser.add_argument('--BEST_WEIGHT_FILE', type=str,
	 	default=RESULTS_DIR + 'best_weights.h5')

	parser.add_argument('--BEST_WEIGHT_FILE_JACCARD', type=str,
	 	default=RESULTS_DIR  + 'best_weights_jaccard.h5')

	parser.add_argument('--PROFILE_FILE', type=str, 
		default=RESULTS_DIR + 'profile.txt')

	parser.add_argument('--MODEL_PYTHON_MODULE', type=str,
	 	default="conv3_dense2")

	parser.add_argument('--DATASET_FILEPATH', type=str,
	 	#default="/home/jvarley/shape_completion/train/shape_completion_experiments/datasets/YCB_Dataset.yaml")
                #default="/home/jvarley/shape_completion/train/shape_completion_experiments/datasets/YCB_GRASP_25_Dataset.yaml")
                #default="/home/jvarley/shape_completion/train/shape_completion_experiments/datasets/YCB_GRASP_100_Dataset.yaml")
				default="/home/jvarley/shape_completion/train/shape_completion_experiments/datasets/YCB_GRASP_590_Dataset.yaml")

	args = parser.parse_args()

	return args
