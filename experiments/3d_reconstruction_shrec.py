from keras.datasets import shrec_h5py_reconstruction_dataset
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop
from keras.utils import np_utils, generic_utils
from datasets.graspit_models_dataset import *
#from keras.layers.advanced_activations import *
from operator import mul
import visualization.visualize as viz
import mcubes
import os
batch_size = 32
patch_size = 24

nb_train_batches = 20
nb_test_batches = 8
nb_epoch = 100

H5_DATASET_FILE = None
INDICES_FILE = None
LOSS_FILE = None
ERROR_FILE = None
JACCARD_FILE = None

CURRENT_WEIGHT_FILE = None
BEST_WEIGHT_FILE = None




def numpy_jaccard_similarity(a, b):
    '''
    Returns the number of pixels of the intersection of two voxel grids divided by the number of pixels in the union.
    The inputs are expected to be numpy 5D ndarrays in BZCXY format.
    '''
    return np.mean(np.sum(a*b, axis=1) / np.sum((a+b)-a*b, axis=1))

def train(model, train_dataset, test_dataset):

    with open(LOSS_FILE, "w") as loss_file:
        print("logging loss")

    with open(ERROR_FILE, "w") as error_file:
        print("logging error")

    with open(JACCARD_FILE, "w") as jaccard_file:
        print("logging jaccard_file")

    lowest_error = 1000000

    for e in range(nb_epoch):
        print('beginning epoch: ' + str(e))

        train_iterator = train_dataset.iterator(batch_size=batch_size,
                                                num_batches=nb_test_batches)

        for b in range(nb_train_batches):
            X_batch, Y_batch = train_iterator.next()
            Y_batch = Y_batch.reshape(Y_batch.shape[0], reduce(mul, Y_batch.shape[1:]))
            loss = model.train(X_batch, Y_batch)
            print 'loss: ' + str(loss)
            with open(LOSS_FILE, "a") as loss_file:
                loss_file.write(str(loss) + '\n')


        test_iterator = test_dataset.iterator(batch_size=batch_size,
                                              num_batches=nb_train_batches)

        average_error = 0
        for b in range(nb_test_batches):
            X_batch, Y_batch = test_iterator.next()
            Y_batch = Y_batch.reshape(Y_batch.shape[0], reduce(mul, Y_batch.shape[1:]))
            error = model.test(X_batch, Y_batch)
            prediction = model._predict(X_batch)
            binarized_prediction = np.array(prediction > 0.5, dtype=int)
            jaccard_similarity = numpy_jaccard_similarity(Y_batch, binarized_prediction)
            average_error += error
            print('error: ' + str(error))
            print('jaccard_similarity: ' + str(jaccard_similarity))
            with open(ERROR_FILE, "a") as error_file:
                error_file.write(str(error) + '\n')
            with open(JACCARD_FILE, "a") as jaccard_file:
                jaccard_file.write(str(jaccard_similarity) + '\n')
        average_error /= nb_test_batches

        if e % 4 == 0:
            model.save_weights(CURRENT_WEIGHT_FILE)

        if lowest_error >= average_error:
            lowest_error = average_error
            print('new lowest error ' + str(lowest_error))
            model.save_weights(BEST_WEIGHT_FILE)


def test(model, dataset, weights_filepath=BEST_WEIGHT_FILE):

    model.load_weights(weights_filepath)

    train_iterator = dataset.iterator(batch_size=batch_size,
                                      num_batches=nb_test_batches)

    batch_x, batch_y = train_iterator.next()

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    pred = model._predict(batch_x)
    pred = pred.reshape(batch_size, patch_size, 1, patch_size, patch_size)



    #pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    # for i in range(batch_size):
    #     v, t = mcubes.marching_cubes(pred_as_b012c[i, :, :, :, 0], 0.5)
    #     mcubes.export_mesh(v, t, results_dir + '/drill_' + str(i) + '.dae', 'drill')
    #     viz.visualize_batch_x(pred, i, str(i), results_dir + "/pred_" + str(i))
    #     viz.visualize_batch_x(batch_x, i, str(i), results_dir + "/input_" + str(i))
    #     viz.visualize_batch_x(batch_y, i, str(i), results_dir + "/expected_" + str(i))
    for i in range(batch_size):
        viz.visualize_batch_x_y_overlay(batch_x, batch_y, pred, i=i,  title=str(i))
        # viz.visualize_batch_x(pred, i, 'pred_' + str(i), )
        # viz.visualize_batch_x(batch_x, i,'batch_x_' + str(i), )
        # viz.visualize_batch_x(batch_y, i, 'batch_y_' + str(i), )


    import IPython
    IPython.embed()

def get_model():
    model = Sequential()

    filter_size = 5
    nb_filter_in = 1
    nb_filter_out = 96
    #24-5+1 = 20
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(.5))
    #out 10

    filter_size = 3
    nb_filter_in = nb_filter_out
    nb_filter_out = 96
    #10-3+1 = 8
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(.5))
    #out 4

    filter_size = 3
    nb_filter_in = nb_filter_out
    nb_filter_out = 96
    #4-3+1 = 2
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
    model.add(Dropout(.5))
    #out 2

    dim = 2
    #model.add(Flatten(nb_filter_out*dim*dim*dim))
    model.add(Flatten())
    model.add(Dense(nb_filter_out*dim*dim*dim, 3500, init='normal', activation='relu'))
    model.add(Dense(3500, 4000, init='normal', activation='relu'))
    model.add(Dense(4000, patch_size*patch_size*patch_size, init='normal', activation='sigmoid'))

    # let's train the model using SGD + momentum (how original).
    sgd = RMSprop()
    model.compile(loss='cross_entropy_error', optimizer=sgd)

    return model


if __name__ == "__main__":

    for NUM_OBJECTS in [1, 5, 10, 25, 50, 100]:

        DATA_DIR = 'reconstruction_results_novel_view_shrec/' + str(NUM_OBJECTS) + '/'
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        H5_DATASET_FILE = '/srv/3d_conv_data/shrec_24x24x24_2_' + str(NUM_OBJECTS) + '_objects.h5'
        INDICES_FILE = DATA_DIR + 'shrec_recon_indices_relu_' + str(NUM_OBJECTS) + '.npy'

        LOSS_FILE = DATA_DIR + __file__.split('.')[0] + '_relu_loss.txt'
        ERROR_FILE = DATA_DIR + __file__.split('.')[0] + '_relu_error.txt'
        JACCARD_FILE = DATA_DIR + __file__.split('.')[0] + '_relu_jaccard.txt'

        CURRENT_WEIGHT_FILE = DATA_DIR + __file__.split('.')[0] + '_relu_current_weights.h5'
        BEST_WEIGHT_FILE = DATA_DIR + __file__.split('.')[0] + '_relu_best_weights.h5'

        model = get_model()

        train_dataset = shrec_h5py_reconstruction_dataset.ReconstructionDataset(hdf5_filepath=H5_DATASET_FILE,
                                                                                mode='train',
                                                                                train_indices_file=INDICES_FILE)

        test_dataset = shrec_h5py_reconstruction_dataset.ReconstructionDataset(hdf5_filepath=H5_DATASET_FILE,
                                                                               mode='test',
                                                                               train_indices_file=INDICES_FILE)

        train(model, train_dataset, test_dataset)
        #test(model, test_dataset)












