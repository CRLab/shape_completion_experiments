from keras.datasets import hdf5_reconstruction_dataset
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop
from keras.utils import np_utils, generic_utils
from datasets.graspit_models_dataset import *
import visualization.visualize as viz
import mcubes
import os

batch_size = 32
patch_size = 24

nb_train_batches = 20
nb_test_batches = 8
nb_epoch = 2000

LOSS_FILE = __file__.split('.')[0] + '_loss.txt'
ERROR_FILE = __file__.split('.')[0] + '_error.txt'
CURRENT_WEIGHT_FILE = __file__.split('.')[0] + '_current_weights.h5'
BEST_WEIGHT_FILE = __file__.split('.')[0] + '_best_weights.h5'


def train(model, train_dataset, test_dataset):
    with open(LOSS_FILE, "w") as loss_file:
        print("logging loss")

    with open(ERROR_FILE, "w") as error_file:
        print("logging error")

    lowest_error = 1000000

    for e in range(nb_epoch):

        train_iterator = train_dataset.iterator(batch_size=batch_size,
                                                num_batches=nb_test_batches,
                                                flatten_y=True)

        for b in range(nb_train_batches):
            X_batch, Y_batch = train_iterator.next()
            loss = model.train(X_batch, Y_batch)
            print 'loss: ' + str(loss)
            with open(LOSS_FILE, "a") as loss_file:
                loss_file.write(str(loss) + '\n')

        test_iterator = test_dataset.iterator(batch_size=batch_size,
                                              num_batches=nb_train_batches,
                                              flatten_y=True)

        average_error = 0
        for b in range(nb_test_batches):
            X_batch, Y_batch = test_iterator.next()
            error = model.test(X_batch, Y_batch)
            average_error += error
            print('error: ' + str(error))
            with open(ERROR_FILE, "a") as error_file:
                error_file.write(str(error) + '\n')
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
                                      num_batches=nb_test_batches,
                                      flatten_y=False)

    batch_x, batch_y = train_iterator.next()

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    pred = model._predict(batch_x)
    pred = pred.reshape(batch_size, patch_size, 1, patch_size, patch_size)

    pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    # for i in range(batch_size):
    #     v, t = mcubes.marching_cubes(pred_as_b012c[i, :, :, :, 0], 0.5)
    #     mcubes.export_mesh(v, t, results_dir + '/drill_' + str(i) + '.dae',
    #                        'drill')
    #     viz.visualize_batch_x(pred, i, str(i),
    #                           results_dir + "/pred_" + str(i))
    #     viz.visualize_batch_x(batch_x, i, str(i),
    #                           results_dir + "/input_" + str(i))
    #     viz.visualize_batch_x(batch_y, i, str(i),
    #                           results_dir + "/expected_" + str(i))
    for i in range(batch_size):
        viz.visualize_batch_x_y_overlay(batch_x, batch_y, pred, str(i))
        # viz.visualize_batch_x(pred, i, 'pred_' + str(i), )
        # viz.visualize_batch_x(batch_x, i,'batch_x_' + str(i), )
        # viz.visualize_batch_x(batch_y, i, 'batch_y_' + str(i), )

    import IPython
    IPython.embed()


def test_real_world(model, weights_filepath="weights_current.h5"):
    model.load_weights(weights_filepath)

    dataset = get_graspit_dataset()
    train_iterator = dataset.iterator(batch_size=batch_size,
                                      num_batches=nb_test_batches)

    batch_x = train_iterator.next()
    import IPython
    IPython.embed()

    results_dir = 'results_graspit'
    # if not os.path.exists(results_dir):
    #     os.mkdir(results_dir)
    #
    # pred = model._predict(batch_x)
    # pred = pred.reshape(batch_size, patch_size, 1, patch_size, patch_size)
    #
    #
    # pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)
    #
    # for i in range(batch_size):
    #     v, t = mcubes.marching_cubes(pred_as_b012c[i, :, :, :, 0], 0.5)
    #     mcubes.export_mesh(v, t, results_dir + '/drill_' + str(i) + '.dae',
    #                        'drill')
    #     viz.visualize_batch_x(pred, i, str(i),
    #                           results_dir + "/pred_" + str(i))
    #     viz.visualize_batch_x(batch_x, i, str(i),
    #                           results_dir + "/input_" + str(i))
    #     viz.visualize_batch_x(batch_y, i, str(i),
    #                           results_dir + "/expected_" + str(i))


def get_model():
    model = Sequential()

    filter_size = 5
    nb_filter_in = 1
    nb_filter_out = 96
    # 24-5+1 = 20
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in,
                            nb_row=filter_size, nb_col=filter_size,
                            nb_depth=filter_size, border_mode='valid'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(.5))
    # out 10

    filter_size = 3
    nb_filter_in = nb_filter_out
    nb_filter_out = 96
    # 10-3+1 = 8
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in,
                            nb_row=filter_size, nb_col=filter_size,
                            nb_depth=filter_size, border_mode='valid'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(.5))
    # out 4

    filter_size = 3
    nb_filter_in = nb_filter_out
    nb_filter_out = 96
    # 4-3+1 = 2
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in,
                            nb_row=filter_size, nb_col=filter_size,
                            nb_depth=filter_size, border_mode='valid'))
    model.add(Dropout(.5))
    # out 2

    dim = 2
    # model.add(Flatten(nb_filter_out*dim*dim*dim))
    model.add(Flatten())
    model.add(Dense(nb_filter_out * dim * dim * dim, 3500, init='normal'))
    model.add(Dense(3500, 4000, init='normal'))
    model.add(Dense(4000, patch_size * patch_size * patch_size, init='normal',
                    activation='sigmoid'))

    # let's train the model using SGD + momentum (how original).
    sgd = RMSprop()
    model.compile(loss='cross_entropy_error', optimizer=sgd)

    return model


def main():
    model = get_model()

    hdf5_filepath = '/srv/3d_conv_data/monitor_24x24x24_25objects.h5'
    train_dataset = hdf5_reconstruction_dataset.ReconstructionDataset(
        hdf5_filepath=hdf5_filepath)

    hdf5_filepath = '/srv/3d_conv_data/monitor_24x24x24_10objects.h5'
    test_dataset = hdf5_reconstruction_dataset.ReconstructionDataset(
        hdf5_filepath=hdf5_filepath)

    # train(model, train_dataset, test_dataset)
    test(model, test_dataset)


if __name__ == "__main__":
    main()
