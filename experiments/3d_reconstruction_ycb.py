from datasets import ycb_reconstruction_dataset
from datasets import hdf5_reconstruction_dataset
from datasets import ycb_hdf5_reconstruction_dataset
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop
from keras.utils import np_utils, generic_utils

import visualization.visualize as viz
import mcubes
import os

import cProfile, pstats, StringIO
pr = cProfile.Profile()

batch_size = 16
patch_size = 30

nb_train_batches = 100
nb_test_batches = 10
#nb_epoch = 2000
nb_epoch = 500

LOSS_FILE = __file__.split('.')[0] + '_loss.txt'
ERROR_FILE = __file__.split('.')[0] + '_error.txt'
CURRENT_WEIGHT_FILE = __file__.split('.')[0] + '_current_weights.h5'
BEST_WEIGHT_FILE = __file__.split('.')[0] + '_best_weights.h5'

def train(model, dataset):

    with open(LOSS_FILE, "w") as loss_file:
        print("logging loss")

    with open(ERROR_FILE, "w") as error_file:
        print("logging error")

    lowest_error = 1000000

    for e in range(nb_epoch):
        pr.enable()
        train_iterator = dataset.iterator(batch_size=batch_size,
                                          num_batches=nb_test_batches,
                                          flatten_y=True)

        for b in range(nb_train_batches):
            X_batch, Y_batch = train_iterator.next(train=1)
            loss = model.train(X_batch, Y_batch)
            print 'loss: ' + str(loss)
            with open(LOSS_FILE, "a") as loss_file:
                loss_file.write(str(loss) + '\n')

        test_iterator = dataset.iterator(batch_size=batch_size,
                                         num_batches=nb_train_batches,
                                         flatten_y=True)

        average_error = 0
        for b in range(nb_test_batches):
            X_batch, Y_batch = test_iterator.next(train=0)
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

        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        stats_str = s.getvalue()
        f = open("profile.txt", 'w')
        f.write(stats_str)
        f.close()

def test(model, dataset, weights_filepath="weights_current.h5"):

    model.load_weights(weights_filepath)

    train_iterator = dataset.iterator(batch_size=batch_size,
                                          num_batches=nb_test_batches,
                                          flatten_y=False)

    batch_x, batch_y = train_iterator.next(train=0)

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    pred = model._predict(batch_x)
    pred = pred.reshape(batch_size, patch_size, 1, patch_size, patch_size)


    pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

    for i in range(batch_size):
        v, t = mcubes.marching_cubes(pred_as_b012c[i, :, :, :, 0], 0.5)
        mcubes.export_mesh(v, t, results_dir + '/drill_' + str(i) + '.dae', 'drill')
        viz.visualize_batch_x(pred, i, str(i), results_dir + "/pred_" + str(i))
        viz.visualize_batch_x(batch_x, i, str(i), results_dir + "/input_" + str(i))
        viz.visualize_batch_x(batch_y, i, str(i), results_dir + "/expected_" + str(i))

def get_model():
    model = Sequential()

    filter_size = 5
    nb_filter_in = 1
    nb_filter_out = 64
    #30-5+1 = 26
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(.5))
    #out 13

    filter_size = 4
    nb_filter_in = nb_filter_out
    nb_filter_out = 64
    #13-4+1 = 10
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(.5))
    #out 5

    filter_size = 3
    nb_filter_in = nb_filter_out
    nb_filter_out = 64
    #5-3+1 = 3
    model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
    model.add(Dropout(.5))
    #out 3

    dim = 3
    #model.add(Flatten(nb_filter_out*dim*dim*dim))
    model.add(Flatten())
    model.add(Dense(nb_filter_out*dim*dim*dim, 3000, init='normal'))
    model.add(Dense(3000, 4000, init='normal'))
    model.add(Dense(4000, patch_size*patch_size*patch_size, init='normal', activation='sigmoid'))

    # let's train the model using SGD + momentum (how original).
    sgd = RMSprop()
    model.compile(loss='cross_entropy_error', optimizer=sgd)

    return model

def get_dataset():

    #dataset = ycb_reconstruction_dataset.YcbDataset("/srv/data/shape_completion_data/ycb/", "rubbermaid_ice_guard_pitcher_blue", patch_size)
    #dataset = hdf5_reconstruction_dataset.ReconstructionDataset('/srv/data/shape_completion_data/ycb/wescott_orange_grey_scissors/h5/wescott_orange_grey_scissors.h5')
    model_names = ['black_and_decker_lithium_drill_driver', 'block_of_wood_6in', 'blue_wood_block_1inx1in',
                   'brine_mini_soccer_ball', 'campbells_condensed_tomato_soup', 'cheerios_14oz',
                    'clorox_disinfecting_wipes_35', 'comet_lemon_fresh_bleach', 'domino_sugar_1lb',
                   'frenchs_classic_yellow_mustard_14oz', 'melissa_doug_farm_fresh_fruit_banana',
                   'melissa_doug_farm_fresh_fruit_lemon', 'morton_salt_shaker', 'play_go_rainbow_stakin_cups_1_yellow',
                   'play_go_rainbow_stakin_cups_2_orange', 'pringles_original', 'red_metal_cup_white_speckles',
                   'rubbermaid_ice_guard_pitcher_blue', 'soft_scrub_2lb_4oz', 'sponge_with_textured_cover']
    model_names = ['black_and_decker_lithium_drill_driver', 'block_of_wood_6in', 'blue_wood_block_1inx1in',
                   'brine_mini_soccer_ball', 'campbells_condensed_tomato_soup', 'cheerios_14oz',
                    'clorox_disinfecting_wipes_35', 'comet_lemon_fresh_bleach', 'domino_sugar_1lb',
                   'frenchs_classic_yellow_mustard_14oz', 'melissa_doug_farm_fresh_fruit_banana',
                   'melissa_doug_farm_fresh_fruit_lemon', 'morton_salt_shaker', 'play_go_rainbow_stakin_cups_1_yellow',
                   'play_go_rainbow_stakin_cups_2_orange', 'pringles_original',
                   'rubbermaid_ice_guard_pitcher_blue', 'soft_scrub_2lb_4oz', 'sponge_with_textured_cover']
    dataset = ycb_hdf5_reconstruction_dataset.YcbReconstructionDataset('/srv/data/shape_completion_data/ycb/', model_names)
    return dataset

if __name__ == "__main__":
    model = get_model()
    dataset = get_dataset()
    train(model, dataset)
    test(model, dataset, BEST_WEIGHT_FILE)

    print('Script Completed')
    import IPython
    IPython.embed()
    assert False
