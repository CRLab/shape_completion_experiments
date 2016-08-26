
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam

def get_model(patch_size):

    model = Sequential()

    
    filter_size = 4
    nb_filter_in = 1
    nb_filter_out = 64

    # input: 1 cube of side length 40
    # output: 64 cubes of side length 40-4+1 = 37
    model.add(Convolution3D(nb_filter=nb_filter_out,
                            stack_size=nb_filter_in,
                            nb_row=filter_size,
                            nb_col=filter_size,
                            nb_depth=filter_size,
                            border_mode='valid',
                            activation='relu', init='he_normal'))
    # output: 64 cubes of side length 36/2 = 18
    #model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(.5))

    filter_size = 4
    nb_filter_in = nb_filter_out
    nb_filter_out = 64

    # output: 64 cubes of side length 37-4+1 = 34
    model.add(Convolution3D(nb_filter=nb_filter_out,
                            stack_size=nb_filter_in,
                            nb_row=filter_size,
                            nb_col=filter_size,
                            nb_depth=filter_size,
                            border_mode='valid',
                            activation='relu', init='he_normal'))

    # output: 64 cubes of size length 15/2 = 7
    #model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # During training: drop (set to zero) each of the current outputs with a 0.5
    # probability.
    # During testing: multiply each of the current outputs by that probability.
    #model.add(Dropout(.5))

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    filter_size = 4
    nb_filter_in = nb_filter_out
    nb_filter_out = 64

    # output: 64 cubes of side length 17-4+1 = 14
    model.add(Convolution3D(nb_filter=nb_filter_out,
                            stack_size=nb_filter_in,
                            nb_row=filter_size,
                            nb_col=filter_size,
                            nb_depth=filter_size,
                            border_mode='valid',
                            activation='relu', init='he_normal'))
    #model.add(Dropout(.5))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    """
    filter_size = 4
    nb_filter_in = nb_filter_out
    nb_filter_out = 64

    # output: 64 cubes of side length 16-4+1 = 13
    model.add(Convolution3D(nb_filter=nb_filter_out,
                            stack_size=nb_filter_in,
                            nb_row=filter_size,
                            nb_col=filter_size,
                            nb_depth=filter_size,
                            border_mode='valid',
                            activation='relu', init='he_normal'))
    """

    # During training: drop (set to zero) each of the current outputs with a 0.5
    # probability.
    # During testing: multiply each of the current outputs by that probability.
    #model.add(Dropout(.5))
    """
    filter_size = 5
    nb_filter_in = nb_filter_out
    nb_filter_out = 50

    # output: 64 cubes of side length 10-4+1 = 7
        model.add(Convolution3D(nb_filter=nb_filter_out,
                            stack_size=nb_filter_in,
                            nb_row=filter_size,
                            nb_col=filter_size,
                            nb_depth=filter_size,
                            border_mode='valid',
                            activation='relu'))
    """
    dim = 7

    # output: a vector of size 64*5*5*5 = 8000
    model.add(Flatten())
    model.add(Dense(nb_filter_out * dim * dim * dim, 5000, init='he_normal', activation='relu'))

    #model.add(Dense(3000, 3000, init='he_normal', activation='relu'))


    #model.add(Dense(3000, 5000, init='he_normal', activation='relu'))

    model.add(Dense(5000, patch_size * patch_size * patch_size, init='glorot_normal',
                    activation='sigmoid'))

    optimizer = Adam(lr=.0001)
    model.compile(loss='cross_entropy_error', optimizer=optimizer)

    return model