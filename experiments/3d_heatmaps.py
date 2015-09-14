from keras.datasets import point_cloud_hdf5_dataset
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils

import numpy as np
import visualization.visualize as viz

LOSS_FILE = 'loss_3d_heatmaps.txt'
ERROR_FILE = 'error_3d_heatmaps.txt'

batch_size = 16
patch_size = 32

nb_train_batches = 8
nb_test_batches = 2
nb_classes = 32
nb_epoch = 2000

model = Sequential()
filter_size = 5
nb_filter_in = 1
nb_filter_out = 64
#32-5+1 = 28
model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(.5))

filter_size = 3
nb_filter_in = nb_filter_out
nb_filter_out = 128
#14-3+1 = 12
model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(.5))

filter_size = 6
nb_filter_in = nb_filter_out
nb_filter_out = nb_classes
#6-6+1
model.add(Convolution3D(nb_filter=nb_filter_out, stack_size=nb_filter_in, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
model.add(Flatten())

#64-5+1 = 60
#30-3+1 = 28
#14-6+1 = 9


# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='cross_entropy_error', optimizer=sgd)

hdf5_filepath = '/srv/3d_conv_data/training_data/contact_and_potential_grasps-3_23_15_34-3_23_16_35.h5'
topo_view_key = 'rgbd'
y_key = 'grasp_type'

dataset = point_cloud_hdf5_dataset.PointCloud_HDF5_Dataset(topo_view_key,
                                                   y_key,
                                                   hdf5_filepath,
                                                   patch_size)




def test(model):
    patch_size = 64
    dataset = point_cloud_hdf5_dataset.PointCloud_HDF5_Dataset(topo_view_key,
                                                   y_key,
                                                   hdf5_filepath,
                                                   patch_size)


    train_iterator = dataset.iterator(batch_size=1,
                              num_batches=1,
                              mode='even_shuffled_sequential')

    batch_x,batch_y = train_iterator.next()

    # out = model._predict(x)
    #
    # out_as_b012c = out.transpose(0, 3, 4, 1, 2)
    #
    #
    # #viz.visualize_batch_x(y)
    #
    # for i in range(32):
    #     img = out_as_b012c[0,:,:,:,i]
    #     #img = img.sum(axis=3)
    #
    #     import IPython
    #     IPython.embed()
    #     assert False



    ouput_dim = 64-32
    output = np.zeros((ouput_dim+32, ouput_dim+32, ouput_dim+32, nb_classes))
    for x in range(ouput_dim):
        print x
        for y in range(ouput_dim):
            for z in range(ouput_dim):
                #print x,y,z

                input_chunk = np.zeros((1, 32, 1, 32, 32))
                input_chunk[0] = batch_x[:,z:z+32, :,x:x+32, y:y+32 ]
                #print input_chunk.shape
                out = model._predict(input_chunk)
                #print out.shape

                output[x+16, y+16, z+16, :] = out

    import IPython
    IPython.embed()


def train(model, dataset):

    with open(LOSS_FILE, "w") as loss_file:
        print("logging loss")

    with open(ERROR_FILE, "w") as error_file:
        print("logging error")

    lowest_error = 1000000

    for e in range(nb_epoch):


        train_iterator = dataset.iterator(batch_size=batch_size,
                                  num_batches=nb_train_batches,
                                  mode='even_shuffled_sequential')

        for b in range(nb_train_batches):
            X_batch, Y_batch = train_iterator.next()
            loss = model.train(X_batch, Y_batch)
            print 'loss: ' + str(loss)
            with open(LOSS_FILE, "a") as loss_file:
                loss_file.write(str(loss) + '\n')



        test_iterator = dataset.iterator(batch_size=batch_size,
                                 num_batches=nb_train_batches,
                                 mode='even_shuffled_sequential')

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
            model.save_weights("weights_current_3d_heatmaps.h5")

        if lowest_error >= average_error:
            lowest_error = average_error
            print('new lowest error ' + str(lowest_error))
            model.save_weights("weights_current_best_3d_heatmaps.h5")

if __name__=="__main__":
    #train(model, dataset)
    test(model)








