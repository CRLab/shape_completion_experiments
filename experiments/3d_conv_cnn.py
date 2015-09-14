from keras.datasets import geometric_3d_dataset
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils


batch_size = 4
patch_size = 32

filter_size = 5
nb_filter = 8

nb_train_batches = 100
nb_test_batches = 20
nb_classes = 3
nb_epoch = 200

model = Sequential()
model.add(Convolution3D(nb_filter=nb_filter, stack_size=1, nb_row=filter_size, nb_col=filter_size, nb_depth=filter_size, border_mode='valid'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten(nb_filter*14*14*14))
model.add(Dense(nb_filter*14*14*14, nb_classes, init='normal'))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

dataset = geometric_3d_dataset.Geometric3DDataset(patch_size=patch_size,
                                                  task=geometric_3d_dataset.Geometric3DDataset.CLASSIFICATION_TASK,
                                                  centered=True)

for e in range(nb_epoch):

    train_iterator = dataset.iterator(batch_size=batch_size,
                                      num_batches=nb_test_batches)

    for b in range(nb_train_batches):
        X_batch, Y_batch = train_iterator.next()
        loss = model.train(X_batch, Y_batch)
        print 'loss: ' + str(loss)

    test_iterator = dataset.iterator(batch_size=batch_size,
                                     num_batches=nb_train_batches)

    for b in range(nb_test_batches):
        X_batch, Y_batch = test_iterator.next()
        error = model.test(X_batch, Y_batch)
        print 'error: ' + str(error)








