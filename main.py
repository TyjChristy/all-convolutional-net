# import dependencies
import argparse
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.preprocessing as preprocessing
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.utils as utils

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type = int, default = 350, help = 'total number of epochs')

args = parser.parse_args()

# load and prepare data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# convert labels to categorical matrices
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# augment data
datagen = preprocessing.image.ImageDataGenerator(zca_whitening = True, width_shift_range = 0.05, height_shift_range = 0.05, horizontal_flip = True)
datagen.fit(x_train)

# define model
def allconvnet(input_shape):
    '''all convolutional neural network'''
    x_init = layers.Input(input_shape)
    x = layers.Dropout(0.2)(x_init)

    x = layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (3, 3), strides = (2, 2), activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (3, 3), strides = (2, 2), activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)

    x = layers.Conv2D(10, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(10, activation = 'softmax')(x)

    model = models.Model(inputs = x_init, outputs = x)

    return model

# create session
gpu_options = tf.GPUOptions(allow_growth = True)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
K.set_session(sess)

# instantiate model
model = allconvnet((32, 32, 3))
model.summary()

# compile model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# train model
model.fit_generator(datagen.flow(x_train, y_train, batch_size = 32), epochs = args.epochs, validation_data = (x_test, y_test))

# save model
model.save('model.h5')