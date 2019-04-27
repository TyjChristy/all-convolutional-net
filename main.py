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

import dnns

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e',
                    '--epochs',
                    type=int,
                    default=350,
                    help='total number of epochs')
parser.add_argument(
    '-m',
    '--model',
    type=str,
    default='c3',
    help='model architecture to train. allowed values are [a-c][0-3]')

args = parser.parse_args()

# load and prepare data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# convert labels to categorical matrices
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# augment data
datagen = preprocessing.image.ImageDataGenerator(zca_whitening=True,
                                                 width_shift_range=0.05,
                                                 height_shift_range=0.05,
                                                 horizontal_flip=True)
datagen.fit(x_train)

# create session
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)

# instantiate model
input_shape = (32, 32, 3)

# base nets
if args.model == 'a0':
    model = dnns.base_net_a(input_shape)
elif args.model == 'b0':
    model = dnns.base_net_b(input_shape)
elif args.model == 'c0':
    model = dnns.base_net_c(input_shape)

# strided cnns
elif args.model == 'a1':
    model = dnns.strided_cnn_a(input_shape)
elif args.model == 'b1':
    model = dnns.strided_cnn_b(input_shape)
elif args.model == 'c1':
    model = dnns.strided_cnn_c(input_shape)

# conv pool nets
elif args.model == 'a2':
    model = dnns.conv_pool_cnn_a(input_shape)
elif args.model == 'b2':
    model = dnns.conv_pool_cnn_b(input_shape)
elif args.model == 'c2':
    model = dnns.conv_pool_cnn_c(input_shape)

# all conv nets
elif args.model == 'a3':
    model = dnns.all_conv_net_a(input_shape)
elif args.model == 'b3':
    model = dnns.all_conv_net_b(input_shape)
else:
    model = dnns.all_conv_net_c(input_shape)

model.summary()

# compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train model
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    epochs=args.epochs,
                    validation_data=(x_test, y_test))

# save model
model.save('model.h5')
