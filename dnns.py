# import dependencies
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.regularizers as regularizers

# define models
def base_net_a(input_shape):
    '''base model A'''
    x_init = layers.Input(input_shape)
    x = layers.Dropout(0.2)(x_init)

    x = layers.Conv2D(96, (5, 5), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (5, 5), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), pading = 'same')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)

    x = layers.Conv2D(10, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Activation('softmax')(x)

    model = models.Model(inputs = x_init, outputs = x)

    return model

def strided_cnn_a(input_shape):
    '''strided convolutional net A'''
    x_init = layers.Input(input_shape)
    x = layers.Dropout(0.2)(x_init)

    x = layers.Conv2D(96, (5, 5), strides = (2, 2), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (5, 5), strides = (2, 2), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)

    x = layers.Conv2D(10, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Activation('softmax')(x)

    model = models.Model(inputs = x_init, outputs = x)

    return model

def conv_pool_cnn_a(input_shape):
    '''convolutional net A with pooling'''
    x_init = layers.Input(input_shape)
    x = layers.Dropout(0.2)(x_init)

    x = layers.Conv2D(96, (5, 5), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (5, 5), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), pading = 'same')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)

    x = layers.Conv2D(10, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Activation('softmax')(x)

    model = models.Model(inputs = x_init, outputs = x)

    return model

def all_conv_net_a(input_shape):
    '''all convolutional net A'''
    x_init = layers.Input(input_shape)
    x = layers.Dropout(0.2)(x_init)

    x = layers.Conv2D(96, (5, 5), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (5, 5), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)

    x = layers.Conv2D(10, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Activation('softmax')(x)

    model = models.Model(inputs = x_init, outputs = x)

    return model

def base_net_b(input_shape):
    '''base model B'''
    x_init = layers.Input(input_shape)
    x = layers.Dropout(0.2)(x_init)

    x = layers.Conv2D(96, (5, 5), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (5, 5), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), pading = 'same')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)

    x = layers.Conv2D(10, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Activation('softmax')(x)

    model = models.Model(inputs = x_init, outputs = x)

    return model

def strided_cnn_b(input_shape):
    '''strided convolutional net B'''
    x_init = layers.Input(input_shape)
    x = layers.Dropout(0.2)(x_init)

    x = layers.Conv2D(96, (5, 5), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (1, 1), strides = (2, 2), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (5, 5), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), strides = (2, 2), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)

    x = layers.Conv2D(10, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Activation('softmax')(x)

    model = models.Model(inputs = x_init, outputs = x)

    return model

def conv_pool_cnn_b(input_shape):
    '''convolutional net B with pooling'''
    x_init = layers.Input(input_shape)
    x = layers.Dropout(0.2)(x_init)

    x = layers.Conv2D(96, (5, 5), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (5, 5), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), pading = 'same')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)

    x = layers.Conv2D(10, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Activation('softmax')(x)

    model = models.Model(inputs = x_init, outputs = x)

    return model

def all_conv_net_b(input_shape):
    '''all convolutional net B'''
    x_init = layers.Input(input_shape)
    x = layers.Dropout(0.2)(x_init)

    x = layers.Conv2D(96, (5, 5), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (5, 5), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)

    x = layers.Conv2D(10, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Activation('softmax')(x)

    model = models.Model(inputs = x_init, outputs = x)

    return model

def base_net_c(input_shape):
    '''base model C'''
    x_init = layers.Input(input_shape)
    x = layers.Dropout(0.2)(x_init)

    x = layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)

    x = layers.Conv2D(10, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Activation('softmax')(x)

    model = models.Model(inputs = x_init, outputs = x)

    return model

def strided_cnn_c(input_shape):
    '''strided convolutional net C'''
    x_init = layers.Input(input_shape)
    x = layers.Dropout(0.2)(x_init)

    x = layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)

    x = layers.Conv2D(10, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Activation('softmax')(x)

    model = models.Model(inputs = x_init, outputs = x)

    return model

def conv_pool_cnn_c(input_shape):
    '''convolutional net C with pooling'''
    x_init = layers.Input(input_shape)
    x = layers.Dropout(0.2)(x_init)

    x = layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)

    x = layers.Conv2D(10, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Activation('softmax')(x)

    model = models.Model(inputs = x_init, outputs = x)

    return model

def all_conv_net_c(input_shape):
    '''all convolutional net C'''
    x_init = layers.Input(input_shape)
    x = layers.Dropout(0.2)(x_init)

    x = layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(96, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(192, (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.Conv2D(192, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)

    x = layers.Conv2D(10, (1, 1), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.001))(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Activation('softmax')(x)

    model = models.Model(inputs = x_init, outputs = x)

    return model