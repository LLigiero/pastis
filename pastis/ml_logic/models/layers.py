from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, UpSampling2D
from keras.layers import Concatenate, LeakyReLU, MaxPool2D


# ----- CONVOLUTIONAL BLOCKS -----

def down_block(inputs=None, filters=64, kernel_size=(3, 3), padding="same", strides=1, maxpool=True):
    '''Contracting block : two 3x3 convolutions followed by a 2x2 max pooling '''
    # 1st convolution layer
    conv1 = Conv2D(filters, kernel_size, padding=padding, strides=strides)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)

    # 2nd convolution layer
    conv2 = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)

    # Max Pool layer
    max_pool = MaxPool2D((2, 2))(conv2)

    if maxpool==True:
        return conv2, max_pool
    else:
        return conv2

def up_block(inputs, skip_connection, filters=64, kernel_size=(3, 3), padding="same", strides=1):
    '''Expansive block : upsampling of the features followed by a concatenante from the contracting block
    and two 2x2 convolutions'''

    # Upsampling
    up_sampling = UpSampling2D((2, 2))(inputs)

    # Concatenate
    concat = Concatenate(axis=3)([up_sampling, skip_connection])

    # 1st convolution layer
    conv1 = Conv2D(filters, kernel_size, padding=padding, strides=strides)(concat)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)

    # 2nd convolution layer
    conv2 = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv1)
    conv2 = BatchNormalization()(conv1)
    conv2 = LeakyReLU()(conv1)

    return conv2

def bottleneck(inputs, filters, kernel_size=(3, 3), padding="same", strides=1):
    '''Bridge between down_block and up_block : two 3x3 convolutional layers.
    '''
    conv1 = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(inputs)
    conv2 = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(conv1)
    return conv2
