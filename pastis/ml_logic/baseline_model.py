
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Concatenante, LeakyRelu, Activation, MaxPool2D
from tensorflow.keras.activations import *

# ----- CONVOLUTIONAL BLOCKS -----

def down_block(inputs=None, filters=64, kernel_size=(3, 3), padding="same", strides=1, maxpool=True):
    '''Contracting block : two 3x3 convolutions followed by a 2x2 max pooling '''
    # 1st convolution layer
    conv1 = Conv2D(filters, kernel_size, padding=padding, strides=strides)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyRelu()(conv1)

    # 2nd convolution layer
    conv2 = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyRelu()(conv2)

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
    concat = Concatenate()(axis=3, [up_sampling, skip_connection])

    # 1st convolution layer
    conv1 = Conv2D(filters, kernel_size, padding=padding, strides=strides)(concat)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyRelu()(conv1)

    # 2nd convolution layer
    conv2 = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv1)
    conv2 = BatchNormalization()(conv1)
    conv2 = LeakyRelu()(conv1)

    return conv2

def bottleneck(inputs, filters, kernel_size=(3, 3), padding="same", strides=1):
    '''Bridge between down_block and up_block : two 3x3 convolutional layers.
    Return
    '''
    conv1 = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(inputs)
    conv2 = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(conv1)
    return conv2

# ----- UNET BASELINE MODEL -----

# Img_width = 128
# Img_heigth = 128
# Img_channel = 10

def baseline_unet_model(input_shape=(128,128,10), dropout=0):
    '''U-net model architecture has 3 parts :
            - down_block (contracting block),
            - bottelneck (bridge),
            - up_block(expanding block)
        return the unet-model
    '''
    filters = [64,128,256,512,1024]
    # filters = [16, 32, 32, 64, 64]

    inputs = Input(input_shape)

    # encoder
    conv1, pool1 = down_block(inputs, filters[0]) #128 -> 64
    conv2, pool2 = down_block(pool1, filters[1]) #64 -> 32
    conv3, pool3 = down_block(pool2, filters[2]) #32 -> 16
    conv4, pool4 = down_block(pool3, filters[3]) #16->8

    # bottleneck
    bn = bottleneck(pool4, filters[4])

    # decoder
    up1 = up_block(bn, conv4, filters[3]) #8 -> 16
    up2 = up_block(up1, conv3,  filters[2]) #16 -> 32
    up3 = up_block(up2, conv2,  filters[1]) #32 -> 64
    up4 = up_block(up3, conv1,  filters[0]) #64 -> 128

    # Dropoutlayer
    if dropout >0:
        up4 = Dropout(dropout)(up4)

    # Final layer : 1x1 convolution with softmax activation
    outputs = Conv2D(20, (1, 1), activation="softmax")(up4)

    # baseline unet
    model = Model(inputs, outputs, name='baseline_unet')
    model.summary()

    return model
