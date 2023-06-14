import tensorflow as keras
from keras.layers import *
from keras.models import Model
from keras import Input
from keras import layers
from keras import Sequential

loss = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.005)
initializer = keras.random_normal_initializer(0., 0.02)

# Encoder
def _encoder_optic(filters, input_shape=(128,128,10)):
    model = Sequential()
    inputs = Input(input_shape)
    x = model(inputs)
    outputs = []
    for f in filters:
        # Convolution block
        x = Conv2D(filters=f, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=f, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        outputs.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=98, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs.append(x)

    encoder = Model(inputs, outputs, name='encoder_opt')
    encoder.trainable=True

    return encoder

# Encoder
def _encoder_radar(filters, input_shape=(128,128,3)):
    model = Sequential()
    inputs = Input(input_shape)
    x = model(inputs)
    outputs = []
    for f in filters:
        # Convolution block
        x = Conv2D(filters=f, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=f, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        outputs.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=98, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs.append(x)

    encoder = Model(inputs, outputs, name='encoder_radar')
    encoder.trainable=True

    return encoder


# Decoder
def _decoder_radar(skip_connections_opt,skip_connections_radar, output):

    x = output
    for f, skip_opt,skip_radar in zip([64, 32, 32, 20], skip_connections_opt, skip_connections_radar):
        # Upsampling block
        x = Conv2DTranspose(filters=f, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Concatenate()([x, skip_opt,skip_radar])  # Concatenate the upsampled feature map with the corresponding skip connection

        x = Conv2D(filters=f, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(filters=f, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

    return x
