import tensorflow as keras
from keras.layers import *
from keras.models import Model
from keras import Input
from keras import layers
from keras import Sequential

loss = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.005)
initializer = keras.random_normal_initializer(0., 0.02)
TIME_SERIES_LENGTH = 70

# Encoder
def _encoder(filters, input_shape=(128,128,10)):
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

    encoder = Model(inputs, outputs, name='encoder')
    encoder.trainable=True

    return encoder


# Decoder
def _decoder(skip_connections, output):

    x = output
    for f, skip in zip([64, 32, 32, 20], skip_connections):
        # Upsampling block
        x = Conv2DTranspose(filters=f, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Concatenate()([x, skip])  # Concatenate the upsampled feature map with the corresponding skip connection

        x = Conv2D(filters=f, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(filters=f, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

    return x
