NUM_CLASSES = 20
VOID_LABEL = 19
TIME_SERIES_LENGTH = 70
BATCH_SIZE = 8

import tensorflow as keras
from keras.layers import *
from keras.models import Model
from keras import Input
from keras import layers
from keras import Sequential

loss = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.005)
initializer = keras.random_normal_initializer(0., 0.02)

class UNetConvLSTMModel:
    def __init__(self, num_classes=20):
        self.num_classes = num_classes
        self.model = self.build_model()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Encoder
    def _encoder (self, filters, input_shape=(128,128,10)):
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
            #x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x) # apply max pooling to each time step

        x = Conv2D(filters=98, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        outputs.append(x)

        encoder = Model(inputs, outputs, name='encoder')
        encoder.trainable=True

        return encoder

    def _decoder(self, skip_connections, output):

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

    def build_model(self):
        inputs = Input(shape=(TIME_SERIES_LENGTH, 128, 128, 10))
        encoder_filters = [16, 32, 32, 64]
        # Encoder
        encoder = self._encoder(filters=encoder_filters)
        encoder_outputs = [encoder(inputs[:,i]) for i in range(TIME_SERIES_LENGTH)]
        encoder_outputs = list(zip(*encoder_outputs))
        encoder_output = keras.stack(encoder_outputs[-1], axis=1)

        skip_outputs = [layers.Concatenate()(encoder_output) for encoder_output in encoder_outputs[:-1]]
        skip_connections = []
        for filters, skip_output in zip(encoder_filters, skip_outputs):
            x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=initializer)(skip_output)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            skip_connections.append(x)


        # ConvLSTM layer
        clstm_output = ConvLSTM2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
                                  data_format='channels_last', return_state=True)(encoder_output)[-1]

        # Decoder
        decoder_output = self._decoder(skip_connections[::-1], clstm_output)
        output = Conv2D(NUM_CLASSES, kernel_size=(1, 1), activation='softmax')(decoder_output)
        model = Model(inputs=inputs, outputs=output)
        model.summary()
        return model



if __name__ == '__main__':
    num_classes = 20
    model = UNetConvLSTMModel(num_classes).model
    model.summary()
