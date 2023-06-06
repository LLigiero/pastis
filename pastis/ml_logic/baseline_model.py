import numpy as np

from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, BatchNormalization, UpSampling2D
from keras.layers import Concatenate, LeakyReLU, MaxPool2D
from keras.activations import *

from keras import optimizers
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping
from keras.metrics import IoU, MeanIoU

from pastis.ml_logic.metrics import m_iou

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
    concat = Concatenate(axis=3)([up_sampling, skip_connection]) # TODO check l'axis

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
    Return
    '''
    conv1 = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(inputs)
    conv2 = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(conv1)
    return conv2

# ----- UNET BASELINE MODEL -----

# Img_width = 128
# Img_heigth = 128
# Img_channel = 10

def baseline_unet_model(input_shape:tuple=(128,128,10), dropout:float=0) -> Model:
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

    print("✅ Model initialized")

    return model

# ----- COMPILE MODEL -----

def compile_model(model: Model, learning_rate=0.05) -> Model:
    """
    Compile the U-net model
    """
    NUM_CLASSES = 20 # à déplacer dans .env

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    loss = CategoricalCrossentropy()
    # iou = IoU(NUM_CLASSES, list(range(0, NUM_CLASSES)), sparse_y_true=False, sparse_y_pred=False, name='IoU')
    # miou = MeanIoU(NUM_CLASSES)
    miou = m_iou(NUM_CLASSES)
    metrics = ['acc', miou.mean_iou]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=True)

    print("✅ Model compiled")

    return model

# ----- TRAIN MODEL -----

def train_model(
        model: Model,
        train_ds,
        batch_size=256,
        patience=2,
        validation_ds=None, # overrides validation_split
    ) -> tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    if validation_ds :
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=1,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )
    else :
        history = model.fit(
            train_ds,
            epochs=1,
            batch_size=batch_size,
            verbose=1
        )
    print(f"✅ Model trained with IuO: {history.history()}")

    #print(f"✅ Model trained with IuO: {round(np.min(history.history['IuO']), 2)}")

    return model, history


# ----- EVALUATE MODEL -----

def evaluate_model(
        model: Model,
        test_ds,
        batch_size=64
    ) -> tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """
    metrics = model.evaluate(
        test_ds,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    print(f"✅ Model evaluated, IuO: {round(metrics['IuO'], 2)}")

    return metrics
