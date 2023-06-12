import os
import time
import numpy as np

from tensorflow import keras

from keras.models import Model
from keras.layers import Input, Conv2D, Dropout
from keras.activations import *
from keras import optimizers
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from pastis.params import SAVE_PATH, NUM_CLASSES
from pastis.ml_logic.models.metrics import m_iou, _iou
from pastis.ml_logic.models.unet_baseline.layers import up_block,down_block,bottleneck
from pastis.ml_logic.utils import normalize_patch_spectra



# ----- UNET BASELINE MODEL -----

# Img_width = 128
# Img_heigth = 128
# Img_channel = 10

class Unet_baseline():
    def __init__(self):
        self.init_model()
        self.compile_model()

    def init_model(self,input_shape:tuple=(128,128,10), dropout:float=0) -> Model:
        '''U-net model architecture has 3 parts :
                - down_block (contracting block),
                - bottelneck (bridge),
                - up_block(expanding block)
            return the unet-model
        https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406
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
        outputs = Conv2D(NUM_CLASSES, (1, 1), activation="softmax")(up4)

        # baseline unet
        self.model = Model(inputs, outputs, name='baseline_unet')
        self.model.summary()

        print("✅ Model initialized")

        return self.model

    # ----- COMPILE MODEL -----

    def compile_model(self, learning_rate=0.05) -> Model:
        """
        Compile the U-net model, using custom metric : Mean Intersection over Union (IoU)
        from metrics.py
        """
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        loss = CategoricalCrossentropy()
        # iou = IoU(NUM_CLASSES, list(range(0, NUM_CLASSES)), sparse_y_true=False, sparse_y_pred=False, name='IoU')
        # miou = MeanIoU(NUM_CLASSES)
        miou = m_iou(NUM_CLASSES)
        iou = _iou(NUM_CLASSES)
        metrics = ['acc', miou.mean_iou, iou.iou]
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=True)

        print("✅ Model compiled")
        print(type(self.model))

        return self.model

    # ----- TRAIN MODEL -----

    def fit_model(
            self,
            train_ds,
            epochs=2,
            batch_size=32, # TO DO check batch_size
            patience=2,
            validation_ds=None, # overrides validation_SAVE_PATH=./models_outputsplit

        ) -> tuple[Model]:
        """
        Fit the model and return a tuple (fitted_model, history)
        """

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        params_path = os.path.join(SAVE_PATH, "params", timestamp + "modcheck")
        csv_path = os.path.join(SAVE_PATH, "csv", timestamp + "_csvlog.csv")

        es = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        ## add callbacks for metrics:
        #       - tf.keras.callbacks.ModelCheckpoint
        mc= ModelCheckpoint(params_path,save_best_only=True)

        #       - tf.keras.callbacks.CSVLogger
        csvlog= CSVLogger(csv_path)


        self.history = self.model.fit(
            train_ds.batch(batch_size),
            validation_data=validation_ds.batch(batch_size),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es,mc,csvlog],
            verbose=1
        )

        print('-'*50)
        print(f"✅ Model trained with :\n\
            - Accuracy : {round(np.max(self.history.history['acc']), 4)} \n\
            - Mean IuO: {round(np.max(self.history.history['mean_iou']), 4)}"
            )
        print('-'*50)

        print(type(self.model))

        return self.history


    # ----- EVALUATE MODEL -----

    def evaluate_model(
            self,
            test_ds,
            verbose=0,
            batch_size=32,
            return_dict=True
        ) -> tuple[Model, dict]:
        """
        Evaluate trained model performance on the test dataset
        """
        self.metrics = self.model.evaluate(
            test_ds.batch(batch_size),
            verbose=0,
            # callbacks=None,
            return_dict=return_dict
        )

        print(f"✅ Model evaluated, IuO: {round(self.metrics['mean_iou'], 2)}")

        return self.metrics

    def predict_model(self,X_pred):
        """
        x=numpy array with correct shape

        Make a prediction using the latest trained model
        """
        #model = load_model()
        assert self.model is not None

        X_processed= normalize_patch_spectra(X_pred)
        X_processed= X_processed.swapaxes(1, 3).swapaxes(1, 2)
        y_pred = self.model.predict(X_processed)

        return y_pred
