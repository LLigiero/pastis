import tensorflow as keras
import time
import os
import numpy as np
from statistics import mean

from keras import layers, optimizers, Input
from keras.layers import *
from keras.models import Model
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from pastis.ml_logic.models.metrics import m_iou
from pastis.ml_logic.models.unet_radar.layers_radar import _encoder_optic, _encoder_radar, _decoder_radar

from pastis.params import *
from pastis.ml_logic.utils import rename_file



class UNetConvLSTMModel_Multimodal:
    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        self.build_model()
        self.compile_model()
        self.name='UNetConvLSTMModel8multimodal'
        # self.model = self.build_model()
        # self.model.compile()

    def build_model(self):

        inputs = Input(shape=(TIME_SERIES_LENGTH, 128, 128, 13))
        inputs_opt, inputs_radar = Lambda(lambda tt: (tt[:,:,:,:,:10], tt[:,:,:,:,10:]))(inputs)
        #inputs_opt = Input(shape=(TIME_SERIES_LENGTH, 128, 128, 10))
        initializer = keras.random_normal_initializer(0., 0.02)
        encoder_filters = [16, 32, 32, 64]

        # Encoder
        encoder_opt = _encoder_optic(filters=encoder_filters)
        encoder_outputs_opt = [encoder_opt(inputs_opt[:,i]) for i in range(TIME_SERIES_LENGTH)]
        encoder_outputs_opt = list(zip(*encoder_outputs_opt))
        encoder_output_opt = keras.stack(encoder_outputs_opt[-1], axis=1)

        skip_outputs_opt = [layers.Concatenate()(encoder_output_opt) for encoder_output_opt in encoder_outputs_opt[:-1]]
        skip_connections_opt = []
        for filters, skip_output in zip(encoder_filters, skip_outputs_opt):
            x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=initializer)(skip_output)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            skip_connections_opt.append(x)

        #inputs_radar = Input(shape=(TIME_SERIES_LENGTH, 128, 128, 3))

        # Encoder
        encoder_radar = _encoder_radar(filters=encoder_filters)
        encoder_outputs_radar = [encoder_radar(inputs_radar[:,i]) for i in range(TIME_SERIES_LENGTH)]
        encoder_outputs_radar = list(zip(*encoder_outputs_radar))
        encoder_output_radar = keras.stack(encoder_outputs_radar[-1], axis=1)

        skip_outputs_radar = [layers.Concatenate()(encoder_output_radar) for encoder_output_radar in encoder_outputs_radar[:-1]]
        skip_connections_radar = []
        for filters, skip_output in zip(encoder_filters, skip_outputs_radar):
            x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=initializer)(skip_output)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            skip_connections_radar.append(x)

        layer_concat = Concatenate()([encoder_output_opt, encoder_output_radar])

        # ConvLSTM layer
        clstm_output = ConvLSTM2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
                                  data_format='channels_last', return_state=True)(layer_concat)[-1]

        # Decoder
        decoder_output = _decoder_radar(skip_connections_opt[::-1],skip_connections_radar[::-1], clstm_output)
        output = Conv2D(NUM_CLASSES, kernel_size=(1, 1), activation='softmax')(decoder_output)
        self.model = Model(inputs=inputs, outputs=output)
        #self.model = Model(inputs=[inputs_opt,inputs_radar], outputs=output)
        #self.model.summary()

        print("✅ U-net_ConvLSTM Multimodal Model initialized")

        return self.model

 # ----- COMPILE MODEL -----

    def compile_model(self, learning_rate=0.005) -> Model:
        """
        Compile the U-net 2D + ConvLSTM Multimodal model, using custom metric :
        Mean Intersection over Union (IoU)
        from metrics.py
        """
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        loss = CategoricalCrossentropy()
        miou = m_iou(NUM_CLASSES)
        #iou = _iou(NUM_CLASSES)
        metrics = ['acc', miou.mean_iou]
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=True)

        print("✅ U-net_ConvLSTM Multimodal Model compiled")

        return self.model

   # ----- TRAIN MODEL -----

    def fit_model(
            self,
            train_ds,
            epochs=25,
            batch_size=2,
            patience=5,
            validation_ds=None,

        ) -> tuple[Model]:
        """
        Fit the model and return history
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
            validation_data= (validation_ds.batch(batch_size)),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es,mc,csvlog],
            verbose=1
        )
        #rename file with accuracy and model name
        metrics = str(round(mean(self.history.history['acc']),3))
        rename_file(params_path, metrics ,self.model.name)
        rename_file(csv_path, metrics ,self.model.name)

        print('-'*50)
        print(f"✅ Model trained with :\n\
            - Accuracy : {round(np.max(self.history.history['acc']), 4)} \n\
            - Mean IuO: {round(np.max(self.history.history['mean_iou']), 4)}"
            )
        print('-'*50)

        #print(type(self.model))

        return self.model, self.history


    # ----- EVALUATE MODEL -----

    def evaluate_model(
            self,
            test_ds,
            verbose=0,
            batch_size=2,
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
        print(f"✅ Model evaluated with :\n\
            - Accuracy : {round(self.metrics['acc'],2)} \n\
            - Mean IuO: {round(self.metrics['mean_iou'],2)}"
            )

        return self.metrics
