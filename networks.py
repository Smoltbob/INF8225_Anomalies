"""
This file contains all the unsupervised models for outliers detection.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from functools import reduce
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn.utils.testing import assert_array_almost_equal
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
import keras.backend as kb
import matplotlib.pyplot as plt
import os

best_nn_setting = (128, 64, 32, 16, 8)
repeat_number = 2 # 3 # 20  # 50
input_roi_size = 64


# learning rate schedule
def step_decay(epoch):
    if epoch < 100:
        l_rate = 1e-3
    else:
        l_rate = 1e-4
    return l_rate


def pixel_wise_mse(target, output):
    """
    Computes the pixel-wise mean square error loss function.
    :param target:
    :param output:
    :return:
    """
    eps = kb.epsilon()
    output = kb.clip(output, eps, 1. - eps)
    return kb.sum(kb.pow(target - output, 2), axis=-1) / (2 * input_roi_size)


class BuildOurMethodV1:
    """
    First version of our method: 3 CAE + DAE.
    The 3 CAEs are implemented based on the description given in the following article:
    - "Object-centric Auto-encoders and Dummy Anomalies for Abnormal Event Detection in Video"
    Implementation inspired by the following:
    - https://blog.keras.io/building-autoencoders-in-keras.html;
    - https://github.com/civisanalytics/muffnn/blob/master/muffnn/autoencoder/autoencoder.py
    """
    def __init__(self,
                 hidden_units=best_nn_setting,
                 batch_size=64,
                 n_epochs=200,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 loss='mean_squared_error',
                 metrics='accuracy',
                 validation_split=0.20,
                 model_dir_path='',
                 iteration_number=0):
        """
        Builds and compiles an auto-encoder.
        :param hidden_units:
        :param batch_size:
        :param n_epochs:
        :param hidden_activation:
        :param output_activation:
        :param loss:
        :param metrics:
        :param validation_split:
        :param model_dir_path:
        :param iteration_number:
        """
        self.hidden_units = hidden_units
        self.dae_batch_size = batch_size
        self.cae_batch_size = batch_size
        self.dae_n_epochs = n_epochs
        self.cae_n_epochs = n_epochs
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer_cae = Adam(lr=0.0)
        self.optimizer_dae = Adam(lr=0.0)
        self.loss = loss
        self.metrics = metrics
        self.validation_split = validation_split
        self.dae_latent_dim = hidden_units[-1]
        self.model_dir_path = model_dir_path
        self.iteration_number = iteration_number
        self.input_shape = (input_roi_size, input_roi_size, 1)
        self.cae_latent_shape = (8, 8, 16)
        self.dae_input_size = 3 * reduce(lambda x, y: x*y, self.cae_latent_shape)
        self.cae_appearances_global_mse = None
        self.cae_prev_motion_global_mse = None
        self.cae_next_motion_global_mse = None
        self.cae_appearances_accuracy = None
        self.cae_prev_motion_accuracy = None
        self.cae_next_motion_accuracy = None
        self.mse_per_sample = None
        self.global_mse = None
        self.accuracy = None

        self.model_cae_appearances_name = 'model_cae_appearances_' + str(self.iteration_number)
        self.model_cae_prev_motion_name = 'model_cae_prev_motion_' + str(self.iteration_number)
        self.model_cae_next_motion_name = 'model_cae_next_motion_' + str(self.iteration_number)
        self.model_dae_cfv_name = 'model_dae_cfv_' + str(self.iteration_number)

        # Build encoder and decoder
        self.cae_appearances_encoder = self.build_cae_encoder()
        self.cae_appearances_decoder = self.build_cae_decoder()
        self.cae_prev_motion_encoder = self.build_cae_encoder()
        self.cae_prev_motion_decoder = self.build_cae_decoder()
        self.cae_next_motion_encoder = self.build_cae_encoder()
        self.cae_next_motion_decoder = self.build_cae_decoder()
        self.dae_cfv_encoder = self.build_dae_encoder()
        self.dae_cfv_decoder = self.build_dae_decoder()

        # Build and compile the auto-encoders separately
        input_data = Input(shape=self.input_shape)

        self.cae_appearances = Model(input_data, self.cae_appearances_decoder(self.cae_appearances_encoder(input_data)),
                                     name='cae_appearances')
        self.cae_appearances.compile(optimizer=self.optimizer_cae, loss=pixel_wise_mse, metrics=[self.metrics])

        self.cae_prev_motion = Model(input_data, self.cae_prev_motion_decoder(self.cae_prev_motion_encoder(input_data)),
                                     name='cae_prev_motion')
        self.cae_prev_motion.compile(optimizer=self.optimizer_cae, loss=pixel_wise_mse, metrics=[self.metrics])

        self.cae_next_motion = Model(input_data, self.cae_next_motion_decoder(self.cae_next_motion_encoder(input_data)),
                                     name='cae_next_motion')
        self.cae_next_motion.compile(optimizer=self.optimizer_cae, loss=pixel_wise_mse, metrics=[self.metrics])

        input_cfv_data = Input(shape=tuple([self.dae_input_size]))
        r_input_cfv_data = self.dae_cfv_decoder(self.dae_cfv_encoder(input_cfv_data))
        self.dae_cfv = Model(input_cfv_data, r_input_cfv_data, name='dae_cfv')
        self.dae_cfv.compile(optimizer=self.optimizer_dae, loss=self.loss, metrics=[self.metrics])

        # learning schedule callback
        lr_scheduler = LearningRateScheduler(step_decay)
        self.callbacks_list = [lr_scheduler]

        self.cae_appearances_encoder.summary()
        self.cae_appearances_decoder.summary()
        self.dae_cfv_encoder.summary()
        self.dae_cfv_decoder.summary()

    def build_cae_encoder(self):
        """
        Builds the encoder part of the CAE network.
        :return: encoder model
        """
        _input = Input(shape=self.input_shape)    # adapt this if using 'channels_first' image data format

        _x = Conv2D(32, (3, 3), activation=self.hidden_activation, padding='same')(_input)
        _x = MaxPooling2D((2, 2), padding='same')(_x)
        _x = Conv2D(32, (3, 3), activation=self.hidden_activation, padding='same')(_x)
        _x = MaxPooling2D((2, 2), padding='same')(_x)
        _x = Conv2D(16, (3, 3), activation=self.hidden_activation, padding='same')(_x)
        _encoded = MaxPooling2D((2, 2), padding='same')(_x)

        return Model(_input, _encoded, name='cae_encoder')

    def build_cae_decoder(self):
        """
        Builds the decoder part of the CAE network.
        :return: decoder model
        """
        _latent_input = Input(shape=self.cae_latent_shape)

        _x = Conv2D(16, (3, 3), activation=self.hidden_activation, padding='same')(_latent_input)
        _x = UpSampling2D((2, 2))(_x)
        _x = Conv2D(32, (3, 3), activation=self.hidden_activation, padding='same')(_x)
        _x = UpSampling2D((2, 2))(_x)
        _x = Conv2D(32, (3, 3), activation=self.hidden_activation, padding='same')(_x)
        _x = UpSampling2D((2, 2))(_x)
        _decoded = Conv2D(1, (3, 3), activation=self.output_activation, padding='same')(_x)

        return Model(_latent_input, _decoded, name='cae_decoder')

    def build_dae_encoder(self):
        # Encoder
        _input = Input(shape=tuple([self.dae_input_size]))
        _encoded = _input

        for units in self.hidden_units:
            _encoded = Dense(units, activation=self.hidden_activation)(_encoded)

        return Model(_input, _encoded, name='dae_encoder')

    def build_dae_decoder(self):
        # Decoder
        _latent_input = Input(shape=(self.dae_latent_dim,))
        _x = _latent_input

        for units in reversed(self.hidden_units[:-1]):
            _x = Dense(units, activation=self.hidden_activation)(_x)

        _output = Dense(self.dae_input_size, activation=self.output_activation)(_x)

        return Model(_latent_input, _output, name='dae_decoder')

    def train(self,
              train_roi_image_data,
              train_prev_diff_data,
              train_next_diff_data,
              save_model=True,
              test_saved_model=True,
              plot_histories=True,
              train_cfv_ae_only=False):
        """
        Fit the dae model.
        :param train_roi_image_data:
        :param train_prev_diff_data:
        :param train_next_diff_data:
        :param save_model:
        :param test_saved_model:
        :param plot_histories:
        :param train_cfv_ae_only:
        :return:
        """
        cae_appearances_history = None
        cae_prev_motion_history = None
        cae_next_motion_history = None
        # Reshape input images if necessary
        if train_roi_image_data.shape[-1] != 1:
            train_roi_image_data = train_roi_image_data.reshape((*train_roi_image_data.shape, 1))
        if train_prev_diff_data.shape[-1] != 1:
            train_prev_diff_data = train_prev_diff_data.reshape((*train_prev_diff_data.shape, 1))
        if train_next_diff_data.shape[-1] != 1:
            train_next_diff_data = train_next_diff_data.reshape((*train_next_diff_data.shape, 1))
        if not train_cfv_ae_only:
            print('=====================================================================')
            print('Training the CAE Appearances model:')
            # Fit the model
            cae_appearances_history = self.cae_appearances.fit(train_roi_image_data, train_roi_image_data,
                                                               validation_split=self.validation_split,
                                                               epochs=self.cae_n_epochs,
                                                               batch_size=self.cae_batch_size,
                                                               shuffle=True, callbacks=self.callbacks_list)
            print('                                                                done!')
            print('=====================================================================')
            print('Evaluate the CAE Appearances model:')
            predicted_appearances_data = self.cae_appearances.predict(train_roi_image_data)
            assert predicted_appearances_data.shape == train_roi_image_data.shape, \
                'The predicted data shape is not right!'
            predicted_appearances_data_2 = self.cae_appearances_decoder.predict(
                self.cae_appearances_encoder.predict(train_roi_image_data))
            assert np.all(predicted_appearances_data == predicted_appearances_data_2), \
                'Inconsistency between encoder and decoder!'
            cae_appearances_scores = self.cae_appearances.evaluate(train_roi_image_data, train_roi_image_data)
            self.cae_appearances_global_mse = cae_appearances_scores[0]
            self.cae_appearances_accuracy = cae_appearances_scores[1]
            print("%s: %.2f%%" % (self.cae_appearances.metrics_names[1], self.cae_appearances_accuracy*100))
            print('                                                                done!')
            print('=====================================================================')
            print('Training the CAE Previous Motion model:')
            # Fit the model
            cae_prev_motion_history = self.cae_prev_motion.fit(train_prev_diff_data, train_prev_diff_data,
                                                               validation_split=self.validation_split,
                                                               epochs=self.cae_n_epochs,
                                                               batch_size=self.cae_batch_size,
                                                               shuffle=True, callbacks=self.callbacks_list)
            print('                                                                done!')
            print('=====================================================================')
            print('Evaluate the CAE Previous Motion model:')
            predicted_prev_motion_data = self.cae_prev_motion.predict(train_prev_diff_data)
            assert predicted_prev_motion_data.shape == train_prev_diff_data.shape, \
                'The predicted data shape is not right!'
            predicted_prev_motion_data_2 = self.cae_prev_motion_decoder.predict(
                self.cae_prev_motion_encoder.predict(train_prev_diff_data))
            assert np.all(predicted_prev_motion_data == predicted_prev_motion_data_2), \
                'Inconsistency between encoder and decoder!'
            cae_prev_motion_scores = self.cae_prev_motion.evaluate(train_prev_diff_data, train_prev_diff_data)
            self.cae_prev_motion_global_mse = cae_prev_motion_scores[0]
            self.cae_prev_motion_accuracy = cae_prev_motion_scores[1]
            print("%s: %.2f%%" % (self.cae_prev_motion.metrics_names[1], self.cae_prev_motion_accuracy * 100))
            print('                                                                done!')
            print('=====================================================================')
            print('Training the CAE Next Motion model:')
            # Fit the model
            cae_next_motion_history = self.cae_next_motion.fit(train_next_diff_data, train_next_diff_data,
                                                               validation_split=self.validation_split,
                                                               epochs=self.cae_n_epochs,
                                                               batch_size=self.cae_batch_size,
                                                               shuffle=True, callbacks=self.callbacks_list)
            print('                                                                done!')
            print('=====================================================================')
            print('Evaluate the CAE Next Motion model:')
            predicted_next_motion_data = self.cae_next_motion.predict(train_next_diff_data)
            assert predicted_next_motion_data.shape == train_next_diff_data.shape, \
                'The predicted data shape is not right!'
            predicted_next_motion_data_2 = self.cae_next_motion_decoder.predict(
                self.cae_next_motion_encoder.predict(train_next_diff_data))
            assert np.all(predicted_next_motion_data == predicted_next_motion_data_2), \
                'Inconsistency between encoder and decoder!'
            cae_next_motion_scores = self.cae_next_motion.evaluate(train_next_diff_data, train_next_diff_data)
            self.cae_next_motion_global_mse = cae_next_motion_scores[0]
            self.cae_next_motion_accuracy = cae_next_motion_scores[1]
            print("%s: %.2f%%" % (self.cae_next_motion.metrics_names[1], self.cae_next_motion_accuracy * 100))
            print('                                                                done!')
        print('=====================================================================')
        print('Training the DAE CFV model:')
        # Get the appearance and motion latent features
        if train_cfv_ae_only:
            model_cae_appearances_filename = self.model_cae_appearances_name + '.h5'
            model_cae_prev_motion_filename = self.model_cae_prev_motion_name + '.h5'
            model_cae_next_motion_filename = self.model_cae_next_motion_name + '.h5'
            loaded_appearances_model = load_model(os.path.join(self.model_dir_path, model_cae_appearances_filename))
            loaded_prev_motion_model = load_model(os.path.join(self.model_dir_path, model_cae_prev_motion_filename))
            loaded_next_motion_model = load_model(os.path.join(self.model_dir_path, model_cae_next_motion_filename))
            print('Loaded models from disk')
            cfv_appearances = loaded_appearances_model.predict(train_roi_image_data)
            cfv_prev_motion = loaded_prev_motion_model.predict(train_prev_diff_data)
            cfv_next_motion = loaded_next_motion_model.predict(train_next_diff_data)
        else:
            cfv_appearances = self.cae_appearances_encoder.predict(train_roi_image_data)
            cfv_prev_motion = self.cae_prev_motion_encoder.predict(train_prev_diff_data)
            cfv_next_motion = self.cae_next_motion_encoder.predict(train_next_diff_data)
        cfv_all = np.hstack((cfv_prev_motion.reshape(len(cfv_prev_motion), -1),
                             cfv_appearances.reshape(len(cfv_appearances), -1),
                             cfv_next_motion.reshape(len(cfv_next_motion), -1)))
        # Scale the CFV data
        scale_cfv_all = MinMaxScaler()
        cfv_all_scaled = scale_cfv_all.fit_transform(cfv_all)
        # Fit the model
        dae_cfv_history = self.dae_cfv.fit(cfv_all_scaled, cfv_all_scaled,
                                           validation_split=self.validation_split,
                                           epochs=self.dae_n_epochs,
                                           batch_size=self.dae_batch_size,
                                           shuffle=True,
                                           callbacks=self.callbacks_list)
        print('                                                                done!')
        print('=====================================================================')
        print('Evaluate the DAE CFV model:')
        predicted_cfv_all = self.dae_cfv.predict(cfv_all_scaled)
        assert predicted_cfv_all.shape == cfv_all_scaled.shape, 'The predicted data shape is not right!'
        self.mse_per_sample = [mean_squared_error(cfv_all_scaled[i, :], predicted_cfv_all[i, :])
                               for i in range(cfv_all_scaled.shape[0])]
        scores = self.dae_cfv.evaluate(cfv_all_scaled, cfv_all_scaled)
        self.global_mse = scores[0]
        self.accuracy = scores[1]
        print("%s: %.2f%%" % (self.dae_cfv.metrics_names[1], self.accuracy*100))
        print('                                                                done!')
        print('=====================================================================')

        if save_model:
            print('Saving models to disk:')
            # Save models to HDF5
            model_cae_appearances_filename = self.model_cae_appearances_name + '.h5'
            model_cae_prev_motion_filename = self.model_cae_prev_motion_name + '.h5'
            model_cae_next_motion_filename = self.model_cae_next_motion_name + '.h5'
            model_dae_cfv_filename = self.model_dae_cfv_name + '.h5'
            scale_filename = 'scale_' + self.model_dae_cfv_name + '.pkl'

            # Save models
            if not train_cfv_ae_only:
                self.cae_appearances_encoder.save(os.path.join(self.model_dir_path, model_cae_appearances_filename))
                self.cae_prev_motion_encoder.save(os.path.join(self.model_dir_path, model_cae_prev_motion_filename))
                self.cae_next_motion_encoder.save(os.path.join(self.model_dir_path, model_cae_next_motion_filename))
            self.dae_cfv.save(os.path.join(self.model_dir_path, model_dae_cfv_filename))
            joblib.dump(scale_cfv_all, os.path.join(self.model_dir_path, scale_filename))
            print('                                                                done!')
            print('=====================================================================')
            if test_saved_model:
                print('Testing saved models:')
                loaded_appearances_model = load_model(os.path.join(self.model_dir_path, model_cae_appearances_filename))
                loaded_prev_motion_model = load_model(os.path.join(self.model_dir_path, model_cae_prev_motion_filename))
                loaded_next_motion_model = load_model(os.path.join(self.model_dir_path, model_cae_next_motion_filename))
                loaded_dae_cfv_model = load_model(os.path.join(self.model_dir_path, model_dae_cfv_filename))
                loaded_scale_cfv_all = joblib.load(os.path.join(self.model_dir_path, scale_filename))
                print('Loaded models from disk')
                # Get the appearance and motion latent features
                test_cfv_appearances = loaded_appearances_model.predict(train_roi_image_data)
                test_cfv_prev_motion = loaded_prev_motion_model.predict(train_prev_diff_data)
                test_cfv_next_motion = loaded_next_motion_model.predict(train_next_diff_data)
                test_cfv_all = np.hstack((test_cfv_prev_motion.reshape(len(test_cfv_prev_motion), -1),
                                          test_cfv_appearances.reshape(len(test_cfv_appearances), -1),
                                          test_cfv_next_motion.reshape(len(test_cfv_next_motion), -1)))
                test_cfv_all_scaled = loaded_scale_cfv_all.transform(test_cfv_all)
                test_predicted_cfv_all = loaded_dae_cfv_model.predict(test_cfv_all_scaled)
                assert test_predicted_cfv_all.shape == test_cfv_all.shape, 'The predicted data shape is not right!'
                test_scores = loaded_dae_cfv_model.evaluate(test_cfv_all_scaled, test_cfv_all_scaled)
                # Assert if the score is different than the original one
                assert_array_almost_equal(scores, test_scores, decimal=5,
                                          err_msg='Loaded model gives different scores than the original one.')
                print('Successfully evaluated the loaded models.')
                print('                                                                done!')
                print('=====================================================================')

        if plot_histories:
            if not train_cfv_ae_only:
                # summarize history for loss
                fig = plt.figure()
                plt.plot(cae_appearances_history.history['loss'])
                plt.plot(cae_appearances_history.history['val_loss'])
                plt.title('Model Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                figure_name = self.model_cae_appearances_name + '_loss.pdf'
                fig.savefig(os.path.join(self.model_dir_path, figure_name), bbox_inches='tight')
                # summarize history for loss
                fig = plt.figure()
                plt.plot(cae_prev_motion_history.history['loss'])
                plt.plot(cae_prev_motion_history.history['val_loss'])
                plt.title('Model Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                figure_name = self.model_cae_prev_motion_name + '_loss.pdf'
                fig.savefig(os.path.join(self.model_dir_path, figure_name), bbox_inches='tight')
                # summarize history for loss
                fig = plt.figure()
                plt.plot(cae_next_motion_history.history['loss'])
                plt.plot(cae_next_motion_history.history['val_loss'])
                plt.title('Model Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                figure_name = self.model_cae_next_motion_name + '_loss.pdf'
                fig.savefig(os.path.join(self.model_dir_path, figure_name), bbox_inches='tight')
            # summarize history for loss
            fig = plt.figure()
            plt.plot(dae_cfv_history.history['loss'])
            plt.plot(dae_cfv_history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            figure_name = self.model_dae_cfv_name + '_loss.pdf'
            fig.savefig(os.path.join(self.model_dir_path, figure_name), bbox_inches='tight')


def test_trained_ae_model(test_roi_image_data,
                          test_prev_diff_data,
                          test_next_diff_data,
                          model_dir_path='',
                          iteration_number=0):
    """
    Uses the saved pre-trained auto-encoder for scoring the test data.
    :param test_roi_image_data:
    :param test_prev_diff_data:
    :param test_next_diff_data:
    :param model_dir_path:
    :param iteration_number:
    """
    # Reshape input images if necessary
    if test_roi_image_data.shape[-1] != 1:
        test_roi_image_data = test_roi_image_data.reshape((*test_roi_image_data.shape, 1))
    if test_prev_diff_data.shape[-1] != 1:
        test_prev_diff_data = test_prev_diff_data.reshape((*test_prev_diff_data.shape, 1))
    if test_next_diff_data.shape[-1] != 1:
        test_next_diff_data = test_next_diff_data.reshape((*test_next_diff_data.shape, 1))

    model_cae_appearances_name = 'model_cae_appearances_' + str(iteration_number)
    model_cae_prev_motion_name = 'model_cae_prev_motion_' + str(iteration_number)
    model_cae_next_motion_name = 'model_cae_next_motion_' + str(iteration_number)
    model_dae_cfv_name = 'model_dae_cfv_' + str(iteration_number)

    # Load the saved models and compare the score
    model_cae_appearances_filename = model_cae_appearances_name + '.h5'
    model_cae_prev_motion_filename = model_cae_prev_motion_name + '.h5'
    model_cae_next_motion_filename = model_cae_next_motion_name + '.h5'
    model_dae_cfv_filename = model_dae_cfv_name + '.h5'
    scale_filename = 'scale_' + model_dae_cfv_name + '.pkl'

    loaded_cae_appearances_model = load_model(os.path.join(model_dir_path, model_cae_appearances_filename))
    loaded_cae_prev_motion_model = load_model(os.path.join(model_dir_path, model_cae_prev_motion_filename))
    loaded_cae_next_motion_model = load_model(os.path.join(model_dir_path, model_cae_next_motion_filename))
    loaded_dae_cfv_model = load_model(os.path.join(model_dir_path, model_dae_cfv_filename))
    loaded_scale_cfv_all = joblib.load(os.path.join(model_dir_path, scale_filename))
    print('Loaded models from disk')

    # evaluate loaded models on test data
    # Get the appearance and motion latent features
    encoded_cfv_appearances = loaded_cae_appearances_model.predict(test_roi_image_data)
    encoded_cfv_prev_motion = loaded_cae_prev_motion_model.predict(test_prev_diff_data)
    encoded_cfv_next_motion = loaded_cae_next_motion_model.predict(test_next_diff_data)
    test_cfv_all = np.hstack((encoded_cfv_prev_motion.reshape(len(encoded_cfv_prev_motion), -1),
                              encoded_cfv_appearances.reshape(len(encoded_cfv_appearances), -1),
                              encoded_cfv_next_motion.reshape(len(encoded_cfv_next_motion), -1)))
    test_cfv_all_scaled = loaded_scale_cfv_all.transform(test_cfv_all)

    scores = loaded_dae_cfv_model.evaluate(test_cfv_all_scaled, test_cfv_all_scaled)
    print('%s: %.2f%%' % (loaded_dae_cfv_model.metrics_names[1], scores[1]*100))

    # extract results
    global_mse = scores[0]
    predicted_cfv_all = loaded_dae_cfv_model.predict(test_cfv_all_scaled)
    assert predicted_cfv_all.shape == test_cfv_all_scaled.shape, 'The predicted data shape is not right!'
    mse_per_sample = [mean_squared_error(test_cfv_all_scaled[i, :], predicted_cfv_all[i, :])
                      for i in range(test_cfv_all_scaled.shape[0])]

    kb.clear_session()

    return global_mse, mse_per_sample


def get_global_result(gt_results,
                      ts_results,
                      result_dir_path='',
                      iteration_number=0,
                      plot_figure=True):
    """
    Computes the ROC and AUC using all the sequences.
    :param gt_results: contains only the gt labels, NOT the frame indexes
    :param ts_results: contains only the scores, NOT the frame indexes
    :param result_dir_path:
    :param iteration_number:
    :param plot_figure:
    :return:
    """
    roc_result_name = 'roc_of_model_' + str(iteration_number)
    # Check if the sizes are correct
    assert len(gt_results) == len(ts_results), 'Something is wrong with the length of result array.'
    flat_ts_results = []
    flat_gt_results = []
    for gt_seq, ts_seq in zip(gt_results, ts_results):
        assert len(gt_seq) == len(ts_seq), 'Something is wrong with the length of result sequence array.'
        flat_ts_results += ts_seq
        flat_gt_results += gt_seq
    # Flatten both results.
    flat_ts_results = np.array(flat_ts_results)
    flat_gt_results = np.array(flat_gt_results)
    assert len(flat_gt_results) == len(flat_ts_results), 'Something is wrong with the dimension of result array.'
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(flat_gt_results, flat_ts_results)
    roc_auc = auc(fpr, tpr)
    if plot_figure:
        # Save figure
        fig = plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        figure_name = roc_result_name + '.pdf'
        fig.savefig(os.path.join(result_dir_path, figure_name), bbox_inches='tight')
    return roc_auc
