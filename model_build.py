#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @Time    : 2018/6/6 17:43
# @Author  : Allen
# @File    : model_build.py

import os
import fileUtils

import numpy as np
import datetime
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, Flatten, LSTM
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping


class model(object):

    def model_2cnnI2cnnC(self, main_shape):
        """
        2cnn Flatten + 2cnn Flatten
                3Dense
        :return:
        """
        main_input = Input(shape=main_shape)
        tower_1 = Conv1D(filters=128, kernel_size=3)(main_input)
        tower_1 = LeakyReLU()(tower_1)
        tower_1 = Conv1D(filters=128, kernel_size=3)(tower_1)
        tower_1 = LeakyReLU()(tower_1)
        tower_1 = Flatten()(tower_1)

        tower_2 = Conv1D(filters=128, kernel_size=3)(main_input)
        tower_2 = LeakyReLU()(tower_2)
        tower_2 = Conv1D(filters=128, kernel_size=3)(tower_2)
        tower_2 = LeakyReLU()(tower_2)
        tower_2 = Flatten()(tower_2)

        x = keras.layers.concatenate([tower_1, tower_2], axis=1)

        x = Dense(128)(x)
        x = LeakyReLU()(x)
        x = Dense(64)(x)
        x = LeakyReLU()(x)
        x = Dense(32)(x)
        x = LeakyReLU()(x)
        predictions = Dense(1)(x)
        adam = Adam(lr=0.001)
        model = Model(inputs=main_input, outputs=predictions)
        model.compile(loss='mean_squared_error',
                      optimizer=adam,
                      metrics=['accuracy'])
        return model

    # def model_cnnIcnnCcnn(self, main_shape):
    #     """
    #     cnn + cnn
    #        cnn
    #       Flatten
    #       3Dense
    #     :return:
    #     """
    #     main_input = Input(shape=main_shape)
    #     tower_1 = Conv1D(filters=128, kernel_size=3)(main_input)
    #     tower_1 = LeakyReLU()(tower_1)
    #     tower_2 = Conv1D(filters=128, kernel_size=3)(main_input)
    #     tower_2 = LeakyReLU()(tower_2)
    #     x = keras.layers.concatenate([tower_1, tower_2], axis=1)
    #     x = Conv1D(filters=128, kernel_size=3)(x)
    #     x = LeakyReLU()(x)
    #     x = Flatten()(x)  # 把多维输入进行一维
    #     x = Dense(128)(x)
    #     x = LeakyReLU()(x)
    #     x = Dense(64)(x)
    #     x = LeakyReLU()(x)
    #     x = Dense(32)(x)
    #     x = LeakyReLU()(x)
    #     predictions = Dense(1)(x)
    #     adam = Adam(lr=0.001)
    #     model = Model(inputs=main_input, outputs=predictions)
    #     model.compile(loss='mean_squared_error',
    #                   optimizer=adam,
    #                   metrics=['accuracy'])
    #     return model

    def model_cnnIlstmCcnn(self, main_shape):
        """
        cnn + lstm
           cnn
          Flatten
          3Dense
        :return:
        """
        main_input = Input(shape=main_shape)
        tower_1 = Conv1D(filters=128, kernel_size=3)(main_input)
        tower_1 = LeakyReLU()(tower_1)

        tower_2 = LSTM(128, return_sequences=True)(main_input)
        tower_2 = LeakyReLU()(tower_2)

        x = keras.layers.concatenate([tower_1, tower_2], axis=1)

        x = Conv1D(filters=128, kernel_size=3)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)  # 把多维输入进行一维

        x = Dense(128)(x)
        x = LeakyReLU()(x)
        x = Dense(64)(x)
        x = LeakyReLU()(x)
        x = Dense(32)(x)
        x = LeakyReLU()(x)
        predictions = Dense(1)(x)
        adam = Adam(lr=0.001)
        model = Model(inputs=main_input, outputs=predictions)
        model.compile(loss='mean_squared_error',
                      optimizer=adam,
                      metrics=['accuracy'])
        return model

    def model_2cnnIlstmC(self, main_shape):
        """
        2cnn Flatten + lstm Flatten
                  3Dense
        :return:
        """
        main_input = Input(shape=main_shape)
        tower_1 = Conv1D(filters=128, kernel_size=3)(main_input)
        tower_1 = LeakyReLU()(tower_1)
        tower_1 = Conv1D(filters=128, kernel_size=3)(tower_1)
        tower_1 = LeakyReLU()(tower_1)
        tower_1 = Flatten()(tower_1)
        tower_2 = LSTM(128, return_sequences=True)(main_input)
        tower_2 = LeakyReLU()(tower_2)
        tower_2 = Flatten()(tower_2)

        x = keras.layers.concatenate([tower_1, tower_2], axis=1)
        x = Dense(128)(x)
        x = LeakyReLU()(x)
        x = Dense(64)(x)
        x = LeakyReLU()(x)
        x = Dense(32)(x)
        x = LeakyReLU()(x)
        predictions = Dense(1)(x)
        adam = Adam(lr=0.001)
        model = Model(inputs=main_input, outputs=predictions)
        model.compile(loss='mean_squared_error',
                      optimizer=adam,
                      metrics=['accuracy'])
        return model

    def model_2cnnIlstmCcnn(self, main_shape):
        """
        2cnn + lstm
           cnn
          Flatten
          3Dense
        :return:
        """
        main_input = Input(shape=main_shape)
        tower_1 = Conv1D(filters=128, kernel_size=3)(main_input)
        tower_1 = LeakyReLU()(tower_1)
        tower_1 = Conv1D(filters=128, kernel_size=3)(tower_1)
        tower_1 = LeakyReLU()(tower_1)

        tower_2 = LSTM(128, return_sequences=True)(main_input)
        tower_2 = LeakyReLU()(tower_2)

        x = keras.layers.concatenate([tower_1, tower_2], axis=1)
        # x = Flatten()(x)
        x = Conv1D(filters=128, kernel_size=3)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)  # 把多维输入进行一维

        x = Dense(128)(x)
        x = LeakyReLU()(x)
        x = Dense(64)(x)
        x = LeakyReLU()(x)
        x = Dense(32)(x)
        x = LeakyReLU()(x)
        predictions = Dense(1)(x)
        adam = Adam(lr=0.001)
        model = Model(inputs=main_input, outputs=predictions)
        model.compile(loss='mean_squared_error',
                      optimizer=adam,
                      metrics=['accuracy'])
        return model

    def model_2cnnI2cnnC2lstm(self, main_shape):
        """
        2cnn + 2lstm
          3Dense
        :return:
        """
        main_input = Input(shape=main_shape)
        tower_1 = LSTM(128, return_sequences=True)(main_input)
        tower_1 = LeakyReLU()(tower_1)
        tower_1 = Flatten()(tower_1)

        tower_2 = LSTM(128, return_sequences=True)(main_input)
        tower_2 = LeakyReLU()(tower_2)
        tower_2 = Flatten()(tower_2)
        x = keras.layers.concatenate([tower_1, tower_2], axis=1)

        # x = LSTM(128, return_sequences=True)(x)
        # x = LeakyReLU()(x)

        x = Dense(128)(x)
        x = LeakyReLU()(x)
        x = Dense(64)(x)
        x = LeakyReLU()(x)
        x = Dense(32)(x)
        x = LeakyReLU()(x)
        predictions = Dense(1)(x)
        adam = Adam(lr=0.001)
        model = Model(inputs=main_input, outputs=predictions)
        model.compile(loss='mean_squared_error',
                      optimizer=adam,
                      metrics=['accuracy'])
        return model

    def model_2cnn(self, input_shape):
        model = Sequential()
        model.add(Conv1D(filters=128,
                         kernel_size=3,
                         input_shape=input_shape))
        model.add(LeakyReLU())
        model.add(Conv1D(filters=256,
                         kernel_size=3))
        model.add(LeakyReLU())
        model.add(Flatten())  # 把多维输入进行一维化
        model.add(Dense(128))
        model.add(LeakyReLU())
        model.add(Dense(64))
        model.add(LeakyReLU())
        model.add(Dense(32))
        model.add(LeakyReLU())
        model.add(Dense(1))
        adam = Adam(lr=0.001)
        model.compile(loss='mean_squared_error',
                      optimizer=adam,
                      metrics=['accuracy'])
        return model

    def modelnames(self):
        return filter(lambda x: x.startswith('model_') and callable(getattr(self, x)), dir(self))


def build_models(input_shape):
    model_temp = model()
    names = model_temp.modelnames()
    models = dict()
    for name in names:
        method = getattr(model_temp, name)
        models[name] = method(input_shape)
    return models


def train(model_name, model, train_x, train_y, batch_size=256, epochs=32, idicator_name=None):
    print('current model is %s...' % model_name)
    x = train_x[:, :, 2:train_x.shape[2]]
    # print('max=====', np.max(x))
    # print('min=====', np.min(x))
    y = train_y[:, train_y.shape[1] - 1]
    temp_dir = fileUtils.temp_dir
    file_name = '%s_value_%s_shape_%s_%s_1.0_.hf5' % (model_name, 0, idicator_name, datetime.datetime.now().strftime("%Y%m%d%H"))
    path = os.path.abspath(os.path.join(temp_dir, file_name))
    checkpoint = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    # tensorboad = Tensorboard(log_dir='log')
    # lrate = LearningRateScheduler(step_decay)
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto',
                                 epsilon=0.0001, cooldown=0, min_lr=0)
    earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    model.fit(x, y,
              batch_size=batch_size, epochs=epochs,
              callbacks=[checkpoint,
                         reducelr,
                         earlystop],
              verbose=1, validation_split=0.2, shuffle=True)
    print('%s training finished' % model_name)
    return file_name


def predict(model_name, pred_x):
    model = load_model(model_name)
    x = pred_x[:, :, 2:pred_x.shape[2]]
    pred_y = model.predict(x)
    print('pred_y=====\n', pred_y)
    return pred_y
