#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/12 上午12:02
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.append("../")
from abc import ABCMeta
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda, merge
from keras import backend as K
from keras.callbacks import TensorBoard
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from callbacks import ModelCheckPointByBatch_EarlyStop_LRScheduler
from callbacks.lr_schedule import divide_decay
from utils.keras_callbaks import ModelSave_EarlyStop_LRDecay


class BaseModel(object):
    """ Abstract base model for all text matching model """
    __metaclass__ = ABCMeta

    def __init__(self, data, cfg, lr_drop_epoch, model_name):
        self.data = data
        self.cfg = cfg
        self.lr_drop_epoch = lr_drop_epoch
        self.model_name = model_name
        self.time_str = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))

    @NotImplementedError
    def build_model(self, data):
        """ 构建模型 """
        raise NotImplementedError

    def _run_out_of_fold(self, fold, batch_size, predict_batch_size, random_state, use_tensorbord):
        """ roof 方式训练模型 """
        best_model_dir = self.cfg.model_save_base_dir + self.model_name + '/'
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        pred_train_full = np.zeros(len(self.data['train_q1_words_seqs']))
        pred_test_full = 0
        cv_logloss = []
        roof_flod = fold
        best_train_logloss = []

        kf = StratifiedKFold(n_splits=roof_flod, shuffle=True, random_state=random_state)
        for kfold, (train_index, valid_index) in enumerate(kf.split(self.data['train_q1_words_seqs'], self.data['labels'])):
            print('\n============== perform fold {}, total folds {} =============='.format(kfold, roof_flod))

            train_input1 = self.data['train_q1_words_seqs'][train_index]
            train_input2 = self.data['train_q2_words_seqs'][train_index]
            train_y = self.data['labels'][train_index]

            valid_input1 = self.data['train_q1_words_seqs'][valid_index]
            valid_input2 = self.data['train_q2_words_seqs'][valid_index]
            valid_y = self.data['labels'][valid_index]

            # # data augment
            # train_input1 = np.array(train_input_1 + train_input_2)
            # train_input2 = np.array(train_input_2 + train_input_1)
            # train_y = np.array(train_y + train_y)

            model = self.build_model(self.data)
            # model = self.to_multi_gpu(model, n_gpus=2)

            # save initial weights, 手动保存的一个初始化 weights
            initial_model_name = '{}_initial_weights.h5'.format(self.model_name)
            initial_model_path = best_model_dir + initial_model_name
            if os.path.exists(initial_model_path):
                print('load initial weights')
                model.load_weights(filepath=initial_model_path)
            else:
                model.save_weights(initial_model_path, overwrite=True)

            input_channels = len(model.input_shape) / 2
            train_x = []
            for i in range(input_channels):
                train_x.extend([train_input1, train_input2])

            valid_x_1 = []
            valid_x_2 = []
            for i in range(input_channels):
                valid_x_1.extend([valid_input1, valid_input2])
                valid_x_2.extend([valid_input2, valid_input1])

            ########################################
            ## training the model and predict
            ########################################

            best_model_name = '{}_{}_kfold{}_batch_size{}_time{}.h5'.format(
                self.model_name, self.cfg.params_to_string(), kfold, batch_size, self.time_str
            )
            best_model_path = best_model_dir + best_model_name

            # callback = ModelSave_EarlyStop_LRDecay(model_path=best_model_path,
            #                                          save_best_only=True, save_weights_only=True,
            #                                          monitor='val_loss', mode='min',
            #                                          train_monitor='loss',
            #                                          lr_decay=1, patience=5, verbose=0)

            callback = ModelCheckPointByBatch_EarlyStop_LRScheduler(
                best_model_path=best_model_path,
                monitor='val_loss', train_monitor='loss',
                save_best_only=True, save_weights_only=True,
                validation_data=(valid_x_1, valid_y),
                metric_fun=log_loss,
                valid_batch_size=predict_batch_size,
                valid_batch_interval=int(len(train_y) / batch_size / 2),
                # valid_batch_interval=int(len(train_y) / batch_size),

                stop_patience_epoch=3, stop_min_delta=0.,
                lr_schedule_patience_epoch=2, schedule_fun=divide_decay,

                mode='min', verbose=0,
            )

            callbacks = [callback]
            if use_tensorbord:
                tensorbord = TensorBoard(log_dir='./los/{}/'.format(self.model_name))
                callbacks.append(tensorbord)

            # if os.path.exists(best_model_path):
            #     model.load_weights(best_model_path)

            model.fit(x=train_x,
                      y=train_y,
                      epochs=self.cfg.epochs,
                      batch_size=batch_size,
                      validation_data=(valid_x_1, valid_y),
                      verbose=1,
                      callbacks=callbacks)

            model.load_weights(filepath=best_model_path)

            # predict valid
            valid_pred_1 = model.predict(valid_x_1, batch_size=predict_batch_size)[:, 0]
            valid_pred_2 = model.predict(valid_x_2, batch_size=predict_batch_size)[:, 0]
            valid_pred = (valid_pred_1 + valid_pred_2) / 2.0

            train_logloss, valid_logloss = callback.train_best, callback.best
            print('train_logloss: {}, valid_logloss: {}'.format(train_logloss, valid_logloss))
            best_train_logloss.append(train_logloss)
            cv_logloss.append(valid_logloss)

            text_x_1 = []
            text_x_2 = []
            for i in range(input_channels):
                text_x_1.extend([self.data['test_q1_words_seq'], self.data['test_q2_words_seq']])
                text_x_2.extend([self.data['test_q2_words_seq'], self.data['test_q1_words_seq']])

            test_pred_1 = model.predict(text_x_1, batch_size=predict_batch_size)[:, 0]
            test_pred_2 = model.predict(text_x_2, batch_size=predict_batch_size)[:, 0]
            test_pred = (test_pred_1 + test_pred_2) / 2.0

            # run-out-of-fold predict
            pred_train_full[valid_index] = valid_pred
            pred_test_full += test_pred

        print('cv result:')
        print(cv_logloss)
        mean_valid_logloss = np.mean(cv_logloss)
        print('mean train logloss: {}, valid logloss: {}'.format(np.mean(best_train_logloss), mean_valid_logloss))

        print("saving predictions for ensemble")
        test_df = pd.DataFrame({'y_pre': pred_train_full})
        test_df.to_csv('{}train_{}_{}_cv{}_{}.csv'.format(
                self.cfg.save_ensemble_dir, self.model_name,
                self.cfg.params_to_string() + '_fold{}'.format(fold) + '_batch_size{}'.format(batch_size),
                mean_valid_logloss, self.time_str
            ),
            index=False
        )

        test_predict = pred_test_full / float(roof_flod)
        test_df = pd.DataFrame({'y_pre': test_predict})
        test_df.to_csv('{}test_{}_{}_cv{}_{}.csv'.format(
                self.cfg.save_ensemble_dir, self.model_name,
                self.cfg.params_to_string() + '_fold{}'.format(fold) + '_batch_size{}'.format(batch_size),
                mean_valid_logloss, self.time_str
            ),
            index=False
        )

    def _simple_train_predict(self):
        pass

    def train_and_predict(self, roof, fold, batch_size, predict_batch_size, random_state=42, use_tensorbord=False):
        if roof:
            self._run_out_of_fold(fold, batch_size, predict_batch_size, random_state, use_tensorbord)
        else:
            self._simple_train_predict()

    def slice_batch(self, x, n_gpus, part):
        sh = K.shape(x)

        L = sh[0] / n_gpus

        if part == n_gpus - 1:
            return x[part * L:]

        return x[part * L:(part + 1) * L]

    def to_multi_gpu(self, model, n_gpus=2):

        with tf.device('/cpu:0'):

            x = Input(model.input_shape[1:], name=model.input_names[0])

        towers = []

        merged = None
        for g in range(n_gpus):
            with tf.device('/gpu:' + str(g)):
                slice_g = Lambda(self.slice_batch, lambda shape: shape,
                                 arguments={'n_gpus': n_gpus, 'part': g})(x)

                towers.append(model(slice_g))

            with tf.device('/cpu:0'):
                merged = merge(towers, mode='concat', concat_axis=0)

        return Model(input=[x], output=merged)
