#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/12 上午12:02
"""
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.append("../")
from abc import ABCMeta
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.metrics import log_loss
from utils.keras_callbaks import ModelSave_EarlyStop_LRDecay
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras import optimizers


class BaseModel(object):
    """ Abstract base model for all text matching model """
    __metaclass__ = ABCMeta

    def __init__(self, data, cfg, lr_drop_epoch, model_name, engineer_feature_count=0, word_chars="words"):
        self.data = data
        self.cfg = cfg
        self.lr_drop_epoch = lr_drop_epoch
        self.model_name = model_name
        self.engineer_feature_count = engineer_feature_count
        self.word_chars = word_chars
        self.time_str = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))

    @NotImplementedError
    def build_model(self, data):
        """ 构建模型 """
        raise NotImplementedError

    def _run_out_of_fold(self, fold, batch_size, predict_batch_size, random_state,
                         use_tensorbord, use_pseudo_label, pseudo_label_ratio):
        """ roof 方式训练模型 """
        best_model_dir = self.cfg.model_save_base_dir + self.model_name + '/'
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        pred_train_full = np.zeros(len(self.data['train_q1_{}_seqs'.format(self.word_chars)]))
        pred_test_full = 0
        cv_logloss = []
        roof_flod = fold

        model = self.build_model(self.data)
        # model = self.to_multi_gpu(model, n_gpus=2)

        # save initial weights, 手动保存的一个初始化 weights
        # initial_model_name = '{}_initial_weights.h5'.format(self.model_name)
        # initial_model_name = '{}_add_features_initial_weights.h5'.format(self.model_name)
        initial_model_name = '{}_add_selected_features_initial_weights_{}.h5'.format(self.model_name, self.word_chars)
        initial_model_path = best_model_dir + initial_model_name
        if os.path.exists(initial_model_path):
            print('load initial weights')
            model.load_weights(filepath=initial_model_path)
        else:
            model.save_weights(initial_model_path, overwrite=True)

        kf = StratifiedKFold(n_splits=roof_flod, shuffle=True, random_state=random_state)
        for kfold, (train_index, valid_index) in enumerate(
                kf.split(self.data['train_q1_{}_seqs'.format(self.word_chars)], self.data['labels'])):
            print('\n============== perform fold {}, total folds {} =============='.format(kfold, roof_flod))

            train_input1 = self.data['train_q1_{}_seqs'.format(self.word_chars)][train_index]
            train_input2 = self.data['train_q2_{}_seqs'.format(self.word_chars)][train_index]
            train_features = self.data['train_features'][train_index]
            train_y = self.data['labels'][train_index]

            if use_pseudo_label:
                random.seed(random_state)

                test_size = self.data['test_q1_{}_seq'.format(self.word_chars)].shape[0]
                pseudo_index = random.sample(range(0, test_size), int(pseudo_label_ratio * test_size))

                train_input1 = np.concatenate((train_input1, self.data['test_q1_{}_seq'.format(self.word_chars)][pseudo_index]), axis=0)
                train_input2 = np.concatenate((train_input2, self.data['test_q2_{}_seq'.format(self.word_chars)][pseudo_index]), axis=0)
                train_features = np.concatenate((train_features, self.data['test_features'][pseudo_index]), axis=0)
                train_y = np.concatenate((train_y, self.data['test_pred_labels'][pseudo_index]), axis=0)

                # shuffle
                shuffle_index = random.sample(range(0, train_y.shape[0]), train_y.shape[0])
                train_input1 = train_input1[shuffle_index]
                train_input2 = train_input2[shuffle_index]
                train_y = train_y[shuffle_index]

            valid_input1 = self.data['train_q1_{}_seqs'.format(self.word_chars)][valid_index]
            valid_input2 = self.data['train_q2_{}_seqs'.format(self.word_chars)][valid_index]
            valid_features = self.data['train_features'][valid_index]
            valid_y = self.data['labels'][valid_index]

            train_x = [train_input1, train_input2, train_features]

            valid_x_1 = [valid_input1, valid_input2, valid_features]
            valid_x_2 = [valid_input2, valid_input1, valid_features]

            ########################################
            ## training the model and predict
            ########################################

            best_model_name = '{}_{}_kfold{}_batch_size{}_time{}.h5'.format(
                self.model_name, self.cfg.params_to_string(), kfold, batch_size, self.time_str
            )
            best_model_path = best_model_dir + best_model_name

            early_stop = EarlyStopping(monitor='val_loss',
                                       min_delta=1e-4,
                                       patience=3,
                                       mode='min')
            model_ckpt = ModelCheckpoint(filepath=best_model_path,
                                         monitor='val_loss',
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='min')
            callbacks = [early_stop, model_ckpt]

            if use_tensorbord:
                tensorbord = TensorBoard(log_dir='./los/{}/'.format(self.model_name))
                callbacks.append(tensorbord)

            # if os.path.exists(best_model_path):
            #     model.load_weights(best_model_path)

            model.load_weights(filepath=initial_model_path)
            model.compile(
                loss='binary_crossentropy',
                optimizer=self.cfg.mine_multi_channel_cfg['optimizer'],
                metrics=['binary_accuracy']
            )

            model.fit(x=train_x,
                      y=train_y,
                      epochs=self.cfg.epochs,
                      batch_size=batch_size,
                      validation_data=(valid_x_1, valid_y),
                      verbose=1,
                      callbacks=callbacks)

            # print('--> set embedding trainable, fine tuning')
            # # load previous best model
            # model.load_weights(filepath=best_model_path)
            # # clear！
            # callback.wait = 0
            # # set embedding layer trainable
            # model.get_layer('embedding').trainable = True
            # # recompile
            # model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-3), metrics=['accuracy'])
            # # retrain
            # model.fit(x=train_x,
            #           y=train_y,
            #           epochs=self.cfg.epochs,
            #           batch_size=batch_size,
            #           validation_data=(valid_x_1, valid_y),
            #           verbose=1,
            #           callbacks=callbacks)
            # load final best model
            model.load_weights(filepath=best_model_path)

            # predict valid
            valid_pred_1 = model.predict(valid_x_1, batch_size=predict_batch_size)[:, 0]
            valid_pred_2 = model.predict(valid_x_2, batch_size=predict_batch_size)[:, 0]
            valid_pred = (valid_pred_1 + valid_pred_2) / 2.0

            valid_logloss = early_stop.best
            print('current best valid_logloss: {}'.format(valid_logloss))
            cv_logloss.append(valid_logloss)

            text_x_1 = [self.data['test_q1_{}_seq'.format(self.word_chars)], self.data['test_q2_{}_seq'.format(self.word_chars)], self.data['test_features']]
            text_x_2 = [self.data['test_q2_{}_seq'.format(self.word_chars)], self.data['test_q1_{}_seq'.format(self.word_chars)], self.data['test_features']]

            test_pred_1 = model.predict(text_x_1, batch_size=predict_batch_size)[:, 0]
            test_pred_2 = model.predict(text_x_2, batch_size=predict_batch_size)[:, 0]
            test_pred = (test_pred_1 + test_pred_2) / 2.0

            # run-out-of-fold predict
            pred_train_full[valid_index] = valid_pred
            pred_test_full += test_pred

        print('cv result:')
        print(cv_logloss)
        mean_valid_logloss = np.mean(cv_logloss)
        print('mean valid logloss: {}'.format(mean_valid_logloss))

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

    def _simple_train_predict(self, predict_batch_size):
        best_model_dir = self.cfg.model_save_base_dir + self.model_name + '/'
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        best_model_name = '{}_time{}.h5'.format(
            self.model_name, self.time_str
        )
        best_model_path = best_model_dir + best_model_name

        model = self.build_model(self.data)

        train_q1_words_seqs, _, _, _ = train_test_split(self.data['train_q1_words_seqs'],
                                                        self.data['labels'],
                                                        random_state=2018,
                                                        test_size=0.5)
        train_q2_words_seqs, _, labels, _ = train_test_split(self.data['train_q2_words_seqs'],
                                                             self.data['labels'],
                                                             random_state=2018,
                                                             test_size=0.5)

        x_train1, x_valid1, y_train, y_valid = train_test_split(train_q1_words_seqs, labels,
                                                                random_state=2018,
                                                                test_size=0.1)
        x_train2, x_valid2, _, _ = train_test_split(train_q2_words_seqs, labels,
                                                    random_state=2018,
                                                    test_size=0.1)

        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=1e-4,
                                   patience=3,
                                   mode='min')
        model_ckpt = ModelCheckpoint(filepath=best_model_path,
                                     monitor='val_loss',
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min')
        callbacks = [early_stop, model_ckpt]

        model.fit([x_train1, x_train2], y_train, batch_size=predict_batch_size, epochs=self.cfg.epochs,
                  validation_data=([x_valid1, x_valid2], y_valid), callbacks=callbacks)

        # load final best model
        model.load_weights(filepath=best_model_path)
        # predict valid
        valid_pred_1 = model.predict([x_valid1, x_valid2], batch_size=predict_batch_size)[:, 0]
        valid_pred_2 = model.predict([x_valid2, x_valid1], batch_size=predict_batch_size)[:, 0]
        valid_pred = (valid_pred_1 + valid_pred_2) / 2.0
        print('valid logloss: {}'.format(log_loss(y_valid, valid_pred)))
        text_x_1 = [self.data['test_q1_words_seq'], self.data['test_q2_words_seq']]
        text_x_2 = [self.data['test_q2_words_seq'], self.data['test_q1_words_seq']]

        test_pred_1 = model.predict(text_x_1, batch_size=predict_batch_size)[:, 0]
        test_pred_2 = model.predict(text_x_2, batch_size=predict_batch_size)[:, 0]
        test_pred = (test_pred_1 + test_pred_2) / 2.0

        test_df = pd.DataFrame({'y_pre': test_pred})
        test_df.to_csv('{}{}_{}.csv'.format(
            self.cfg.single_result_dir, self.model_name, self.time_str
        ),
            index=False
        )

    def train_and_predict(self, roof, fold, batch_size, predict_batch_size, random_state=42,
                          use_tensorbord=False, use_pseudo_label=False, pseudo_label_ratio=0.0):
        if roof:
            self._run_out_of_fold(fold, batch_size, predict_batch_size, random_state, use_tensorbord,
                                  use_pseudo_label, pseudo_label_ratio)
        else:
            self._simple_train_predict(predict_batch_size)
