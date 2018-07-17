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
from sklearn.model_selection import StratifiedKFold, train_test_split


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
        initial_model_name = '{}_{}_initial_weights_{}.h5'.format(self.model_name, self.word_chars, self.time_str)
        initial_model_path = best_model_dir + initial_model_name
        if os.path.exists(initial_model_path):
            print('load initial weights from {}'.format(initial_model_path))
            model.load_weights(filepath=initial_model_path)
        else:
            model.save_weights(initial_model_path, overwrite=True)

        kf = StratifiedKFold(n_splits=roof_flod, shuffle=True, random_state=random_state)
        for kfold, (train_index, test_index) in enumerate(
                kf.split(self.data['train_q1_{}_seqs'.format(self.word_chars)], self.data['labels'])):
            print('\n============== perform fold {}, total folds {} =============='.format(kfold, roof_flod))

            train_input1 = self.data['train_q1_{}_seqs'.format(self.word_chars)][train_index]
            train_input2 = self.data['train_q2_{}_seqs'.format(self.word_chars)][train_index]
            train_features = self.data['train_features'][train_index]
            train_y = self.data['labels'][train_index]

            # 划分出 fit 的 valid 数据
            test_size = 0.2
            train_q1, valid_q1, y_train, y_valid = train_test_split(train_input1, train_y, random_state=2018, test_size=test_size)
            train_q2, valid_q2, y_train, y_valid = train_test_split(train_input2, train_y, random_state=2018, test_size=test_size)
            train_features, valid_features, y_train, y_valid = train_test_split(train_features, train_y, random_state=2018, test_size=test_size)

            if use_pseudo_label:
                random.seed(random_state)

                test_size = self.data['test_q1_{}_seq'.format(self.word_chars)].shape[0]
                pseudo_index = random.sample(range(0, test_size), int(pseudo_label_ratio * test_size))

                train_q1 = np.concatenate((train_q1, self.data['test_q1_{}_seq'.format(self.word_chars)][pseudo_index]), axis=0)
                train_q2 = np.concatenate((train_q2, self.data['test_q2_{}_seq'.format(self.word_chars)][pseudo_index]), axis=0)
                train_features = np.concatenate((train_features, self.data['test_features'][pseudo_index]), axis=0)
                y_train = np.concatenate((y_train, self.data['test_pred_labels'][pseudo_index]), axis=0)

                # shuffle
                shuffle_index = random.sample(range(0, y_train.shape[0]), y_train.shape[0])
                train_q1 = train_q1[shuffle_index]
                train_q2 = train_q2[shuffle_index]
                y_train = y_train[shuffle_index]

            # fit 未看到的test数据
            test_q1 = self.data['train_q1_{}_seqs'.format(self.word_chars)][test_index]
            test_q2 = self.data['train_q2_{}_seqs'.format(self.word_chars)][test_index]
            test_features = self.data['train_features'][test_index]
            test_y = self.data['labels'][test_index]

            train_x = [train_q1, train_q2, train_features]
            valid_x = [valid_q1, valid_q2, valid_features]

            test_x_1 = [test_q1, test_q2, test_features]
            test_x_2 = [test_q2, test_q1, test_features]

            ########################################
            ## training the model and predict
            ########################################

            # best_model_name = '{}_{}_kfold{}_batch_size{}_time{}.h5'.format(
            #     self.model_name, self.cfg.params_to_string(), kfold, batch_size, self.time_str
            # )
            best_model_name = '{}_{}_best_weights_fold_{}_{}.h5'.format(self.model_name, self.word_chars, kfold, self.time_str)
            best_model_path = best_model_dir + best_model_name

            early_stop = EarlyStopping(monitor='val_loss',
                                       min_delta=1e-4,
                                       patience=2,
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
                      y=y_train,
                      epochs=self.cfg.epochs,
                      batch_size=batch_size,
                      validation_data=(valid_x, y_valid),
                      verbose=1,
                      callbacks=callbacks)

            # load final best model
            model.load_weights(filepath=best_model_path)

            # 预测fit未看到的 test
            test_pred_1 = model.predict(test_x_1, batch_size=predict_batch_size)[:, 0]
            test_pred_2 = model.predict(test_x_2, batch_size=predict_batch_size)[:, 0]
            test_pred = (test_pred_1 + test_pred_2) / 2.0

            test_logloss = log_loss(test_y, test_pred)
            print('current best test_logloss: {}'.format(test_logloss))
            cv_logloss.append(test_logloss)

            exact_text_x_1 = [self.data['test_q1_{}_seq'.format(self.word_chars)],
                              self.data['test_q2_{}_seq'.format(self.word_chars)],
                              self.data['test_features']]
            exact_text_x_2 = [self.data['test_q2_{}_seq'.format(self.word_chars)],
                              self.data['test_q1_{}_seq'.format(self.word_chars)],
                              self.data['test_features']]

            exact_test_pred_1 = model.predict(exact_text_x_1, batch_size=predict_batch_size)[:, 0]
            exact_test_pred_2 = model.predict(exact_text_x_2, batch_size=predict_batch_size)[:, 0]
            exact_test_pred = (exact_test_pred_1 + exact_test_pred_2) / 2.0

            # run-out-of-fold predict
            pred_train_full[test_index] = test_pred
            pred_test_full += exact_test_pred

        print('cv result:')
        print(cv_logloss)
        mean_valid_logloss = np.mean(cv_logloss)
        print('mean valid logloss: {}'.format(mean_valid_logloss))

        print("saving predictions for ensemble")
        test_df = pd.DataFrame({'y_pre': pred_train_full})
        test_df.to_csv('{}train_{}_{}_cv{}_time{}.csv'.format(
                self.cfg.save_stable_ensemble_dir, self.model_name, self.word_chars, mean_valid_logloss, self.time_str
            ),
            index=False
        )

        test_predict = pred_test_full / float(roof_flod)
        test_df = pd.DataFrame({'y_pre': test_predict})
        test_df.to_csv('{}test_{}_{}_cv{}_time{}.csv'.format(
                self.cfg.save_stable_ensemble_dir, self.model_name, self.word_chars, mean_valid_logloss, self.time_str
            ),
            index=False
        )
        print('--------------------------------------------')
        print('test predict mean: {:.6f}, std: {:.6f}'.format(np.mean(test_predict), np.std(test_predict)))
        print('done.')

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
                                   patience=2,
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
