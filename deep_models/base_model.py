#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/12 上午12:02
"""
import os
import sys
import time
import math
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.path.append("../")
from abc import ABCMeta

from keras.callbacks import TensorBoard
from sklearn.model_selection import StratifiedKFold
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

    def _step_decay(self, epoch):
        """ Drop-Based Learning Rate Schedule
        LearningRate = InitialLearningRate * DropRate^floor(Epoch / EpochDrop)
        """
        initial_lrate = self.cfg.initial_lr
        lr_decay = self.cfg.lr_decay
        epochs_drop = self.lr_drop_epoch
        lrate = initial_lrate * math.pow(lr_decay, math.floor((1 + epoch) / epochs_drop))
        return lrate

    def _run_out_of_fold(self, fold, batch_size, predict_batch_size, random_state, use_tensorbord):
        """ roof 方式训练模型 """
        best_model_dir = self.cfg.model_save_base_dir + self.model_name + '/'
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        pred_train_full = np.zeros(len(self.data['train_q1_words_seqs']))
        pred_test_full = 0
        cv_logloss = []
        roof_flod = fold

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

            ########################################
            ## training the model and predict
            ########################################

            best_model_name = '{}_{}_kfold{}_batch_size{}_time{}.h5'.format(
                self.model_name, self.cfg.params_to_string(), kfold, batch_size, self.time_str
            )
            best_model_path = best_model_dir + best_model_name
            early_stop = ModelSave_EarlyStop_LRDecay(model_path=best_model_path,
                                                     save_best_only=True, save_weights_only=True,
                                                     monitor='val_loss', lr_decay=1,
                                                     patience=5, verbose=0, mode='min')
            # lr_scheduler = LearningRateScheduler(self.step_decay)
            callbacks = [early_stop]
            if use_tensorbord:
                tensorbord = TensorBoard(log_dir='./los/{}/'.format(self.model_name))
                callbacks.append(tensorbord)

            # if os.path.exists(best_model_path):
            #     model.load_weights(best_model_path)

            input_channels = len(model.input_shape) / 2
            train_x = []
            for i in range(input_channels):
                train_x.extend([train_input1, train_input2])

            valid_x_1 = []
            valid_x_2 = []
            for i in range(input_channels):
                valid_x_1.extend([valid_input1, valid_input2])
                valid_x_2.extend([valid_input2, valid_input1])

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

            valid_logloss = early_stop.best
            print('valid_logloss:', valid_logloss)
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
        mean_cv_logloss = np.mean(cv_logloss)
        print('Mean cv logloss:', mean_cv_logloss)

        print("saving predictions for ensemble")
        test_df = pd.DataFrame({'y_pre': pred_train_full})
        test_df.to_csv('{}train_{}_{}_cv{}_{}.csv'.format(
                self.cfg.save_ensemble_dir, self.model_name,
                self.cfg.params_to_string() + '_fold{}'.format(fold) + '_batch_size{}'.format(batch_size),
                mean_cv_logloss, self.time_str
            ),
            index=False
        )

        test_predict = pred_test_full / float(roof_flod)
        test_df = pd.DataFrame({'y_pre': test_predict})
        test_df.to_csv('{}test_{}_{}_cv{}_{}.csv'.format(
                self.cfg.save_ensemble_dir, self.model_name,
                self.cfg.params_to_string() + '_fold{}'.format(fold) + '_batch_size{}'.format(batch_size),
                mean_cv_logloss, self.time_str
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
