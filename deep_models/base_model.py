#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/12 上午12:02
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append("../")
from abc import ABCMeta

# 文本处理
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from utils.keras_utils import ModelSave_EarlyStop_LRDecay

class BaseModel(object):
    """ Abstract base model for all text matching model """
    __metaclass__ = ABCMeta

    def __init__(self, data, cfg, model_name):
        self.data = data
        self.cfg = cfg
        self.model_name = model_name
        self.time_str = time.strftime('%m_%d_%H_%M', time.localtime(time.time()))

    @NotImplementedError
    def build_model(self, data):
        """ 构建模型 """
        raise NotImplementedError

    def _run_out_of_fold(self):
        """ roof 方式训练模型 """
        best_model_dir = self.cfg.model_save_base_dir + self.model_name
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        pred_train_full = np.zeros(len(self.data['train_q1_words_seqs']))
        pred_test_full = 0
        cv_logloss = []
        roof_flod = self.cfg.roof_fold

        kf = StratifiedKFold(n_splits=roof_flod, shuffle=True, random_state=42)
        for kfold, (train_index, valid_index) in enumerate(
                kf.split(self.data['train_q1_words_seqs'], self.data['labels'])):
            print('\n============== perform fold {} =============='.format(kfold))

            train_input1 = self.data['train_q1_words_seqs'][train_index]
            train_input2 = self.data['train_q2_words_seqs'][train_index]
            train_y = self.data['labels'][train_index]

            valid_input1 = self.data['train_q1_words_seqs'][valid_index]
            valid_input2 = self.data['train_q2_words_seqs'][valid_index]
            valid_y = self.data['labels'][valid_index]

            model = self.build_model(self.data)
            # model.summary()

            ########################################
            ## training the model and predict
            ########################################

            best_model_name = '{}_{}_kfold{}.h5'.format(self.model_name, self.cfg.params_to_string(), kfold)
            best_model_path = best_model_dir + best_model_name
            early_stop = ModelSave_EarlyStop_LRDecay(model_path=best_model_path,
                                                     save_best_only=True, save_weights_only=True,
                                                     monitor='val_loss', lr_decay=self.cfg.lr_decay,
                                                     patience=5, verbose=0, mode='min')

            # if os.path.exists(best_model_path):
            #     model.load_weights(best_model_path)

            model.fit(x=[train_input1, train_input2],
                      y=train_y,
                      epochs=self.cfg.epochs,
                      batch_size=self.cfg.batch_size,
                      validation_data=([valid_input1, valid_input2], valid_y),
                      verbose=1,
                      callbacks=[early_stop])

            model.load_weights(filepath=best_model_path)

            # predict valid
            valid_pred_1 = model.predict([valid_input1, valid_input2], batch_size=256)[:, 0]
            valid_pred_2 = model.predict([valid_input2, valid_input1], batch_size=256)[:, 0]
            valid_pred = (valid_pred_1 + valid_pred_2) / 2.0

            valid_logloss = early_stop.best
            print('valid_logloss:', valid_logloss)
            cv_logloss.append(valid_logloss)

            test_pred_1 = model.predict([self.data['test_q1_words_seq'], self.data['test_q2_words_seq']],
                                        batch_size=256)[:, 0]
            test_pred_2 = model.predict([self.data['test_q2_words_seq'], self.data['test_q1_words_seq']],
                                        batch_size=256)[:, 0]
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
        test_df.to_csv('{}/train_{}_{}_cv:{}_{}.zip'.format(
            self.cfg.save_ensemble_dir, self.model_name, self.cfg.params_to_string(), mean_cv_logloss, self.time_str
        ),
            compression='zip',
            index=False
        )

        test_predict = pred_test_full / float(roof_flod)
        test_df = pd.DataFrame({'y_pre': test_predict})
        test_df.to_csv('{}/test_{}_{}_cv:{}_{}.zip'.format(
            self.cfg.save_ensemble_dir, self.model_name, self.cfg.params_to_string(), mean_cv_logloss, self.time_str
        ),
            compression='zip',
            index=False
        )

    def _simple_train_predict(self):
        pass

    def train_and_predict(self, roof):
        if roof:
            self._run_out_of_fold()
        else:
            self._simple_train_predict()
