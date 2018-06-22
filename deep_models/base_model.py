#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/12 上午12:02
"""
import os
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Layer, Activation
from keras import initializers, regularizers, constraints
from keras import backend as K
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split


def evaluate_score(predict, y_true):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, predict, pos_label=1)
    auc_score = auc(false_positive_rate, true_positive_rate)
    return auc_score


class BaseModel(object):
    """ Abstract base model for all text matching model """
    __metaclass__ = ABCMeta

    def __init__(self, data, bst_model_path, max_num_voc_words, epochs=50, max_sen_len=10, embed_size=300,
                 batch_size=64,
                 last_act_fn='relu', optimizer='adam', use_pretrained=True,
                 embed_trainable=True, **kwargs):
        """
        :param data: EasyDict 对象
        :param epochs: 迭代次数
        :param max_sen_len:  规整化每个句子的最大长度
        :param embed_size: 词向量维度
        :param last_act_fn: 最后一层的激活函数
        :param batch_size:
        :param optimizer: 优化器
        :param use_pretrained: 是否嵌入层使用预训练的模型
        :param trainable: 是否嵌入层可训练, 该参数只有在use_pretrained为真时有用
        :param kwargs: 不同模型的其他参数
        """
        self.data = data
        self.bst_model_path = bst_model_path
        self.max_num_voc_words = max_num_voc_words
        self.epochs = epochs
        self.max_sen_len = max_sen_len
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.last_act_fn = last_act_fn
        self.optimizer = optimizer
        self.use_pretrained = use_pretrained
        self.embed_trainable = embed_trainable
        self.kwargs = kwargs

        self.is_kfold = kwargs.get('is_kfold', False)
        self.kfold = kwargs.get('kfold', 0)
        if self.is_kfold:
            self.bst_model_path_list = []

    @abstractmethod
    def get_model(self, embed_trainable=None) -> Model:
        """ 构建模型 """
        raise NotImplementedError

    @abstractmethod
    def _get_bst_model_path(self) -> str:
        """return a name which is used for save trained weights"""
        raise NotImplementedError

    def train(self, random_state=42, valid_size=0.1):
        print(self.kwargs)
        model_checkpoint = ModelCheckpoint(self.bst_model_path, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        if self.is_kfold:
            pred_train_full = np.zeros(self.data.train_q1.shape[0])
            pred_test_full = 0
            cv_scores = []

            kf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=random_state)
            for i, (train_index, valid_index) in enumerate(kf.split(self.data.train_q1, self.data.y_train)):
                print('perform fold {}, train: {}, valid: {}'.format(i, len(train_index), len(valid_index)))
                train_q1, valid_q1= self.data.train_q1[train_index], self.data.train_q1[valid_index]
                train_q2, valid_q2= self.data.train_q2[train_index], self.data.train_q2[valid_index]

                y_train, y_valid = self.data.y_train[train_index], self.data.y_train[valid_index]

                model = self.get_model(self.embed_trainable)
                model.summary()
                bmp = self.bst_model_path + '_' + str(i)
                self.bst_model_path_list.append(bmp)
                model_checkpoint = ModelCheckpoint(bmp, save_best_only=True, save_weights_only=True)
                model.fit(x=[train_q1, train_q2],
                          y=y_train,
                          batch_size=self.batch_size,
                          epochs=self.epochs,
                          validation_data=[[valid_q1, valid_q2], y_valid],
                          verbose=1,
                          callbacks=[model_checkpoint, early_stopping])
                print("training done, save to ", bmp)
                valid_prd = model.predict([valid_q1, valid_q2])

                valid_auc = evaluate_score(valid_prd, y_valid)
                cv_scores.append(valid_auc)

                test_prd = model.predict([self.data.test_q1, self.data.test_q2])

                pred_train_full[valid_index] = valid_prd
                pred_test_full += test_prd

            mean_cv_scores = np.mean(cv_scores)
            print('Mean cv auc:', mean_cv_scores)
            print("saving for ensemble")
            train_pred_df = pd.DataFrame({'y_pre': pred_train_full})
            train_pred_df.to_csv(
                "{}_roof{}_predict_train_cv{}.csv".format(self.__class__.__name__, self.kfold, mean_cv_scores),
                index=False, columns=['y_pre'])

            pred_test_full = pred_test_full / float(self.kfold)
            test_pred_df = pd.DataFrame({'y_pre': pred_test_full})
            test_pred_df.to_csv(
                "{}_roof{}_predict_test_cv{}.csv".format(self.__class__.__name__, self.kfold, mean_cv_scores),
                index=False, columns=['y_pre'])
        else:
            model = self.get_model()
            model.summary()

            model.fit(x=[self.data.train_q1, self.data.train_q2],
                      y=self.data.y_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      verbose=1)
            print("training done, save to ", self.bst_model_path)
            predict_test = model.predict([self.data.test_q1, self.data.test_q2])

            prd_test_df = pd.DataFrame({'y_pre': predict_test})
            prd_test_df.to_csv('baseline.csv', index=False, columns=['y_pre'])
