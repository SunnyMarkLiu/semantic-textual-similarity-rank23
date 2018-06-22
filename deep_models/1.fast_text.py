#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/22 下午3:09
"""
import os
import sys

import pandas as pd

sys.path.append("../")
# 文本处理
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras import backend as K
import warnings
warnings.filterwarnings('ignore')

from utils import data_loader
from utils.keras_utils import ModelCheckpointAndLearningRateDecay

# disable TF debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

########################################
## 设置数据路径和模型参数
########################################
# directories
BASE_DIR = '../../input/'
WORD_EMBED_PATH = BASE_DIR + 'word_embed.txt'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
QUESTION_FILE = BASE_DIR + 'question.csv'

# parameters
MAX_SEQUENCE_LENGTH = 20     # 序列的最大长度
MAX_NB_WORDS = 10000         # 词汇表的最大词汇数
EMBEDDING_DIM = 300         # 词向量的维度
VALIDATION_SPLIT = 0.02     # 线下验证集的划分比例
DROPOUT = 0.25              # dropout 比例
EMBEDDING_TRAINABLE = False # 词向量是否可以训练

########################################
## index word vectors
## 读取词向量文本转化为字典
## key为词，value为对应的 300 维的向量
########################################
data = data_loader.load_datas(word_embed_path=WORD_EMBED_PATH, question_file=QUESTION_FILE,
                       train_data_file=TRAIN_DATA_FILE, test_data_file=TEST_DATA_FILE,
                       max_nb_words=MAX_NB_WORDS, max_sequence_length=MAX_SEQUENCE_LENGTH,
                       embedding_dim=EMBEDDING_DIM)

########################################
## built model architecture
########################################
embedding_layer = Embedding(data['nb_words'],
                            EMBEDDING_DIM,
                            weights=[data['embedding_matrix']],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=EMBEDDING_TRAINABLE)
sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
embedded_sequences_2 = embedding_layer(sequence_2_input)

q1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(embedded_sequences_1)
q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM,))(q1)

q2 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(embedded_sequences_2)
q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM,))(q2)
merged = concatenate([q1, q2])
merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)

is_duplicate = Dense(1, activation='sigmoid')(merged)
model = Model(inputs=[sequence_1_input, sequence_2_input],
              outputs=is_duplicate)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
model.summary()
########################################
## training the model and predict
########################################
mest_model_path='./check_points/fast_text/best_fast_text_model.h5'
early_stopping=ModelCheckpointAndLearningRateDecay(model_path=mest_model_path, save_best_only=True, save_weights_only=True,
                                                   monitor='val_loss', lr_decay=0.99, patience=5, verbose=1, mode='min')
model.fit(x=[data['train_q1_words_seqs'], data['train_q2_words_seqs']],
          y=data['labels'],
          epochs=100,
          batch_size=128,
          validation_split=VALIDATION_SPLIT,
          verbose=1,
          callbacks=[early_stopping])

model.load_weights(filepath=mest_model_path)

y_pre = model.predict([data['test_q1_words_seq'], data['test_q2_words_seq']], batch_size=256)
# 测试集也进行增强，预测求平均
y_pre2 = model.predict([data['test_q2_words_seq'], data['test_q1_words_seq']], batch_size=256)

y_pre = y_pre[:, 0]
y_pre2 = y_pre2[:, 0]

submit = pd.DataFrame({'y_pre': y_pre})
submit.to_csv('../../result/fast_text_{}_{}_{}_{}_{}.csv'.format(MAX_SEQUENCE_LENGTH, MAX_NB_WORDS,
                                                                 VALIDATION_SPLIT, DROPOUT, EMBEDDING_TRAINABLE),
              index=False)

submit = pd.DataFrame({'y_pre': (y_pre + y_pre2) / 2})
submit.to_csv('../../result/avg_fast_text_{}_{}_{}_{}_{}.csv'.format(MAX_SEQUENCE_LENGTH, MAX_NB_WORDS,
                                                                 VALIDATION_SPLIT, DROPOUT, EMBEDDING_TRAINABLE),
              index=False)
