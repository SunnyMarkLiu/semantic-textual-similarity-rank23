#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
Implement ARC-II model described in https://arxiv.org/pdf/1503.03244.pdf

@author: SunnyMarkLiu
@time  : 2018/6/26 下午3:26
"""
import sys
import warnings

sys.path.append("../")
from keras.layers import Input, Dense, Conv1D, Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization, Reshape, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Model

warnings.filterwarnings('ignore')
from base_model import BaseModel
from layers.match import Match


class ARCII(BaseModel):

    def build_model(self, data):
        """
        built model architecture
        """
        embedding_layer = Embedding(data['nb_words'],
                                    self.cfg.embedding_dim,
                                    weights=[data['embedding_matrix']],
                                    input_length=self.cfg.max_sequence_length,
                                    trainable=self.cfg.embed_trainable)
        seq_1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int32')
        seq_2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int32')
        embed_seq_1 = embedding_layer(seq_1_input)
        embed_seq_2 = embedding_layer(seq_2_input)

        embed_seq_1 = Dropout(self.cfg.arcii_cfg['embed_dropout'])(embed_seq_1)
        embed_seq_2 = Dropout(self.cfg.arcii_cfg['embed_dropout'])(embed_seq_2)

        # layer1 1D-convolution
        conv1 = Conv1D(
            filters=self.cfg.arcii_cfg['1d_cnn_filters'],
            kernel_size=self.cfg.arcii_cfg['1d_cnn_kernel_size'],
            padding=self.cfg.arcii_cfg['padding'],
            activation=self.cfg.arcii_cfg['activation']
        )(embed_seq_1)

        conv2 = Conv1D(
            filters=self.cfg.arcii_cfg['1d_cnn_filters'],
            kernel_size=self.cfg.arcii_cfg['1d_cnn_kernel_size'],
            padding=self.cfg.arcii_cfg['padding'],
            activation=self.cfg.arcii_cfg['activation']
        )(embed_seq_2)

        cross = Match(match_type='plus')([conv1, conv2])
        conv_out = Reshape((self.cfg.max_sequence_length, self.cfg.max_sequence_length, -1))(cross)

        # layer2 2D-convolution
        for filters, kernel_size in self.cfg.arcii_cfg['2d_cnn_filters_kernels']:
            conv_out = Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=self.cfg.arcii_cfg['2d_cnn_strides'],
                padding=self.cfg.arcii_cfg['padding'],
                activation=self.cfg.arcii_cfg['activation']
            )(conv_out)
            conv_out = MaxPooling2D(
                pool_size=self.cfg.arcii_cfg['2d_pool_size']
            )(conv_out)

        conv_out_flat = Flatten()(conv_out)
        conv_out = BatchNormalization()(conv_out_flat)

        for dense_unit in self.cfg.arcii_cfg['dense_units']:
            conv_out = Dropout(self.cfg.arcii_cfg['dense_dropout'])(conv_out)
            conv_out = Dense(dense_unit, activation=self.cfg.arcii_cfg['activation'])(conv_out)
            conv_out = BatchNormalization()(conv_out)

        preds = Dense(1, activation='sigmoid')(conv_out)
        model = Model(inputs=[seq_1_input, seq_2_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer=self.cfg.arcii_cfg['optimizer'],
                      metrics=['binary_accuracy'])
        # model.summary()

        return model
