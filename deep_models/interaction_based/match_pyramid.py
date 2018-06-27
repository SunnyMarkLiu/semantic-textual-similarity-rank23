#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/26 下午5:20
"""
import sys
import warnings

sys.path.append("../")
from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization, Reshape, Flatten, Dot
from keras.layers.embeddings import Embedding
from keras.models import Model

warnings.filterwarnings('ignore')
from base_model import BaseModel


class MatchPyramid(BaseModel):

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

        embed_seq_1 = Dropout(self.cfg.match_pyramid_cfg['embed_dropout'])(embed_seq_1)
        embed_seq_2 = Dropout(self.cfg.match_pyramid_cfg['embed_dropout'])(embed_seq_2)

        # dot product to get Matching Matrix
        match_matrix = Dot(axes=[2, 2], normalize=False)([embed_seq_1, embed_seq_2])
        # filter channel = 1
        match_matrix = Reshape((self.cfg.max_sequence_length, self.cfg.max_sequence_length, 1))(match_matrix)

        # CNN to extract features
        for filters, kernel_size in self.cfg.match_pyramid_cfg['2d_cnn_filters_kernels']:
            match_matrix = Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=self.cfg.match_pyramid_cfg['2d_cnn_strides'],
                padding=self.cfg.match_pyramid_cfg['padding'],
                activation=self.cfg.match_pyramid_cfg['activation']
            )(match_matrix)
            match_matrix = MaxPooling2D(
                pool_size=self.cfg.match_pyramid_cfg['2d_pool_size']
            )(match_matrix)

        # MLP
        conv_out = Flatten()(match_matrix)
        conv_out = BatchNormalization()(conv_out)

        for dense_unit in self.cfg.match_pyramid_cfg['dense_units']:
            conv_out = Dropout(self.cfg.match_pyramid_cfg['dense_dropout'])(conv_out)
            conv_out = Dense(dense_unit, activation=self.cfg.match_pyramid_cfg['activation'])(conv_out)
            conv_out = BatchNormalization()(conv_out)

        preds = Dense(1, activation='sigmoid')(conv_out)
        model = Model(inputs=[seq_1_input, seq_2_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer=self.cfg.arcii_cfg['optimizer'],
                      metrics=['binary_accuracy'])
        # model.summary()

        return model
