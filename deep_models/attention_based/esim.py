#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/29 下午3:16
"""
import sys

sys.path.append("../")
import warnings

warnings.filterwarnings('ignore')
from keras.models import Model
from keras.initializers import Constant
from keras.utils import plot_model
from base_model import BaseModel
from utils.keras_layers import *


class Esim(BaseModel):

    def build_model(self, data):
        shared_embedding_layer = Embedding(data['nb_words'],
                                    self.cfg.embedding_dim,
                                    weights=[data['embedding_matrix']],
                                    input_length=self.cfg.max_sequence_length,
                                    trainable=self.cfg.embed_trainable)
        shared_embed_bn_layer = BatchNormalization(axis=2)

        seq_1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16')
        seq_2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16')

        # Embedding
        embed_seq_1 = shared_embed_bn_layer(shared_embedding_layer(seq_1_input))
        embed_seq_2 = shared_embed_bn_layer(shared_embedding_layer(seq_2_input))

        # Encode
        shared_encode_layer = Bidirectional(CuDNNGRU(self.cfg.esim_cfg['rnn_units'], return_sequences=True))
        q1_encoded = shared_encode_layer(embed_seq_1)
        q2_encoded = shared_encode_layer(embed_seq_2)

        # Attention
        q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

        # Compose
        q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
        q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

        compose = Bidirectional(CuDNNGRU(self.cfg.esim_cfg['rnn_units'], return_sequences=True))
        q1_compare = compose(q1_combined)
        q2_compare = compose(q2_combined)

        # Aggregate
        q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        # Classifier
        merged = Concatenate()([q1_rep, q2_rep])
        dense = BatchNormalization()(merged)
        print('MLP input:', dense)

        for dense_unit in self.cfg.esim_cfg['dense_units']:
            dense = Dense(
                units=dense_unit,
                activation=self.cfg.esim_cfg['activation'],
                bias_initializer=Constant(value=0.1)
            )(dense)
            dense = BatchNormalization()(dense)
            dense = Dropout(self.cfg.esim_cfg['dense_dropout'])(dense)

        preds = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=[seq_1_input, seq_2_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer=self.cfg.esim_cfg['optimizer'],
                      metrics=['binary_accuracy'])
        # model.summary()
        plot_model(model, to_file='../assets/Esim.png', show_shapes=True, show_layer_names=True)

        return model
