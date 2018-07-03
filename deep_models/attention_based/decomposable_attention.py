#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/7/2 下午10:04
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


class DecomposableAttention(BaseModel):

    def build_model(self, data):
        shared_embedding_layer = Embedding(data['nb_words'],
                                           self.cfg.embedding_dim,
                                           weights=[data['word_embedding_matrix']],
                                           input_length=self.cfg.max_sequence_length,
                                           trainable=self.cfg.embed_trainable)
        shared_embed_bn_layer = BatchNormalization(axis=2)

        seq_1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16')
        seq_2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16')

        # Embedding
        embed_seq_1 = shared_embed_bn_layer(shared_embedding_layer(seq_1_input))
        embed_seq_2 = shared_embed_bn_layer(shared_embedding_layer(seq_2_input))

        # Projection
        projection_layers = []
        if self.cfg.decom_att_cfg['projection_hidden'] > 0:
            projection_layers.extend([
                Dense(self.cfg.decom_att_cfg['projection_hidden'], activation=self.cfg.decom_att_cfg['activation']),
                Dropout(rate=self.cfg.decom_att_cfg['projection_dropout']),
            ])
        projection_layers.extend([
            Dense(self.cfg.decom_att_cfg['projection_dim'], activation=None),
            Dropout(rate=self.cfg.decom_att_cfg['projection_dropout']),
        ])
        q1_encoded = time_distributed(embed_seq_1, projection_layers)
        q2_encoded = time_distributed(embed_seq_2, projection_layers)

        # Attention
        q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

        # Compare
        q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
        q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])
        compare_layers = [
            Dense(self.cfg.decom_att_cfg['compare_dim'], activation=self.cfg.decom_att_cfg['activation']),
            Dropout(self.cfg.decom_att_cfg['compare_dropout']),
            Dense(self.cfg.decom_att_cfg['compare_dim'], activation=self.cfg.decom_att_cfg['activation']),
            Dropout(self.cfg.decom_att_cfg['compare_dropout']),
        ]
        q1_compare = time_distributed(q1_combined, compare_layers)
        q2_compare = time_distributed(q2_combined, compare_layers)

        # Aggregate
        q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        # Classifier
        merged = Concatenate()([q1_rep, q2_rep])
        dense = BatchNormalization()(merged)

        for dense_unit in self.cfg.decom_att_cfg['dense_units']:
            dense = Dense(
                units=dense_unit,
                activation=self.cfg.decom_att_cfg['activation'],
                bias_initializer=Constant(value=0.01)
            )(dense)
            dense = BatchNormalization()(dense)
            dense = Dropout(self.cfg.decom_att_cfg['dense_dropout'])(dense)

        out_ = Dense(1, activation='sigmoid')(dense)

        model = Model(inputs=[seq_1_input, seq_2_input], outputs=out_)
        model.compile(loss='binary_crossentropy', optimizer=self.cfg.decom_att_cfg['optimizer'],
                      metrics=['binary_crossentropy'])

        plot_model(model, to_file='../assets/DecomposableAttention.png', show_shapes=True, show_layer_names=True)

        return model
