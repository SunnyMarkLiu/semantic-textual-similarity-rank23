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
from keras.utils import plot_model
from base_model import BaseModel
from utils.keras_layers import *


class Esim(BaseModel):

    def _local_inference_attention(self, input_1, input_2):
        attention = Dot(axes=-1)([input_1, input_2])
        w_att_1 = Lambda(lambda x: softmax(x, axis=1), output_shape=unchanged_shape)(attention)
        w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2), output_shape=unchanged_shape)(attention))
        q2_encoded_hat = Dot(axes=1)([w_att_1, input_1])
        q1_encoded_hat = Dot(axes=1)([w_att_2, input_2])
        return q1_encoded_hat, q2_encoded_hat

    def build_model(self, data):
        seq_length = self.cfg.max_seq_words_length if self.word_chars == "words" else self.cfg.max_seq_chars_length

        q1_input = Input(shape=(seq_length,), dtype='int16', name='m1_q1_input')
        q2_input = Input(shape=(seq_length,), dtype='int16', name='m1_q2_input')

        embed_weights = data['{}_embedding_matrix'.format(self.word_chars[:-1])]

        embed_layer = Embedding(data['nb_{}'.format(self.word_chars)],
                                self.cfg.embedding_dim,
                                weights=[embed_weights],
                                input_length=seq_length,
                                trainable=self.cfg.embed_trainable,
                                name='embedding')
        # Embedding
        embed_seq_1 = embed_layer(q1_input)
        embed_seq_2 = embed_layer(q2_input)

        # Encode
        shared_encode_layer1 = Bidirectional(CuDNNLSTM(units=self.cfg.esim_cfg['rnn_units'],
                                                       return_sequences=True), merge_mode='concat')
        shared_encode_layer2 = Bidirectional(CuDNNLSTM(units=self.cfg.esim_cfg['rnn_units'],
                                                       return_sequences=True), merge_mode='concat')
        q1_encoded = shared_encode_layer1(embed_seq_1)
        q1_encoded = shared_encode_layer2(q1_encoded)

        q2_encoded = shared_encode_layer1(embed_seq_2)
        q2_encoded = shared_encode_layer2(q2_encoded)

        # Attention
        q1_encoded_hat, q2_encoded_hat = self._local_inference_attention(q1_encoded, q2_encoded)

        # Compose
        q1_combined = Concatenate()([q1_encoded, q1_encoded_hat, diff_features(q1_encoded, q1_encoded_hat)])
        q2_combined = Concatenate()([q2_encoded, q2_encoded_hat, diff_features(q2_encoded, q2_encoded_hat)])

        compose_layer1 = Bidirectional(CuDNNLSTM(units=self.cfg.esim_cfg['rnn_units'],
                                                 return_sequences=True))
        compose_layer2 = Bidirectional(CuDNNLSTM(units=self.cfg.esim_cfg['rnn_units'],
                                                 return_sequences=True))
        q1_compare = compose_layer1(q1_combined)
        q1_compare = compose_layer2(q1_compare)

        q2_compare = compose_layer1(q2_combined)
        q2_compare = compose_layer2(q2_compare)

        # Aggregate
        q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        ########## engineered features #########
        features_input = Input(shape=(self.engineer_feature_count,), dtype='float', name='engineered_features')

        diff_fea = diff_features(q1_rep, q2_rep)

        dense = Concatenate()([diff_fea, features_input])
        dense = BatchNormalization()(dense)
        print('total feature shape:', dense.shape.as_list()[1])

        ################ MLP for prediction ################
        for dense_unit in self.cfg.esim_cfg['dense_units']:
            dense = Dropout(self.cfg.esim_cfg['dense_dropout'])(dense)
            dense = Dense(dense_unit, activation=None)(dense)
            dense = BatchNormalization()(dense)
            dense = Activation(activation=self.cfg.esim_cfg['activation'])(dense)

        preds = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=[q1_input, q2_input, features_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer=self.cfg.esim_cfg['optimizer'],
                      metrics=['binary_accuracy'])
        # model.summary()
        plot_model(model, to_file='../assets/Esim.png', show_shapes=True, show_layer_names=True)

        return model
