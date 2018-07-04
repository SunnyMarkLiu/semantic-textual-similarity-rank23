#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/7/3 下午1:29
"""
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append("../")
from keras.models import Model
from keras.utils import plot_model
from base_model import BaseModel
from utils.keras_layers import *


class MultiChannelMatch(BaseModel):

    def build_model(self, data):
        # Input
        ########## model1: bi-lstm encode + attention soft align #########
        m1_q1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m1_q1_input')
        m1_q2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m1_q2_input')

        shared_m1_embed_layer = Embedding(data['nb_words'], self.cfg.embedding_dim, weights=[data['word_embedding_matrix']],
                                          input_length=self.cfg.max_sequence_length, trainable=self.cfg.embed_trainable)
        shared_m1_lstm_layer = Bidirectional(CuDNNGRU(self.cfg.mine_multi_channel_cfg['m1_rnn_units'], return_sequences=True))

        m1_q1 = shared_m1_embed_layer(m1_q1_input)
        m1_q1 = shared_m1_lstm_layer(m1_q1)

        m1_q2 = shared_m1_embed_layer(m1_q2_input)
        m1_q2 = shared_m1_lstm_layer(m1_q2)

        # Attention
        m1_q1_aligned, m1_q2_aligned = soft_attention_alignment(m1_q1, m1_q2)

        # Pooling
        m1_q1_rep = apply_multiple(m1_q1, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        m1_q2_rep = apply_multiple(m1_q2, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        m1_q1_aligned_rep = apply_multiple(m1_q1_aligned, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        m1_q2_aligned_rep = apply_multiple(m1_q2_aligned, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        ########## model2 1D-CNN #########
        m2_q1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m2_q1_input')
        m2_q2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m2_q2_input')

        shared_m2_embed_layer = Embedding(data['nb_words'], self.cfg.embedding_dim, weights=[data['word_embedding_matrix']],
                                          input_length=self.cfg.max_sequence_length, trainable=self.cfg.embed_trainable)
        m2_q1 = shared_m2_embed_layer(m2_q1_input)
        m2_q2 = shared_m2_embed_layer(m2_q2_input)

        # Run through CONV + GAP layers
        cnn_out1 = []
        cnn_out2 = []
        merge_cnn_out_shape = 0
        for filters, kernel_size in self.cfg.mine_multi_channel_cfg['m2_1d_cnn_filters_kernels']:
            conv_layer = Conv1D(filters=filters, kernel_size=kernel_size,
                                padding=self.cfg.mine_multi_channel_cfg['m2_padding'],
                                activation=self.cfg.mine_multi_channel_cfg['activation'])
            conv1 = conv_layer(m2_q1)
            conv2 = conv_layer(m2_q2)

            # Attention
            # conv1, conv2 = soft_attention_alignment(conv1, conv2)

            glob1 = GlobalAveragePooling1D()(conv1)
            cnn_out1.append(glob1)
            glob2 = GlobalAveragePooling1D()(conv2)
            cnn_out2.append(glob2)

            merge_cnn_out_shape += filters

        m2_q1_rep = concatenate(cnn_out1)
        m2_q2_rep = concatenate(cnn_out2)

        ########## model3 Esim #########
        m3_q1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m3_q1_input')
        m3_q2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m3_q2_input')

        shared_m3_embed_layer = Embedding(data['nb_words'], self.cfg.embedding_dim,
                                          weights=[data['word_embedding_matrix']],
                                          input_length=self.cfg.max_sequence_length, trainable=self.cfg.embed_trainable)

        # Embedding
        m3_q1 = shared_m3_embed_layer(m3_q1_input)
        m3_q2 = shared_m3_embed_layer(m3_q2_input)

        # Encode
        shared_encode_layer = Bidirectional(CuDNNGRU(self.cfg.mine_multi_channel_cfg['m3_rnn_units'], return_sequences=True))
        q1_encoded = shared_encode_layer(m3_q1)
        q2_encoded = shared_encode_layer(m3_q2)

        # Attention
        q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

        # Compose
        q1_combined = Concatenate()([q1_encoded, q2_aligned, diff_features(q1_encoded, q2_aligned)])
        q2_combined = Concatenate()([q2_encoded, q1_aligned, diff_features(q2_encoded, q1_aligned)])

        compose = Bidirectional(CuDNNGRU(self.cfg.mine_multi_channel_cfg['m3_rnn_units'], return_sequences=True))
        q1_compare = compose(q1_combined)
        q2_compare = compose(q2_combined)

        # Aggregate
        m3_q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        m3_q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        ################ clac difference over sentences of every models ################
        m1_diff     = diff_features(m1_q1_rep, m1_q2_rep)
        m1_att_diff = diff_features(m1_q1_aligned_rep, m1_q2_aligned_rep)
        m2_diff     = diff_features(m2_q1_rep, m2_q2_rep)
        m3_diff     = diff_features(m3_q1_rep, m3_q2_rep)

        ################ MLP for prediction ################
        dense = concatenate([m1_diff, m1_att_diff, m2_diff, m3_diff])
        dense = BatchNormalization()(dense)

        print('mlp input:', dense)
        for dense_unit in self.cfg.mine_multi_channel_cfg['mlp_dense_units']:
            dense = Dropout(self.cfg.mine_multi_channel_cfg['mlp_dense_dropout'])(dense)
            dense = Dense(dense_unit, activation=self.cfg.mine_multi_channel_cfg['activation'])(dense)
            dense = BatchNormalization()(dense)

        preds = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=[m1_q1_input, m1_q2_input,
                              m2_q1_input, m2_q2_input,
                              m3_q1_input, m3_q2_input],
                      outputs=preds)

        model.compile(
            loss='binary_crossentropy',
            optimizer=self.cfg.mine_multi_channel_cfg['optimizer'],
            metrics=['binary_accuracy']
        )
        # model.summary()
        plot_model(model, to_file='../assets/MultiChannelMatch.png', show_shapes=True, show_layer_names=True)

        return model
