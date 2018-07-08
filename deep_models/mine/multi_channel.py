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
from keras.models import Model, Sequential
from keras.utils import plot_model
from base_model import BaseModel
from utils.keras_layers import *


class MultiChannelMatch(BaseModel):

    def build_model(self, data):

        ########## model1: bi-lstm encode + attention soft align #########
        m1_q1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m1_q1_input')
        m1_q2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m1_q2_input')

        lstms_model = Sequential()
        lstms_model.add(Embedding(data['nb_words'], self.cfg.embedding_dim, weights=[data['word_embedding_matrix']],
                                          input_length=self.cfg.max_sequence_length, trainable=self.cfg.embed_trainable))
        # lstms_model.add(SpatialDropout1D(0.4))
        lstms_model.add(Bidirectional(CuDNNGRU(self.cfg.mine_multi_channel_cfg['m1_rnn_units'], return_sequences=True)))
        lstms_model.add(Bidirectional(CuDNNGRU(self.cfg.mine_multi_channel_cfg['m1_rnn_units'], return_sequences=True)))

        m1_q1 = lstms_model(m1_q1_input)
        m1_q2 = lstms_model(m1_q2_input)

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

        ########## engineered features #########
        engineered_features = Input(shape=(self.cfg.engineered_feature_size,), dtype='float', name='engineered_features')

        ################ clac difference over sentences of every models ################
        m1_diff     = diff_features(m1_q1_rep, m1_q2_rep)
        m1_att_diff = diff_features(m1_q1_aligned_rep, m1_q2_aligned_rep)
        m2_diff     = diff_features(m2_q1_rep, m2_q2_rep)

        dense = concatenate([m1_diff, m1_att_diff, m2_diff])
        dense = BatchNormalization()(dense)

        print('model1 features:', m1_diff.shape.as_list()[1])
        print('model1-att features:', m1_att_diff.shape.as_list()[1])
        print('model2 features:', m2_diff.shape.as_list()[1])
        print('total feature shape:', dense.shape.as_list()[1])

        ################ MLP for prediction ################
        for dense_unit in self.cfg.mine_multi_channel_cfg['mlp_dense_units']:
            dense = Dropout(self.cfg.mine_multi_channel_cfg['mlp_dense_dropout'])(dense)
            dense = Dense(dense_unit, activation=self.cfg.mine_multi_channel_cfg['activation'])(dense)
            dense = BatchNormalization()(dense)

        preds = Dense(units=1, activation='sigmoid')(dense)
        model = Model(inputs=[m1_q1_input, m1_q2_input,
                              m2_q1_input, m2_q2_input],
                      outputs=preds)

        model.compile(
            loss='binary_crossentropy',
            optimizer=self.cfg.mine_multi_channel_cfg['optimizer'],
            metrics=['binary_accuracy']
        )
        # model.summary()
        plot_model(model, to_file='../assets/MultiChannelMatch.png', show_shapes=True, show_layer_names=True)

        return model
