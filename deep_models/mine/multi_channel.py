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
from layers import common


class MultiChannelMatch(BaseModel):

    def build_model(self, data):
        ########## Input and embedding layer #########
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
        q1_embed = embed_layer(q1_input)
        q2_embed = embed_layer(q2_input)

        ########## model1: bi-lstm encode + attention soft align #########
        bilstm_layer1 = Bidirectional(CuDNNLSTM(units=self.cfg.mine_multi_channel_cfg['m1_rnn_units'],
                                                return_sequences=True))
        bilstm_layer2 = Bidirectional(CuDNNLSTM(units=self.cfg.mine_multi_channel_cfg['m1_rnn_units'],
                                                return_sequences=True))

        m1_q1 = bilstm_layer2(bilstm_layer1(q1_embed))
        m1_q2 = bilstm_layer2(bilstm_layer1(q2_embed))

        # Attention
        m1_q1_aligned, m1_q2_aligned = soft_attention_alignment(m1_q1, m1_q2)

        # Pooling
        m1_q1_rep = apply_multiple(m1_q1, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        m1_q2_rep = apply_multiple(m1_q2, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        m1_q1_aligned_rep = apply_multiple(m1_q1_aligned, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        m1_q2_aligned_rep = apply_multiple(m1_q2_aligned, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        ########## model2 1D-CNN #########
        cnn_out1 = []
        cnn_out2 = []
        merge_cnn_out_shape = 0
        for filters, kernel_size in self.cfg.mine_multi_channel_cfg['m2_1d_cnn_filters_kernels']:
            conv1, conv2 = common.gated_liner_units(q1_embed, q2_embed, filters=filters, kernel_size=kernel_size,
                                                    padding=self.cfg.mine_multi_channel_cfg['m2_padding'],
                                                    activation=self.cfg.mine_multi_channel_cfg['activation'])
            conv1, conv2 = common.gated_liner_units(conv1, conv2, filters=filters, kernel_size=kernel_size,
                                                    padding=self.cfg.mine_multi_channel_cfg['m2_padding'],
                                                    activation=self.cfg.mine_multi_channel_cfg['activation'])

            glob1 = GlobalAveragePooling1D()(conv1)
            cnn_out1.append(glob1)
            glob2 = GlobalAveragePooling1D()(conv2)
            cnn_out2.append(glob2)

            merge_cnn_out_shape += filters

        m2_q1_rep = concatenate(cnn_out1)
        m2_q2_rep = concatenate(cnn_out2)

        ########## engineered features #########
        features_input = Input(shape=(self.engineer_feature_count,), dtype='float', name='engineered_features')

        ################ clac difference over sentences of every models ################
        m1_diff     = diff_features(m1_q1_rep, m1_q2_rep)
        m1_att_diff = diff_features(m1_q1_aligned_rep, m1_q2_aligned_rep)
        m2_diff     = diff_features(m2_q1_rep, m2_q2_rep)

        dense = concatenate([m1_diff, m1_att_diff, m2_diff, features_input])

        # dense = concatenate([m1_q1_aligned_rep, m1_q2_aligned_rep, m1_diff, m1_att_diff,
        #                      m2_q1_rep, m2_q2_rep, m2_diff,
        #                      features_input])

        print('model1 features:', m1_diff.shape.as_list()[1])
        print('model1-att features:', m1_att_diff.shape.as_list()[1])
        print('model2 features:', m2_diff.shape.as_list()[1])
        print('handon features:', features_input.shape.as_list()[1])
        print('total feature shape:', dense.shape.as_list()[1])

        ################ MLP for prediction ################
        for dense_unit in self.cfg.mine_multi_channel_cfg['mlp_dense_units']:
            dense = Dropout(self.cfg.mine_multi_channel_cfg['mlp_dense_dropout'])(dense)
            dense = Dense(dense_unit, activation=None)(dense)
            dense = BatchNormalization()(dense)
            dense = Activation(activation=self.cfg.mine_multi_channel_cfg['activation'])(dense)

        preds = Dense(units=1, activation='sigmoid')(dense)
        model = Model(inputs=[q1_input, q2_input, features_input],
                      outputs=preds)

        model.compile(
            loss='binary_crossentropy',
            optimizer=self.cfg.mine_multi_channel_cfg['optimizer'],
            metrics=['binary_accuracy']
        )
        # model.summary()
        plot_model(model, to_file='../assets/MultiChannelMatch.png', show_shapes=True, show_layer_names=True)

        return model
