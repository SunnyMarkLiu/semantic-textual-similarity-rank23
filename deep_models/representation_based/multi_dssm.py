#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/25 下午5:36
"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append("../")
from keras import layers
from keras.models import Model
from keras.utils import plot_model
from base_model import BaseModel
from layers import common
from utils.keras_layers import *


class DSSM(BaseModel):

    def build_model(self, data):
        """
        built model architecture
        """
        seq_1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int32')
        seq_2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int32')
        embedding_layer = Embedding(data['nb_words'],
                                    self.cfg.embedding_dim,
                                    weights=[data['word_embedding_matrix']],
                                    input_length=self.cfg.max_sequence_length,
                                    trainable=self.cfg.embed_trainable)
        embed_seq_1 = embedding_layer(seq_1_input)
        embed_seq_2 = embedding_layer(seq_2_input)

        q1 = TimeDistributed(Dense(self.cfg.embedding_dim, activation='relu'))(embed_seq_1)
        q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.cfg.embedding_dim,))(q1)

        q2 = TimeDistributed(Dense(self.cfg.embedding_dim, activation='relu'))(embed_seq_2)
        q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.cfg.embedding_dim,))(q2)

        # 计算 q1 和 q2 的 rmse
        minus_q2 = Lambda(lambda x: -x)(q2)
        subtracted_out = layers.add([q1, minus_q2])
        mse_out = Lambda(lambda x: x ** 2)(subtracted_out)
        multiply_out = layers.multiply([q1, q2])

        ########## engineered features #########
        features_input = Input(shape=(self.engineer_feature_count,), dtype='float', name='engineered_features')

        merged = concatenate([subtracted_out, mse_out, multiply_out, features_input])
        merged = BatchNormalization()(merged)

        for dense_unit in self.cfg.dssm_cfg['dense_units']:
            merged = Dropout(self.cfg.dssm_cfg['dense_dropout'])(merged)
            merged = Dense(dense_unit, activation=self.cfg.dssm_cfg['activation'])(merged)
            merged = BatchNormalization()(merged)

        preds = Dense(1, activation='sigmoid')(merged)
        model = Model(inputs=[seq_1_input, seq_2_input, features_input],
                      outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer=self.cfg.dssm_cfg['optimizer'], metrics=['binary_accuracy'])
        # model.summary()
        plot_model(model, to_file='../assets/DSSM.png', show_shapes=True, show_layer_names=True)

        return model


class CNN_DSSM(BaseModel):
    """
    ref: https://www.kaggle.com/rethfro/1d-cnn-single-model-score-0-14-0-16-or-0-23
    """

    def build_model(self, data):
        seq_length = self.cfg.max_sequence_length if self.word_chars == "words" else self.cfg.max_seq_chars_length

        q1_input = Input(shape=(seq_length,), dtype='int16', name='m1_q1_input')
        q2_input = Input(shape=(seq_length,), dtype='int16', name='m1_q2_input')

        embed_layer = Embedding(data['nb_{}'.format(self.word_chars)], self.cfg.embedding_dim,
                                weights=[data['word_embedding_matrix']],
                                input_length=self.cfg.max_sequence_length, trainable=self.cfg.embed_trainable,
                                name='embedding')

        q1_embed = embed_layer(q1_input)
        q2_embed = embed_layer(q2_input)

        # Run through CONV + GAP layers
        cnn_out1 = []
        cnn_out2 = []
        merge_cnn_out_shape = 0
        for filters, kernel_size in self.cfg.cnn_dssm_cfg['1d_cnn_filters_kernels']:
            conv1, conv2 = common.gated_liner_bn_units(q1_embed, q2_embed, filters=filters, kernel_size=kernel_size,
                                                       padding=self.cfg.cnn_dssm_cfg['padding'],
                                                       activation=self.cfg.cnn_dssm_cfg['activation'])
            conv1, conv2 = common.gated_liner_bn_units(conv1, conv2, filters=filters, kernel_size=kernel_size,
                                                       padding=self.cfg.cnn_dssm_cfg['padding'],
                                                       activation=self.cfg.cnn_dssm_cfg['activation'])

            bilstm_layer = Bidirectional(CuDNNLSTM(units=self.cfg.cnn_dssm_cfg['rnn_units'],
                                                    return_sequences=True))
            conv1 = bilstm_layer(conv1)
            glob1 = GlobalAveragePooling1D()(conv1)
            cnn_out1.append(glob1)

            conv2 = bilstm_layer(conv2)
            glob2 = GlobalAveragePooling1D()(conv2)
            cnn_out2.append(glob2)

            merge_cnn_out_shape += filters

        q1_rep = concatenate(cnn_out1)
        q2_rep = concatenate(cnn_out2)

        diff_rep = diff_features(q1_rep, q2_rep)
        ########## engineered features #########
        # features_input = Input(shape=(self.engineer_feature_count,), dtype='float', name='engineered_features')

        # features = Concatenate()([diff_rep, features_input])
        features = diff_rep
        print('total feature shape:', features.shape.as_list()[1])
        merged = BatchNormalization()(features)

        # The MLP that determines the outcome
        for dense_unit in self.cfg.cnn_dssm_cfg['dense_units']:
            merged = Dropout(self.cfg.cnn_dssm_cfg['dense_dropout'])(merged)
            merged = Dense(dense_unit, activation=None)(merged)
            merged = BatchNormalization()(merged)
            merged = Activation(activation=self.cfg.cnn_dssm_cfg['activation'])(merged)

        preds = Dense(1, activation='sigmoid')(merged)
        # model = Model(inputs=[q1_input, q2_input, features_input], outputs=preds)
        model = Model(inputs=[q1_input, q2_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer=self.cfg.cnn_dssm_cfg['optimizer'],
                      metrics=['binary_accuracy'])
        # model.summary()
        plot_model(model, to_file='../assets/CNN_DSSM.png', show_shapes=True, show_layer_names=True)

        return model


class GRU_DSSM(BaseModel):

    def build_model(self, data):
        q1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m1_q1_input')
        q2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m1_q2_input')

        embed_layer = Embedding(data['nb_words'], self.cfg.embedding_dim, weights=[data['word_embedding_matrix']],
                                input_length=self.cfg.max_sequence_length, trainable=self.cfg.embed_trainable,
                                name='embedding')
        q1_embed = embed_layer(q1_input)
        q2_embed = embed_layer(q2_input)

        # LSTM encode question input
        # a shared LSTM layer can give better generalization
        bilstm_layer1 = Bidirectional(CuDNNLSTM(units=self.cfg.gru_dssm_cfg['rnn_units'],
                                                return_sequences=True))
        bilstm_layer2 = Bidirectional(CuDNNLSTM(units=self.cfg.gru_dssm_cfg['rnn_units'],
                                                return_sequences=True))

        bigru_layer1 = Bidirectional(CuDNNGRU(units=self.cfg.gru_dssm_cfg['rnn_units'],
                                                return_sequences=True))
        bigru_layer2 = Bidirectional(CuDNNGRU(units=self.cfg.gru_dssm_cfg['rnn_units'],
                                                return_sequences=True))

        lstm_diff_rep = diff_features(bilstm_layer1(q1_embed), bilstm_layer2(q2_embed))
        gru_diff_rep = diff_features(bigru_layer1(q1_embed), bigru_layer2(q2_embed))

        diff_rep = Concatenate()([lstm_diff_rep, gru_diff_rep])
        merged = BatchNormalization()(diff_rep)

        for dense_unit in self.cfg.gru_dssm_cfg['dense_units']:
            merged = Dropout(self.cfg.gru_dssm_cfg['dense_dropout'])(merged)
            merged = Dense(dense_unit, activation=None)(merged)
            merged = Activation(activation=self.cfg.gru_dssm_cfg['activation'])(merged)
            merged = BatchNormalization()(merged)

        preds = Dense(1, activation='sigmoid')(merged)
        model = Model(inputs=[q1_input, q2_input],
                      outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer=self.cfg.gru_dssm_cfg['optimizer'],
                      metrics=['binary_accuracy'])
        # model.summary()
        plot_model(model, to_file='../assets/GRU_DSSM.png', show_shapes=True, show_layer_names=True)

        return model
