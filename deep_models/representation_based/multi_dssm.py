#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/25 下午5:36
"""
import warnings
import sys
sys.path.append("../")
from keras import backend as K
from keras import layers
from keras.layers import Input, TimeDistributed, Dense, Lambda, CuDNNGRU, Conv1D, GlobalAveragePooling1D
from keras.layers import concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.utils import plot_model

warnings.filterwarnings('ignore')
from base_model import BaseModel


class DSSM(BaseModel):

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

        embed_seq_1 = Dropout(self.cfg.dssm_cfg['embed_dropout'])(embed_seq_1)
        embed_seq_2 = Dropout(self.cfg.dssm_cfg['embed_dropout'])(embed_seq_2)

        q1 = TimeDistributed(Dense(self.cfg.embedding_dim, activation='relu'))(embed_seq_1)
        q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.cfg.embedding_dim,))(q1)

        q2 = TimeDistributed(Dense(self.cfg.embedding_dim, activation='relu'))(embed_seq_2)
        q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.cfg.embedding_dim,))(q2)

        # 计算 q1 和 q2 的 rmse
        minus_q2 = Lambda(lambda x: -x)(q2)
        subtracted_out = layers.add([q1, minus_q2])
        mse_out = Lambda(lambda x: x ** 2)(subtracted_out)
        multiply_out = layers.multiply([q1, q2])

        merged = concatenate([subtracted_out, mse_out, multiply_out])
        merged = BatchNormalization()(merged)

        for dense_unit in self.cfg.dssm_cfg['dense_units']:
            merged = Dropout(self.cfg.dssm_cfg['dense_dropout'])(merged)
            merged = Dense(dense_unit, activation=self.cfg.dssm_cfg['activation'])(merged)
            merged = BatchNormalization()(merged)

        preds = Dense(1, activation='sigmoid')(merged)
        model = Model(inputs=[seq_1_input, seq_2_input],
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
        embedding_layer = Embedding(data['nb_words'],
                                    self.cfg.embedding_dim,
                                    weights=[data['embedding_matrix']],
                                    input_length=self.cfg.max_sequence_length,
                                    trainable=self.cfg.embed_trainable)
        seq_1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int32')
        seq_2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int32')
        embed_seq_1 = embedding_layer(seq_1_input)
        embed_seq_2 = embedding_layer(seq_2_input)

        embed_seq_1 = Dropout(self.cfg.cnn_dssm_cfg['embed_dropout'])(embed_seq_1)
        embed_seq_2 = Dropout(self.cfg.cnn_dssm_cfg['embed_dropout'])(embed_seq_2)

        # Run through CONV + GAP layers
        cnn_out1 = []
        cnn_out2 = []
        merge_cnn_out_shape = 0
        for filters, kernel_size in self.cfg.cnn_dssm_cfg['1d_cnn_filters_kernels']:
            conv_layer = Conv1D(filters=filters, kernel_size=kernel_size,
                                padding=self.cfg.cnn_dssm_cfg['padding'],
                                activation=self.cfg.cnn_dssm_cfg['activation'])
            conv1 = conv_layer(embed_seq_1)
            glob1 = GlobalAveragePooling1D()(conv1)
            cnn_out1.append(glob1)

            conv2 = conv_layer(embed_seq_2)
            glob2 = GlobalAveragePooling1D()(conv2)
            cnn_out2.append(glob2)

            merge_cnn_out_shape += filters

        merged_1 = concatenate(cnn_out1)
        merged_2 = concatenate(cnn_out2)

        # take the explicit difference between the two sentences
        mse_diff = Lambda(lambda x: K.pow(x[0] - x[1], 2), output_shape=(merge_cnn_out_shape,))([merged_1, merged_2])
        # take the multiply different entries to get a different measure of equalness
        mul_diff = Lambda(lambda x: x[0] * x[1], output_shape=(merge_cnn_out_shape,))([merged_1, merged_2])

        # merge features
        merged = concatenate([merged_1, merged_2, mse_diff, mul_diff])
        merged = BatchNormalization()(merged)

        # The MLP that determines the outcome
        for dense_unit in self.cfg.cnn_dssm_cfg['dense_units']:
            merged = Dropout(self.cfg.cnn_dssm_cfg['dense_dropout'])(merged)
            merged = Dense(dense_unit, activation=self.cfg.cnn_dssm_cfg['activation'])(merged)
            merged = BatchNormalization()(merged)

        preds = Dense(1, activation='sigmoid')(merged)
        model = Model(inputs=[seq_1_input, seq_2_input],
                      outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer=self.cfg.cnn_dssm_cfg['optimizer'],
                      metrics=['binary_accuracy'])
        # model.summary()
        plot_model(model, to_file='../assets/CNN_DSSM.png', show_shapes=True, show_layer_names=True)

        return model


class GRU_DSSM(BaseModel):

    def build_model(self, data):
        embedding_layer = Embedding(data['nb_words'],
                                    self.cfg.embedding_dim,
                                    weights=[data['embedding_matrix']],
                                    input_length=self.cfg.max_sequence_length,
                                    trainable=self.cfg.embed_trainable)
        seq_1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int32')
        seq_2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int32')
        embed_seq_1 = embedding_layer(seq_1_input)
        embed_seq_2 = embedding_layer(seq_2_input)

        embed_seq_1 = Dropout(self.cfg.gru_dssm_cfg['embed_dropout'])(embed_seq_1)
        embed_seq_2 = Dropout(self.cfg.gru_dssm_cfg['embed_dropout'])(embed_seq_2)

        # LSTM encode question input
        # a shared LSTM layer can give better generalization
        rnn_layer = CuDNNGRU(self.cfg.gru_dssm_cfg['rnn_units'])
        # rnn_layer = LSTM(
        #     units=self.cfg.gru_dssm_cfg['rnn_units'],
        #     dropout=self.cfg.gru_dssm_cfg['rnn_dropout'],
        #     recurrent_dropout=self.cfg.gru_dssm_cfg['rnn_dropout']
        # )
        q1_encode = rnn_layer(embed_seq_1)
        q2_encode = rnn_layer(embed_seq_2)

        merged = concatenate([q1_encode, q2_encode])
        merged = BatchNormalization()(merged)

        for dense_unit in self.cfg.gru_dssm_cfg['dense_units']:
            merged = Dropout(self.cfg.gru_dssm_cfg['dense_dropout'])(merged)
            merged = Dense(dense_unit, activation=self.cfg.gru_dssm_cfg['activation'])(merged)
            merged = BatchNormalization()(merged)

        preds = Dense(1, activation='sigmoid')(merged)
        model = Model(inputs=[seq_1_input, seq_2_input],
                      outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer=self.cfg.gru_dssm_cfg['optimizer'],
                      metrics=['binary_accuracy'])
        # model.summary()
        plot_model(model, to_file='../assets/GRU_DSSM.png', show_shapes=True, show_layer_names=True)

        return model


class Merge_DSSM(BaseModel):

    def _get_diff_features(self, q1, q2, output_shape):
        # take the explicit difference between the two sentences
        mse_diff = Lambda(lambda x: K.pow(x[0] - x[1], 2), output_shape=output_shape)([q1, q2])
        # take the multiply different entries to get a different measure of equalness
        mul_diff = Lambda(lambda x: x[0] * x[1], output_shape=output_shape)([q1, q2])
        # merge features
        merged = concatenate([mse_diff, mul_diff])
        return merged

    def _mlp_model(self, data):
        embedding_layer = Embedding(data['nb_words'],
                                    self.cfg.embedding_dim,
                                    weights=[data['embedding_matrix']],
                                    input_length=self.cfg.max_sequence_length,
                                    trainable=self.cfg.embed_trainable)
        seq_1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int32')
        seq_2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int32')
        embed_seq_1 = embedding_layer(seq_1_input)
        embed_seq_2 = embedding_layer(seq_2_input)

        embed_seq_1 = Dropout(self.cfg.merge_dssm_cfg['embed_dropout'])(embed_seq_1)
        embed_seq_2 = Dropout(self.cfg.merge_dssm_cfg['embed_dropout'])(embed_seq_2)

        q1_encode = TimeDistributed(Dense(self.cfg.embedding_dim, activation='relu'))(embed_seq_1)
        q1_encode = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.cfg.embedding_dim,))(q1_encode)
        q2_encode = TimeDistributed(Dense(self.cfg.embedding_dim, activation='relu'))(embed_seq_2)
        q2_encode = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.cfg.embedding_dim,))(q2_encode)

        merged = self._get_diff_features(q1_encode, q2_encode, (self.cfg.embedding_dim,))
        mlp_out = BatchNormalization()(merged)

        return seq_1_input, seq_2_input, mlp_out

    def _lstm_model(self, data):
        embedding_layer = Embedding(data['nb_words'],
                                    self.cfg.embedding_dim,
                                    weights=[data['embedding_matrix']],
                                    input_length=self.cfg.max_sequence_length,
                                    trainable=self.cfg.embed_trainable)
        seq_1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int32')
        seq_2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int32')
        embed_seq_1 = embedding_layer(seq_1_input)
        embed_seq_2 = embedding_layer(seq_2_input)

        embed_seq_1 = Dropout(self.cfg.merge_dssm_cfg['embed_dropout'])(embed_seq_1)
        embed_seq_2 = Dropout(self.cfg.merge_dssm_cfg['embed_dropout'])(embed_seq_2)

        # LSTM encode question input
        # a shared LSTM layer can give better generalization
        rnn_layer = CuDNNGRU(self.cfg.gru_dssm_cfg['rnn_units'])
        # rnn_layer = LSTM(
        #     units=self.cfg.merge_dssm_cfg['rnn_units'],
        #     dropout=self.cfg.merge_dssm_cfg['rnn_dropout'],
        #     recurrent_dropout=self.cfg.merge_dssm_cfg['rnn_dropout']
        # )
        q1_encode = rnn_layer(embed_seq_1)
        q2_encode = rnn_layer(embed_seq_2)

        merged = self._get_diff_features(q1_encode, q2_encode, (self.cfg.merge_dssm_cfg['rnn_units'],))
        lstm_out = BatchNormalization()(merged)

        return seq_1_input, seq_2_input, lstm_out

    def _cnn_model(self, data):
        embedding_layer = Embedding(data['nb_words'],
                                    self.cfg.embedding_dim,
                                    weights=[data['embedding_matrix']],
                                    input_length=self.cfg.max_sequence_length,
                                    trainable=self.cfg.embed_trainable)
        seq_1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int32')
        seq_2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int32')
        embed_seq_1 = embedding_layer(seq_1_input)
        embed_seq_2 = embedding_layer(seq_2_input)

        embed_seq_1 = Dropout(self.cfg.merge_dssm_cfg['embed_dropout'])(embed_seq_1)
        embed_seq_2 = Dropout(self.cfg.merge_dssm_cfg['embed_dropout'])(embed_seq_2)

        # Run through CONV + GAP layers
        cnn_out1 = []
        cnn_out2 = []
        merge_cnn_out_shape = 0
        for filters, kernel_size in self.cfg.merge_dssm_cfg['1d_cnn_filters_kernels']:
            conv_layer = Conv1D(filters=filters, kernel_size=kernel_size,
                                padding=self.cfg.merge_dssm_cfg['padding'],
                                activation=self.cfg.merge_dssm_cfg['activation'])
            conv1 = conv_layer(embed_seq_1)
            glob1 = GlobalAveragePooling1D()(conv1)
            cnn_out1.append(glob1)

            conv2 = conv_layer(embed_seq_2)
            glob2 = GlobalAveragePooling1D()(conv2)
            cnn_out2.append(glob2)

            merge_cnn_out_shape += filters

        q1_encode = concatenate(cnn_out1)
        q2_encode = concatenate(cnn_out2)
        merged = self._get_diff_features(q1_encode, q2_encode, (merge_cnn_out_shape,))
        cnn_out = BatchNormalization()(merged)

        return seq_1_input, seq_2_input, cnn_out

    def build_model(self, data):
        # simple nn
        # mlp_input_1, mlp_input_2, mlp_out = self._mlp_model(data)
        # lstm
        lstm_input_1, lstm_input_2, lstm_out = self._lstm_model(data)
        # cnn
        cnn_input_1, cnn_input_2, cnn_out = self._cnn_model(data)
        # merge
        # merged = concatenate([mlp_out, lstm_out, cnn_out])
        merged = concatenate([lstm_out, cnn_out])
        merged = BatchNormalization()(merged)

        for dense_unit in self.cfg.merge_dssm_cfg['dense_units']:
            merged = Dropout(self.cfg.merge_dssm_cfg['dense_dropout'])(merged)
            merged = Dense(dense_unit, activation=self.cfg.merge_dssm_cfg['activation'])(merged)
            merged = BatchNormalization()(merged)

        preds = Dense(1, activation='sigmoid')(merged)
        model = Model(inputs=[
                              lstm_input_1, lstm_input_2,
                              cnn_input_1, cnn_input_2],
                      outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer=self.cfg.merge_dssm_cfg['optimizer'],
                      metrics=['binary_accuracy'])
        # model.summary()
        plot_model(model, to_file='../assets/Merge_DSSM.png', show_shapes=True, show_layer_names=True)

        return model
