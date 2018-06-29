#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/26 下午10:29
"""
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append("../")
import math
from keras import backend as K
from keras.layers import Input, TimeDistributed, Dense, Lambda, Conv1D, Conv2D, GlobalAveragePooling1D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization, Reshape, Flatten, concatenate, CuDNNGRU
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras import optimizers
from keras.utils import plot_model
from base_model import BaseModel


class MultiChannelMatch(BaseModel):

    def build_model(self, data):
        """
        built model architecture
        """
        """Input layers"""
        m1_q1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m1_q1_input')
        m1_q2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m1_q2_input')
        m2_q1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m2_q1_input')
        m2_q2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m2_q2_input')
        m3_q1_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m3_q1_input')
        m3_q2_input = Input(shape=(self.cfg.max_sequence_length,), dtype='int16', name='m3_q2_input')

        ########## model1: embed + maxpooling #########
        shared_m1_embed_layer = Embedding(data['nb_words'], self.cfg.embedding_dim, weights=[data['embedding_matrix']],
                                          input_length=self.cfg.max_sequence_length, trainable=self.cfg.embed_trainable)
        shared_m1_embed_dropout_layer = Dropout(self.cfg.multi_channel_match_cfg['embed_dropout'])
        shared_m1_timedistributed_layer  = TimeDistributed(Dense(self.cfg.multi_channel_match_cfg['mlp_dense_units'], activation='relu'))
        shared_m1_lambda_maxpooling_layer = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.cfg.multi_channel_match_cfg['mlp_dense_units'],))

        m1_q1 = shared_m1_embed_layer(m1_q1_input)
        m1_q1 = shared_m1_embed_dropout_layer(m1_q1)
        m1_q1 = shared_m1_timedistributed_layer(m1_q1)
        m1_q1 = shared_m1_lambda_maxpooling_layer(m1_q1)

        m1_q2 = shared_m1_embed_layer(m1_q2_input)
        m1_q2 = shared_m1_embed_dropout_layer(m1_q2)
        m1_q2 = shared_m1_timedistributed_layer(m1_q2)
        m1_q2 = shared_m1_lambda_maxpooling_layer(m1_q2)

        ########## model2 1D-CNN #########
        shared_m2_embed_layer = Embedding(data['nb_words'], self.cfg.embedding_dim, weights=[data['embedding_matrix']],
                                          input_length=self.cfg.max_sequence_length, trainable=self.cfg.embed_trainable)
        shared_m2_embed_dropout_layer = Dropout(self.cfg.multi_channel_match_cfg['embed_dropout'])

        m2_q1 = shared_m2_embed_layer(m2_q1_input)
        m2_q1 = shared_m2_embed_dropout_layer(m2_q1)

        m2_q2 = shared_m2_embed_layer(m2_q2_input)
        m2_q2 = shared_m2_embed_dropout_layer(m2_q2)

        # Run through CONV + GAP layers
        cnn_out1 = []
        cnn_out2 = []
        merge_cnn_out_shape = 0
        for filters, kernel_size in self.cfg.multi_channel_match_cfg['1d_cnn_filters_kernels']:
            conv_layer = Conv1D(filters=filters, kernel_size=kernel_size,
                                padding=self.cfg.multi_channel_match_cfg['padding'],
                                activation=self.cfg.multi_channel_match_cfg['activation'])
            conv1 = conv_layer(m2_q1)
            glob1 = GlobalAveragePooling1D()(conv1)
            cnn_out1.append(glob1)

            conv2 = conv_layer(m2_q2)
            glob2 = GlobalAveragePooling1D()(conv2)
            cnn_out2.append(glob2)

            merge_cnn_out_shape += filters

        m2_q1 = concatenate(cnn_out1)
        m2_q2 = concatenate(cnn_out2)

        ########## model3 BiLSTM #########
        shared_m3_embed_layer = Embedding(data['nb_words'], self.cfg.embedding_dim, weights=[data['embedding_matrix']],
                                          input_length=self.cfg.max_sequence_length, trainable=self.cfg.embed_trainable)
        shared_m3_embed_dropout_layer = Dropout(self.cfg.multi_channel_match_cfg['embed_dropout'])
        shared_m3_lstm_layer = CuDNNGRU(self.cfg.multi_channel_match_cfg['rnn_units'])

        m3_q1 = shared_m3_embed_layer(m3_q1_input)
        m3_q1 = shared_m3_embed_dropout_layer(m3_q1)
        m3_q1 = shared_m3_lstm_layer(m3_q1)

        m3_q2 = shared_m3_embed_layer(m3_q2_input)
        m3_q2 = shared_m3_embed_dropout_layer(m3_q2)
        m3_q2 = shared_m3_lstm_layer(m3_q2)

        ################ simple ################
        if self.cfg.multi_channel_match_cfg['simple_architecture']:
            # clac difference over sentences of every models
            mse_diff_1 = Lambda(lambda x: K.pow(x[0] - x[1], 2), output_shape=(self.cfg.multi_channel_match_cfg['mlp_dense_units'],))([m1_q1, m1_q2])
            mul_diff_1 = Lambda(lambda x: x[0] * x[1], output_shape=(self.cfg.multi_channel_match_cfg['mlp_dense_units'],))([m1_q1, m1_q2])

            mse_diff_2 = Lambda(lambda x: K.pow(x[0] - x[1], 2), output_shape=(merge_cnn_out_shape,))([m2_q1, m2_q2])
            mul_diff_2 = Lambda(lambda x: x[0] * x[1], output_shape=(merge_cnn_out_shape,))([m2_q1, m2_q2])

            mse_diff_3 = Lambda(lambda x: K.pow(x[0] - x[1], 2), output_shape=(self.cfg.multi_channel_match_cfg['rnn_units'],))([m3_q1, m3_q2])
            mul_diff_3 = Lambda(lambda x: x[0] * x[1], output_shape=(self.cfg.multi_channel_match_cfg['rnn_units'],))([m3_q1, m3_q2])

            merged = concatenate([mse_diff_1, mul_diff_1, mse_diff_2, mul_diff_2, mse_diff_3, mul_diff_3])
            features = BatchNormalization()(merged)
        else:
            ################ complicated ################

            # concate sentence representations
            q1_encode = concatenate([m1_q1, m2_q1, m3_q1])
            q2_encode = concatenate([m1_q2, m2_q2, m3_q2])

            # sentence representation dimention
            represent_dim = self.cfg.multi_channel_match_cfg['mlp_dense_units'] + merge_cnn_out_shape + \
                            self.cfg.multi_channel_match_cfg['rnn_units']
            width = int(math.sqrt(represent_dim))
            heigh = width

            q1_encode = Reshape((width, heigh, -1))(q1_encode)
            q2_encode = Reshape((width, heigh, -1))(q2_encode)

            match_matrix = concatenate([q1_encode, q2_encode], axis=-1)
            match_matrix = Reshape((width, heigh, -1))(match_matrix)

            # # arcii-like operation
            # match_matrix = Match(match_type='plus')([mse_diff, mul_diff])
            # match_matrix = Reshape((represent_dim, represent_dim, -1))(match_matrix)

            # CNN to extract features
            for filters, kernel_size in self.cfg.multi_channel_match_cfg['2d_cnn_filters_kernels']:
                match_matrix = Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=self.cfg.multi_channel_match_cfg['2d_cnn_strides'],
                    padding=self.cfg.multi_channel_match_cfg['padding'],
                    activation=self.cfg.multi_channel_match_cfg['activation']
                )(match_matrix)
                match_matrix = Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=self.cfg.multi_channel_match_cfg['2d_cnn_strides'],
                    padding=self.cfg.multi_channel_match_cfg['padding'],
                    activation=self.cfg.multi_channel_match_cfg['activation']
                )(match_matrix)
                match_matrix = MaxPooling2D(
                    pool_size=self.cfg.multi_channel_match_cfg['2d_pool_size']
                )(match_matrix)
            features = Flatten()(match_matrix)

        ################ MLP for prediction ################
        conv_out = BatchNormalization()(features)

        for dense_unit in self.cfg.multi_channel_match_cfg['dense_units']:
            conv_out = Dropout(self.cfg.multi_channel_match_cfg['dense_dropout'])(conv_out)
            conv_out = Dense(dense_unit, activation=self.cfg.multi_channel_match_cfg['activation'])(conv_out)
            conv_out = BatchNormalization()(conv_out)

        preds = Dense(1, activation='sigmoid')(conv_out)
        model = Model(inputs=[m1_q1_input, m1_q2_input,
                              m2_q1_input, m2_q2_input,
                              m3_q1_input, m3_q2_input],
                      outputs=preds)

        optimizer = optimizers.Adam(lr=self.cfg.multi_channel_match_cfg['lr'])

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
        # model.summary()
        plot_model(model, to_file='../assets/MultiChannelMatch.png', show_shapes=True, show_layer_names=True)

        return model
