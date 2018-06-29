#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/29 下午1:17
"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append("../")
from keras import backend as K
from keras import layers
from keras.layers import Input, TimeDistributed, Dense, Lambda, CuDNNGRU, Conv1D, GlobalAveragePooling1D
from keras.layers import concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.utils import plot_model
from base_model import BaseModel
from layers import ManhattanDistance


class Siamese_LSTM(BaseModel):
    """
    Using MaLSTM model(Siamese networks + LSTM with Manhattan distance) to
    detect semantic similarity between question pairs.
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

        # RNN encoder
        rnn_layer = CuDNNGRU(self.cfg.siamese_lstm_cfg['rnn_units'])

        q1_encode = rnn_layer(embed_seq_1)
        q2_encode = rnn_layer(embed_seq_2)

        q1_encode = BatchNormalization()(q1_encode)
        q2_encode = BatchNormalization()(q2_encode)

        # Pack it all up into a Manhattan Distance model
        manhattan_dist = ManhattanDistance()([q1_encode, q2_encode])

        model = Model(inputs=[seq_1_input, seq_2_input],
                      outputs=manhattan_dist)
        model.compile(loss='mean_squared_error', optimizer=self.cfg.siamese_lstm_cfg['optimizer'], metrics=['binary_accuracy'])
        plot_model(model, to_file='../assets/Siamese_LSTM.png', show_shapes=True, show_layer_names=True)

        return model
