#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/12 下午11:34
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from keras.optimizers import Adam

class Configure(object):
    """ global configuration """

    # directories
    train_data_file     = '/d_2t/lq/projects/semantic_similarity/input/train.csv'
    test_data_file      = '/d_2t/lq/projects/semantic_similarity/input/test.csv'
    question_file       = '/d_2t/lq/projects/semantic_similarity/input/question.csv'

    word_embed_path     = '/d_2t/lq/projects/semantic_similarity/input/word_embed.txt'
    char_embed_file     = '/d_2t/lq/projects/semantic_similarity/input/char_embed..txt'

    # 最有模型保存路径
    model_save_base_dir = '/d_2t/lq/projects/semantic_similarity/deep_models/check_points/'
    save_ensemble_dir   = '/d_2t/lq/projects/semantic_similarity/result/ensemble/'

    # global params
    initial_lr          = 0.01  # 初始 lr
    lr_decay            = 1     # lr 衰减比例
    max_sequence_length = 30    # 序列的最大长度
    max_nb_words        = 20890 # 词汇表的最大词汇数
    embedding_dim       = 300   # 词向量的维度
    embed_trainable     = False  # 词向量是否可训练
    use_data_aug        = False  # 是否使用数据扩充
    aug_frac            = 1   # 数据扩充比例
    random_state        = 42    # 随机数状态
    epochs              = 100   # 训练的最大 epoch，注意设置了 early stopping

    ########### models config ############

    # finetuned
    dssm_cfg = {
        'embed_dropout': 0.3,
        'dense_dropout' : 0.3,
        'dense_units'   : [1024, 512, 256],
        'activation'    : 'relu',
        'optimizer'     : 'adam'
    }

    # finetuned
    gru_dssm_cfg = {
        'embed_dropout': 0.3,
        'rnn_units'     : 300,
        'dense_dropout' : 0.3,
        'dense_units'   : [1024, 512, 256],
        'activation'    : 'relu',
        'optimizer'     : 'adam'
    }

    # finetuned
    cnn_dssm_cfg = {
        'embed_dropout': 0.3,
        '1d_cnn_filters_kernels' : [(128, 2), (128, 3), (128, 4), (128, 5), (128, 6)],
        'padding'   : 'same',
        'dense_units': [512, 256],
        'dense_dropout': 0.5,
        'activation': 'relu',
        'optimizer': Adam(lr=0.001)
    }

    # finetuned
    merge_dssm_cfg = {
        'embed_dropout': 0.3,

        'rnn_units': 400,

        '1d_cnn_filters_kernels': [(128, 2), (128, 3), (128, 4)],
        'padding': 'same',

        'dense_units' : [512],
        'dense_dropout': 0.3,
        'activation': 'relu',
        'optimizer': 'adam'
    }

    # overfitting easily?
    arcii_cfg = {
        'embed_dropout': 0.3,

        # layer 1
        '1d_cnn_filters': 128,
        '1d_cnn_kernel_size': 3,

        # layer 2
        '2d_cnn_filters_kernels': [(32, 3), (64, 3), (128, 3)],
        '2d_cnn_strides': 1,
        '2d_pool_size': 2,
        'padding': 'same',

        'dense_units': [256],
        'dense_dropout': 0.5,
        'activation': 'relu',
        'optimizer': 'adam'
    }

    match_pyramid_cfg = {
        'embed_dropout': 0.3,

        '2d_cnn_filters_kernels': [(32, 3), (64, 3), (128, 3),],
        '2d_cnn_strides': 1,
        '2d_pool_size': 2,
        'padding': 'same',

        'dense_units': [256],
        'dense_dropout': 0.5,
        'activation': 'relu',
        'optimizer': 'adam'
    }

    # my model
    multi_channel_match_cfg = {
        # 4个模型，每个模型的两个 sentence 求 diff， 拼接每个模型的 diff，转成矩阵，后面接 CNN
        # model1（mlp）：diff 输出维度等于 mlp_dense_units 225
        # model2（cnn）：diff 输出维度等于所有卷基层的 filter 数，100 * 4 = 400
        # model3（GRU）：diff 输出维度等于 rnn_units 300
        # model4（CNN-GRU）：diff 输出维度等于 rnn_units 300
        # 拼接之后，整体 diff 输出维度 225+400+300+300 = 1225 = 35 * 35，即矩阵的大小为 35x35，之后进行卷积

        'simple_architecture': True,   # completed perform better

        'mlp_dense_units': 100,
        'embed_dropout': 0.1,
        'rnn_units': 100,

        '1d_cnn_filters_kernels': [(50, 2), (50, 3), (50, 4), (50, 5)],  # 20*20=400, 30*30=900，必须是平方数
        'padding': 'same',

        '2d_cnn_filters_kernels': [(64, 3)],
        '2d_cnn_strides': 1,
        '2d_pool_size': 2,

        'dense_units': [512, 128],
        'dense_dropout': 0.5,

        'activation': 'relu',
        'lr': 0.0001
    }

    def params_to_string(self):
        param_str = 'max_seq_len{}-max_nb_words{}_embed_train{}_aug{}_augfrac{}_seed{}_lr_decay{}'.format(
            self.max_sequence_length,
            self.max_nb_words,
            self.embed_trainable,
            self.use_data_aug,
            self.aug_frac,
            self.random_state,
            self.lr_decay,
        )
        return param_str
