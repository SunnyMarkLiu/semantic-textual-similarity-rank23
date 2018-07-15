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

    word_embed_path     = '/d_2t/lq/projects/semantic_similarity/input/word300_3_0.vector'
    char_embed_path     = '/d_2t/lq/projects/semantic_similarity/input/char_embed.txt'

    # 最有模型保存路径
    model_save_base_dir = '/d_2t/lq/projects/semantic_similarity/deep_models/check_points/'
    single_result_dir   =  '/d_2t/lq/projects/semantic_similarity/result/'
    save_ensemble_dir   = '/d_2t/lq/projects/semantic_similarity/result/ensemble/'

    # 伪标签学习
    pseudo_label_test = '/d_2t/lq/projects/semantic_similarity/input/current_best_test_predict.csv'

    features_path = '/d_2t/lq/projects/semantic_similarity/input/features/'
    # 待 merge 的特征（特征名：merge_on 的特征）
    features = {
        'basic_features' : {'on': 'id', 'how': 'left'},
        'basic_features2': {'on': 'id', 'how': 'left'}
    }

    # global params
    initial_lr          = 0.01  # 初始 lr
    lr_decay            = 1     # lr 衰减比例
    max_seq_words_length = 39    # 序列的最大长度
    max_nb_words        = 20890 # 词汇表的最大词汇数
    embedding_dim       = 300   # 词向量的维度
    embed_trainable     = True  # 词向量是否可训练
    use_data_aug        = False  # 是否使用数据扩充
    aug_frac            = 1   # 数据扩充比例

    max_nb_chars        = 3048
    max_seq_chars_length= 56

    epochs              = 100   # 训练的最大 epoch，注意设置了 early stopping

    ########### models config ############

    # finetuned
    dssm_cfg = {
        'embed_dropout': 0.3,
        'dense_dropout' : 0.3,
        'dense_units'   : [1024, 512, 256],
        'activation'    : 'relu',
        'optimizer'     : Adam(lr=1e-3, clipvalue=10.0)
    }

    gru_dssm_cfg = {
        'rnn_units'     : 300,
        'dense_dropout' : 0.5,
        'dense_units'   : [512, 256],
        'activation'    : 'relu',
        'optimizer'     : Adam(lr=0.0001, clipvalue=10.0)
    }

    # finetuned
    cnn_dssm_cfg = {
        '1d_cnn_filters_kernels' : [(256, 2), (256, 3), (256, 4), (256, 5)],
        'padding'   : 'same',

        'rnn_units': 300,

        'dense_units': [512, 256],
        'dense_dropout': 0.5,
        'activation': 'relu',
        'optimizer': Adam(lr=0.0001, clipvalue=10.0)
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

    siamese_lstm_cfg = {
        'rnn_units': 400,
        'activation': 'relu',
        'optimizer': 'adam'
    }

    esim_cfg = {
        'embed_dropout': 0.2,
        'rnn_units': 300,
        'dense_units': [300, 300],
        'dense_dropout': 0.5,
        'activation': 'elu',
        'optimizer': Adam(lr=1e-4)
    }

    decom_att_cfg= {
        'projection_dim': 300,
        'projection_hidden': 100,
        'activation': 'elu',
        'projection_dropout': 0.2,

        'compare_dim': 500,
        'compare_dropout': 0.2,

        'dense_units': [512, 128],
        'dense_dropout': 0.5,

        'optimizer': Adam(lr=1e-3, clipvalue=10.0)
    }

    mine_multi_channel_cfg = {
        'm1_rnn_units': 192,
        'm1_dropout': 0.2,

        'm2_1d_cnn_filters_kernels': [(100, 2), (100, 3), (100, 4)],
        'm2_padding': 'same',

        'mlp_dense_units': [512, 128],
        'mlp_dense_dropout': 0.5,
        'activation': 'relu',
        'optimizer': Adam(lr=1e-3, clipvalue=10.0)
    }

    def params_to_string(self):
        param_str = 'max_seq_len{}-max_nb_words{}_embed_train{}_lr_decay{}'.format(
            self.max_seq_words_length,
            self.max_nb_words,
            self.embed_trainable,
            self.lr_decay,
        )
        return param_str
