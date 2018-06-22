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


class Configure(object):
    """ global configuration """

    train_file = '/d_2t/lq/projects/magic_mirror_question_pairs/input/train.csv'
    test_file = '/d_2t/lq/projects/magic_mirror_question_pairs/input/test.csv'
    question_file = '/d_2t/lq/projects/magic_mirror_question_pairs/input/question.csv'
    dataset_file = '/d_2t/lq/projects/magic_mirror_question_pairs/input/dataset/max_doc_len{}_max_num_voc_words{}.pkl'

    word_embed_file = '/d_2t/lq/projects/magic_mirror_question_pairs/input/word_embed.txt'
    char_embed_file = '/d_2t/lq/projects/magic_mirror_question_pairs/input/char_embed..txt'

    word_embedding_matrix = '/d_2t/lq/projects/magic_mirror_question_pairs/input/word_embedding.pkl'
    char_embedding_matrix = '/d_2t/lq/projects/magic_mirror_question_pairs/input/char_embedding.pkl'
