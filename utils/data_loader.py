#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/22 下午9:13
"""
import gc
import os
import sys

import numpy as np
import pandas as pd

sys.path.append("../")
# 文本处理
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import warnings

warnings.filterwarnings('ignore')
import cPickle
from conf.configure import Configure
import data_utils


def load_pseudo_label_traindatas():
    """
    伪标签数据
    """
    test_df = pd.read_csv(Configure.pseudo_label_test)
    test_df['label'] = test_df['y_pre'].map(lambda x: int(x > 0.5))
    return test_df


def load_features():
    # 待预测订单的数据 （原始训练集和测试集）
    train = pd.read_csv(Configure.train_data_file, encoding='utf8')
    test = pd.read_csv(Configure.test_data_file, encoding='utf8')
    train['id'] = np.arange(train.shape[0])
    test['id'] = np.arange(test.shape[0])

    # 加载特征， 并合并
    features_merged_dict = Configure.features
    for feature_name in Configure.features:
        print 'merge features:', feature_name
        train_feature, test_feature = data_utils.load_features(feature_name)
        if 'label' in train_feature.columns:
            del train_feature['label']

        train = pd.merge(train, train_feature,
                         on=features_merged_dict[feature_name]['on'],
                         how=features_merged_dict[feature_name]['how'])
        test = pd.merge(test, test_feature,
                        on=features_merged_dict[feature_name]['on'],
                        how=features_merged_dict[feature_name]['how'])

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    train.drop(['id', 'q1', 'q2', 'q1_words', 'q1_chars', 'q2_words', 'q2_chars', 'label'], axis=1, inplace=True)
    test.drop(['id', 'q1', 'q2', 'q1_words', 'q1_chars', 'q2_words', 'q2_chars',], axis=1, inplace=True)

    selected_features = ['q1_q2_intersect', 'char_skew_q2vec', 'common_chars', 'char_kur_q1vec', 'braycurtis_distance',
                         'kur_q1vec', 'same_start_word', 'chars_fuzz_partial_token_set_ratio', 'cityblock_distance',
                         'same_start_char', 'cosine_distance', 'char_minkowski_distance', 'char_kur_q2vec',
                         'common_words', 'char_cityblock_distance', 'minkowski_distance', 'char_skew_q1vec',
                         'char_canberra_distance', 'chars_fuzz_token_sort_ratio', 'kur_q2vec', 'char_diff_len',
                         'word_diff_len', 'skew_q1vec', 'total_unique_chars', 'chars_fuzz_partial_ratio',
                         'char_braycurtis_distance', 'skew_q2vec', 'chars_fuzz_qratio', 'canberra_distance',
                         'words_fuzz_partial_token_set_ratio', 'chars_fuzz_token_set_ratio', 'char_set_diff_len',
                         'q1_freq', 'chars_fuzz_partial_token_sort_ratio', 'q2_freq', 'total_unique_words',
                         'words_fuzz_qratio', 'words_fuzz_partial_token_sort_ratio', 'word_set_diff_len',
                         'words_fuzz_WRatio', 'words_fuzz_token_sort_ratio', 'words_fuzz_token_set_ratio',
                         'words_fuzz_partial_ratio', 'chars_fuzz_WRatio']

    return train[selected_features], test[selected_features]


def load_datas(word_embed_path, question_file, train_data_file, test_data_file,
               max_nb_words, max_sequence_length, embedding_dim,
               char_embed_path, max_nb_chars, max_seq_chars_length,
               use_pseudo_label=False):
    data_file = 'max_nb_words{}_max_sequence_length{}_max_nb_chars{}_max_seq_chars_length{}.pkl'.format(
        max_nb_words, max_sequence_length, max_nb_chars, max_seq_chars_length
    )
    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            data = cPickle.load(f)

            train_features, test_features = load_features()

            data['train_features'] = train_features.values
            data['test_features'] = test_features.values
            if use_pseudo_label:
                test_pred_labels = load_pseudo_label_traindatas()['label']
                data['test_pred_labels'] = test_pred_labels

            return data

    print('load and process text dataset')
    questions = pd.read_csv(question_file)
    train = pd.read_csv(train_data_file)
    test = pd.read_csv(test_data_file)

    train['id'] = np.arange(train.shape[0])
    train = pd.merge(train, questions, left_on=['q1'], right_on=['qid'], how='left')
    train = train.rename(columns={'words': 'q1_words', 'chars': 'q1_chars'})
    del train['qid']
    train = pd.merge(train, questions, left_on=['q2'], right_on=['qid'], how='left')
    train = train.rename(columns={'words': 'q2_words', 'chars': 'q2_chars'})
    train.drop(['q1', 'q2', 'qid'], axis=1, inplace=True)

    test['id'] = np.arange(test.shape[0])
    test = pd.merge(test, questions, left_on=['q1'], right_on=['qid'], how='left')
    test = test.rename(columns={'words': 'q1_words', 'chars': 'q1_chars'})
    del test['qid']
    test = pd.merge(test, questions, left_on=['q2'], right_on=['qid'], how='left')
    test = test.rename(columns={'words': 'q2_words', 'chars': 'q2_chars'})
    test.drop(['q1', 'q2', 'qid'], axis=1, inplace=True)
    labels = train['label'].values
    # 拼接 train 和 test，方便处理
    test['label'] = -1
    all_df = pd.concat([train, test])

    print('max seq words:', max(questions['words'].map(lambda x: len(x.split(' ')))))
    print('max seq chars:', max(questions['chars'].map(lambda x: len(x.split(' ')))))

    print('====== process words ======')
    nb_words, word_embedding_matrix, train_q1_words_seqs, train_q2_words_seqs, test_q1_words_seq, test_q2_words_seq = \
        _process_words(word_embed_path, max_nb_words, questions, all_df, max_sequence_length, embedding_dim, train)

    print('====== process chars ======')
    nb_chars, char_embedding_matrix, train_q1_chars_seqs, train_q2_chars_seqs, test_q1_chars_seq, test_q2_chars_seq = \
        _process_chars(char_embed_path, max_nb_chars, questions, all_df, max_seq_chars_length, embedding_dim, train)

    data = {
        'nb_words': nb_words,
        'word_embedding_matrix': word_embedding_matrix,
        'labels': labels,
        'train_q1_words_seqs': train_q1_words_seqs,
        'train_q2_words_seqs': train_q2_words_seqs,
        'test_q1_words_seq': test_q1_words_seq,
        'test_q2_words_seq': test_q2_words_seq,

        'nb_chars': nb_chars,
        'char_embedding_matrix': char_embedding_matrix,
        'train_q1_chars_seqs': train_q1_chars_seqs,
        'train_q2_chars_seqs': train_q2_chars_seqs,
        'test_q1_chars_seq': test_q1_chars_seq,
        'test_q2_chars_seq': test_q2_chars_seq,
    }

    with open(data_file, "wb") as f:
        cPickle.dump(data, f, -1)

    return data


def _process_words(word_embed_path, max_nb_words, questions, all_df, max_sequence_length, embedding_dim, train):
    print('Indexing word vectors.')
    word_embeddings_index = {}
    with open(word_embed_path) as f:
        for line in f:
            values = line.split()
            word = str(values[0])
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings_index[word] = coefs
    print('Found %d word vectors.' % len(word_embeddings_index))

    tokenizer = Tokenizer(nb_words=max_nb_words)
    tokenizer.fit_on_texts(questions['words'])
    word_index = tokenizer.word_index
    print('Found %s unique words.' % len(word_index))
    print('Tokenize the text and then do padding the sentences to {} words'.format(max_sequence_length))
    q1_words_seqs = tokenizer.texts_to_sequences(all_df['q1_words'])
    q2_words_seqs = tokenizer.texts_to_sequences(all_df['q2_words'])

    q1_words_seqs = pad_sequences(q1_words_seqs, maxlen=max_sequence_length)
    q2_words_seqs = pad_sequences(q2_words_seqs, maxlen=max_sequence_length)

    train_q1_words_seqs = q1_words_seqs[:train.shape[0]]
    train_q2_words_seqs = q2_words_seqs[:train.shape[0]]
    test_q1_words_seq = q1_words_seqs[train.shape[0]:]
    test_q2_words_seq = q2_words_seqs[train.shape[0]:]

    print('Shape of question1 train data tensor:', train_q1_words_seqs.shape)
    print('Shape of question2 train data tensor:', train_q2_words_seqs.shape)
    print('Shape of question1 test data tensor:', test_q1_words_seq.shape)
    print('Shape of question2 test data tensor:', test_q2_words_seq.shape)
    del q1_words_seqs
    del q2_words_seqs
    gc.collect()

    print('Preparing embedding matrix.')
    # prepare embedding matrix
    nb_words = min(max_nb_words, len(word_index))

    word_embedding_matrix = np.zeros((nb_words, embedding_dim))
    for word, i in word_index.items():

        if i >= nb_words:
            continue
        word = str(word).upper()
        embedding_vector = word_embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            word_embedding_matrix[i] = embedding_vector
    print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))
    return nb_words, word_embedding_matrix, train_q1_words_seqs, train_q2_words_seqs, test_q1_words_seq, test_q2_words_seq


def _process_chars(char_embed_path, max_nb_chars, questions, all_df, max_seq_chars_length, embedding_dim, train):
    char_embeddings_index = {}
    with open(char_embed_path) as f:
        for line in f:
            values = line.split()
            char = str(values[0])
            coefs = np.asarray(values[1:], dtype='float32')
            char_embeddings_index[char] = coefs
    print('Found %d char vectors.' % len(char_embeddings_index))

    tokenizer = Tokenizer(nb_words=max_nb_chars)
    tokenizer.fit_on_texts(questions['chars'])
    char_index = tokenizer.word_index
    print('Found %s unique chars.' % len(char_index))
    print('Tokenize the text and then do padding the sentences to {} chars'.format(max_seq_chars_length))
    q1_chars_seqs = tokenizer.texts_to_sequences(all_df['q1_chars'])
    q2_chars_seqs = tokenizer.texts_to_sequences(all_df['q2_chars'])

    q1_chars_seqs = pad_sequences(q1_chars_seqs, maxlen=max_seq_chars_length)
    q2_chars_seqs = pad_sequences(q2_chars_seqs, maxlen=max_seq_chars_length)

    train_q1_chars_seqs = q1_chars_seqs[:train.shape[0]]
    train_q2_chars_seqs = q2_chars_seqs[:train.shape[0]]
    test_q1_chars_seq = q1_chars_seqs[train.shape[0]:]
    test_q2_chars_seq = q2_chars_seqs[train.shape[0]:]

    print('Shape of question1 train data tensor:', train_q1_chars_seqs.shape)
    print('Shape of question2 train data tensor:', train_q2_chars_seqs.shape)
    print('Shape of question1 test data tensor:', test_q1_chars_seq.shape)
    print('Shape of question2 test data tensor:', test_q2_chars_seq.shape)
    del q1_chars_seqs
    del q2_chars_seqs
    gc.collect()

    print('Preparing embedding matrix.')
    # prepare embedding matrix
    nb_chars = min(max_nb_chars, len(char_index))

    char_embedding_matrix = np.zeros((nb_chars, embedding_dim))
    for char, i in char_index.items():

        if i >= nb_chars:
            continue
        char = str(char).upper()
        embedding_vector = char_embeddings_index.get(char)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            char_embedding_matrix[i] = embedding_vector
    print('Null char embeddings: %d' % np.sum(np.sum(char_embedding_matrix, axis=1) == 0))

    return nb_chars, char_embedding_matrix, train_q1_chars_seqs, train_q2_chars_seqs, test_q1_chars_seq, test_q2_chars_seq
