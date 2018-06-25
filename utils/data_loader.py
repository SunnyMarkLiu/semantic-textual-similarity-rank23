#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/22 下午9:13
"""
import gc
import sys

import numpy as np
import pandas as pd

sys.path.append("../")
# 文本处理
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import warnings

warnings.filterwarnings('ignore')


def load_datas(word_embed_path, question_file, train_data_file, test_data_file,
               max_nb_words, max_sequence_length, embedding_dim, use_data_aug, n_gram=None, random_state=42):
    print('Indexing word vectors.')
    word_embeddings_index = {}
    with open(word_embed_path) as f:
        for line in f:
            values = line.split()
            word = str(values[0])
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings_index[word] = coefs
    print('Found %d word vectors.' % len(word_embeddings_index))

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

    # add n-gram features
    if n_gram is not None and n_gram > 1:
        print('add ngram words, ngram =', n_gram)

        def add_ngram_words(input_str, ngram_value=2):
            """
            Extract a set of n-grams from a list of integers.
            """
            input_list = input_str.split(' ')
            if ngram_value == 2:
                ngram_list = ['{}_{}'.format(input_list[i], input_list[i + ngram_value - 1]) \
                              for i in range(len(input_list) - ngram_value + 1)]
            elif ngram_value == 3:
                ngram_list = ['{}_{}_{}'.format(input_list[i], input_list[i + ngram_value - 2], input_list[i + ngram_value - 1]) \
                    for i in range(len(input_list) - ngram_value + 1)]
            else:
                ngram_list = []

            for ngram_words in ngram_list:
                words = ngram_words.split('_')
                if n_gram == 2:
                    n_gram_embed = (word_embeddings_index[words[0]] + word_embeddings_index[words[1]]) / 2.0
                elif n_gram == 3:
                    n_gram_embed = (word_embeddings_index[words[0]] + word_embeddings_index[words[1]] + word_embeddings_index[words[2]]) / 3.0
                else:
                    n_gram_embed = np.random.random(size=(embedding_dim,))

                # 添加ngram词汇
                word_embeddings_index[ngram_words] = n_gram_embed

            input_list = input_list + ngram_list
            result = ' '.join(input_list)
            return result

        train['q1_words'] = train['q1_words'].map(lambda words: add_ngram_words(words, ngram_value=n_gram))
        train['q2_words'] = train['q2_words'].map(lambda words: add_ngram_words(words, ngram_value=n_gram))
        test['q1_words'] = test['q1_words'].map(lambda words: add_ngram_words(words, ngram_value=n_gram))
        test['q2_words'] = test['q2_words'].map(lambda words: add_ngram_words(words, ngram_value=n_gram))

        print('Found %d word vectors after ngram.' % len(word_embeddings_index))

    # 数据增强:交换 q1 和 q2
    if use_data_aug:
        print('exchange q1 and q2 to augment training dataset')
        aug_train = train.copy()
        aug_train.columns = ['label', 'id', 'q2_words', 'q2_chars', 'q1_words', 'q1_chars']
        train = pd.concat([train, aug_train.sample(frac=0.5, random_state=random_state)], axis=0)
    # shuffle
    train = train.sample(frac=1, random_state=random_state)

    # 拼接 train 和 test，方便处理
    test['label'] = -1
    all_df = pd.concat([train, test])

    ########################################
    ## tokenize the text and then do padding the sentences
    ########################################
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
    labels = train['label'].values

    print('Shape of question1 train data tensor:', train_q1_words_seqs.shape)
    print('Shape of question2 train data tensor:', train_q2_words_seqs.shape)
    print('Shape of question1 test data tensor:', test_q1_words_seq.shape)
    print('Shape of question1 test data tensor:', test_q2_words_seq.shape)
    print('Shape of label tensor:', labels.shape)
    del q1_words_seqs
    del q2_words_seqs
    gc.collect()

    print('Preparing embedding matrix.')
    # prepare embedding matrix
    nb_words = min(max_nb_words, len(word_index))

    embedding_matrix = np.zeros((nb_words, embedding_dim))
    for word, i in word_index.items():

        if i >= nb_words:
            continue
        word = str(word).upper()
        embedding_vector = word_embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    data = {'nb_words': nb_words, 'embedding_matrix': embedding_matrix, 'labels': labels,
            'train_q1_words_seqs': train_q1_words_seqs, 'train_q2_words_seqs': train_q2_words_seqs,
            'test_q1_words_seq': test_q1_words_seq, 'test_q2_words_seq': test_q2_words_seq}
    return data
