#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/27 下午8:27
"""
import sys
sys.path.append("../")
import functools
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from collections import Counter
from conf.configure import Configure
from utils import data_utils
from collections import defaultdict
import distance


def word_match_share(row, word_char='words'):
    q1words = {}
    q2words = {}
    for word in row['q1_{}'.format(word_char)]:
        q1words[word] = 1
    for word in row['q2_{}'.format(word_char)]:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


def tfidf_word_match_share(row, weights=None, word_char='words'):
    q1words = {}
    q2words = {}
    for word in row['q1_{}'.format(word_char)]:
        q1words[word] = 1
    for word in row['q2_{}'.format(word_char)]:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


def same_start_word(row):
    if not row['q1_words'] or not row['q2_words']:
        return np.nan
    return int(row['q1_words'][0] == row['q2_words'][0])

def same_start_char(row):
    if not row['q1_chars'] or not row['q2_chars']:
        return np.nan
    return int(row['q1_chars'][0] == row['q2_chars'][0])

def total_unique_words(row):
    return len(set(row['q1_words']).union(row['q2_words']))

def total_unique_chars(row):
    return len(set(row['q1_chars']).union(row['q2_chars']))


def q1_freq(row, q_dict):
    return (len(q_dict[row['q1_words']]))

def q2_freq(row, q_dict):
    return (len(q_dict[row['q2_words']]))

def q1_q2_intersect(row, q_dict):
    return (len(set(q_dict[row['q1_words']]).intersection(set(q_dict[row['q2_words']]))))



def build_features1(df, words_weights, chars_weights, q_dict):
    """ 距离特征 """
    print('len')
    df['word_diff_len'] = np.abs(df.q1_words.apply(lambda x: len(str(x).split(' '))) - df.q2_words.apply(lambda x: len(str(x).split(' '))))
    df['char_diff_len'] = np.abs(df.q1_chars.apply(lambda x: len(str(x).split(' '))) - df.q2_chars.apply(lambda x: len(str(x).split(' '))))
    df['word_set_diff_len'] = np.abs(df.q1_words.apply(lambda x: len(set(str(x).split(' ')))) - df.q2_words.apply(lambda x: len(set(str(x).split(' ')))))
    df['char_set_diff_len'] = np.abs(df.q1_chars.apply(lambda x: len(set(str(x).split(' ')))) - df.q2_chars.apply(lambda x: len(set(str(x).split(' ')))))

    df['common_words'] = df.apply(lambda x: len(set(str(x['q1_words']).split(' ')).intersection(set(str(x['q2_words']).split(' ')))), axis=1)
    df['common_chars'] = df.apply(lambda x: len(set(str(x['q1_chars']).split(' ')).intersection(set(str(x['q2_chars']).split(' ')))), axis=1)

    print('fuzz')
    df['words_fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x['q1_words']), str(x['q2_words'])), axis=1)
    df['words_fuzz_WRatio'] = df.apply(lambda x: fuzz.WRatio(str(x['q1_words']), str(x['q2_words'])), axis=1)
    df['words_fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x['q1_words']), str(x['q2_words'])), axis=1)
    df['words_fuzz_partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio(str(x['q1_words']), str(x['q2_words'])), axis=1)
    df['words_fuzz_partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['q1_words']), str(x['q2_words'])), axis=1)
    df['words_fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['q1_words']), str(x['q2_words'])),axis=1)
    df['words_fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x['q1_words']), str(x['q2_words'])), axis=1)

    df['chars_fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x['q1_chars']), str(x['q2_chars'])), axis=1)
    df['chars_fuzz_WRatio'] = df.apply(lambda x: fuzz.WRatio(str(x['q1_chars']), str(x['q2_chars'])), axis=1)
    df['chars_fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x['q1_chars']), str(x['q2_chars'])), axis=1)
    df['chars_fuzz_partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio(str(x['q1_chars']), str(x['q2_chars'])), axis=1)
    df['chars_fuzz_partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['q1_chars']), str(x['q2_chars'])), axis=1)
    df['chars_fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['q1_chars']), str(x['q2_chars'])),axis=1)
    df['chars_fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x['q1_chars']), str(x['q2_chars'])), axis=1)

    f = functools.partial(q1_q2_intersect, q_dict=q_dict)
    df['q1_q2_intersect'] = df.apply(f, axis=1, raw=True)
    f = functools.partial(q1_freq, q_dict=q_dict)
    df['q1_freq'] = df.apply(f, axis=1, raw=True)
    f = functools.partial(q2_freq, q_dict=q_dict)
    df['q2_freq'] = df.apply(f, axis=1, raw=True)

    df['q1_words'] = df['q1_words'].map(lambda x: x.split(' '))
    df['q2_words'] = df['q2_words'].map(lambda x: x.split(' '))

    df['q1_chars'] = df['q1_chars'].map(lambda x: x.split(' '))
    df['q2_chars'] = df['q2_chars'].map(lambda x: x.split(' '))

    print('match')
    f = functools.partial(word_match_share, word_char='words')
    df['word_match'] = df.apply(f, axis=1, raw=True)
    f = functools.partial(word_match_share, word_char='chars')
    df['char_match'] = df.apply(f, axis=1, raw=True)

    f = functools.partial(tfidf_word_match_share, weights=words_weights, word_char='words')
    df['words_tfidf_wm'] = df.apply(f, axis=1, raw=True)
    f = functools.partial(tfidf_word_match_share, weights=chars_weights, word_char='chars')
    df['chars_tfidf_wm'] = df.apply(f, axis=1, raw=True)

    df['same_start_word'] = df.apply(same_start_word, axis=1, raw=True)
    df['same_start_char'] = df.apply(same_start_char, axis=1, raw=True)
    df['total_unique_words'] = df.apply(total_unique_words, axis=1, raw=True)
    df['total_unique_chars'] = df.apply(total_unique_chars, axis=1, raw=True)

    return df


def str_jaccard(str1, str2):
    res = distance.jaccard(str1, str2)
    return res

# shortest alignment
def str_levenshtein_1(str1, str2):
    res = distance.nlevenshtein(str1, str2,method=1)
    return res

# longest alignment
def str_levenshtein_2(str1, str2):
    res = distance.nlevenshtein(str1, str2,method=2)
    return res

def str_sorensen(str1, str2):
    res = distance.sorensen(str1, str2)
    return res

def wmd(model, s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(norm_model, s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(model, words):
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def build_features2(df, word_model, char_model):
    features = pd.DataFrame({'id': df['id']})
    df['q1_words'] = df['q1_words'].map(lambda x: x.split(' '))
    df['q2_words'] = df['q2_words'].map(lambda x: x.split(' '))
    df['q1_chars'] = df['q1_chars'].map(lambda x: x.split(' '))
    df['q2_chars'] = df['q2_chars'].map(lambda x: x.split(' '))

    question1_vectors = np.zeros((features.shape[0], 300))

    for i, q in enumerate(df.q1_words.values):
        question1_vectors[i, :] = sent2vec(word_model, q)

    question2_vectors = np.zeros((df.shape[0], 300))
    for i, q in enumerate(df.q2_words.values):
        question2_vectors[i, :] = sent2vec(word_model, q)

    features['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    features['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    features['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    features['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    features['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    features['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    features['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

    features['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    features['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    features['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    features['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]




    question1_vectors = np.zeros((features.shape[0], 300))

    for i, q in enumerate(df.q1_chars.values):
        question1_vectors[i, :] = sent2vec(char_model, q)

    question2_vectors = np.zeros((df.shape[0], 300))
    for i, q in enumerate(df.q2_chars.values):
        question2_vectors[i, :] = sent2vec(char_model, q)

    features['char_cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    features['char_cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    features['char_jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    features['char_canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    features['char_euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    features['char_minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    features['char_braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

    features['char_skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    features['char_skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    features['char_kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    features['char_kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

    return features


def main():
    print('load datasets')
    questions = pd.read_csv(Configure.question_file)
    train = pd.read_csv(Configure.train_data_file).sample(n=1000)
    test = pd.read_csv(Configure.test_data_file).sample(n=1000)

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

    feature_name = 'basic_features'
    if not data_utils.is_feature_created(feature_name):
        train_words = pd.Series(train['q1_words'].map(lambda x: x.split(' ')).tolist() + train['q2_words'].map(
            lambda x: x.split(' ')).tolist())
        words = [x for y in train_words for x in y]
        counts = Counter(words)
        words_weights = {word: get_weight(count) for word, count in counts.items()}

        train_chars = pd.Series(train['q1_chars'].map(lambda x: x.split(' ')).tolist() + train['q2_chars'].map(
            lambda x: x.split(' ')).tolist())
        chars = [x for y in train_chars for x in y]
        counts = Counter(chars)
        chars_weights = {word: get_weight(count) for word, count in counts.items()}

        ques = pd.concat([train[['q1_words', 'q2_words']], test[['q1_words', 'q2_words']]], axis=0).reset_index(
            drop='index')
        q_dict = defaultdict(set)
        for i in range(ques.shape[0]):
            q_dict[ques.q1_words[i]].add(ques.q2_words[i])
            q_dict[ques.q2_words[i]].add(ques.q1_words[i])

        print('train build_basic_features')
        train_features = build_features1(train, words_weights, chars_weights, q_dict)
        print('test build_basic_features')
        test_features = build_features1(test, words_weights, chars_weights, q_dict)
        data_utils.save_features(train_features, test_features, feature_name)

    feature_name = 'basic_features2'
    if  data_utils.is_feature_created(feature_name):
        print('create gensim model')
        word_model = gensim.models.KeyedVectors.load_word2vec_format(Configure.word_embed_path, binary=False)
        char_model = gensim.models.KeyedVectors.load_word2vec_format(Configure.char_embed_path, binary=False)

        norm_word_model = gensim.models.KeyedVectors.load_word2vec_format(Configure.word_embed_path, binary=False)
        norm_word_model.init_sims(replace=True)

        norm_char_model = gensim.models.KeyedVectors.load_word2vec_format(Configure.char_embed_path, binary=False)
        norm_char_model.init_sims(replace=True)

        print('train build_features2')
        train_features = build_features2(train, word_model, char_model)
        print('test build_features2')
        test_features = build_features2(test, word_model, char_model)
        data_utils.save_features(train_features, test_features, feature_name)



if __name__ == '__main__':
    print '=========== feature engineering ==========='
    main()
