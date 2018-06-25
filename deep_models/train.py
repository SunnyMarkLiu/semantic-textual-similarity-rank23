#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/25 下午3:54
"""
import os
import sys

sys.path.append("../")
# 文本处理
import warnings

warnings.filterwarnings('ignore')
from utils import data_loader
from multi_dssm import DSSM
from conf.configure import Configure

# disable TF debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

cfg = Configure()


def main():
    print('============= params=============')
    print(cfg.params_to_string())

    print('\n===> load dataset and prepare for model inputs')
    data = data_loader.load_datas(
        word_embed_path=cfg.word_embed_path, question_file=cfg.question_file,
        train_data_file=cfg.train_data_file, test_data_file=cfg.test_data_file,
        max_nb_words=cfg.max_nb_words, max_sequence_length=cfg.max_sequence_length,
        embedding_dim=cfg.embedding_dim, use_data_aug=cfg.use_data_aug,
        aug_frac=cfg.aug_frac, random_state=cfg.random_state, n_gram=cfg.n_gram
    )

    # create model
    print('===> create model')
    model = DSSM(data, cfg, model_name='DSSM')
    print('===> train and predict')
    model.train_and_predict(roof=True)
    print('done')

if __name__ == '__main__':
    main()
