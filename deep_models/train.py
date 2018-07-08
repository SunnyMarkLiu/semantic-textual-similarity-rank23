#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/25 下午3:54
"""
import os
import sys

sys.path.append("../")

import warnings
warnings.filterwarnings('ignore')

import importlib
from utils import data_loader
from tensorflow import flags
from conf.configure import Configure

# disable TF debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

flags.DEFINE_string('model', 'representation_based.DSSM', "path of the Class for the classifier")
flags.DEFINE_integer('fold', 5, "run out of fold")
flags.DEFINE_integer('batch_size', 64, "training batch size")
flags.DEFINE_integer('predict_batch_size', 64, "predict batch size")
flags.DEFINE_float('lr_drop_epoch', 5.0, "every x epoch then drop learning rate")
flags.DEFINE_integer('random_state', 1000, "random_state")
flags.DEFINE_bool('use_tensorbord', False, "use tensorbord to check model")
FLAGS = flags.FLAGS

def main():
    cfg = Configure()
    print('============= params=============')
    print('param:', cfg.params_to_string() + '_fold{}'.format(FLAGS.fold))
    print('model:', FLAGS.model)

    print('\n===> load dataset and prepare for model inputs')
    data = data_loader.load_datas(
        word_embed_path=cfg.word_embed_path, question_file=cfg.question_file,
        train_data_file=cfg.train_data_file, test_data_file=cfg.test_data_file,
        max_nb_words=cfg.max_nb_words, max_sequence_length=cfg.max_sequence_length,
        embedding_dim=cfg.embedding_dim,

        char_embed_path=cfg.char_embed_path, max_nb_chars=cfg.max_nb_chars, max_seq_chars_length=cfg.max_seq_chars_length
    )

    # create model
    print('===> create model')
    cls_name = FLAGS.model
    module_name = ".".join(cls_name.split('.')[:-1])
    cls_name = cls_name.split('.')[-1]
    _module = importlib.import_module(module_name)
    cls = _module.__dict__.get(cls_name)

    model = cls(data=data, cfg=cfg, lr_drop_epoch=FLAGS.lr_drop_epoch, model_name=cls_name,
                engineer_feature_count=data['train_features'].shape[1])
    print('===> train and predict')
    model.train_and_predict(roof=True, fold=FLAGS.fold, batch_size=FLAGS.batch_size,
                            predict_batch_size=FLAGS.predict_batch_size,
                            random_state=FLAGS.random_state)
    print('done')

if __name__ == '__main__':
    main()
