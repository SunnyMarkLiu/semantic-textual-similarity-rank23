#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/27 下午8:27
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


def load_dataset():
    """ 加载 level1 的 roof 的预测结果 """
    train_df = pd.read_csv('../input/train.csv')

    base = '../result/ensemble/'
    roof_result = os.listdir(base)
    train = pd.DataFrame({'label': train_df['label']})
    test = pd.DataFrame()
    for f in roof_result:
        feature = '_'.join(f[:-4].split('_')[1:])
        if f.startswith('test'):
            test_i = pd.read_csv(base + f)
            test[feature] = test_i['y_pre']
        else:
            train_i = pd.read_csv(base + f)
            train[feature] = train_i['y_pre']

    train = train[['label'] + test.columns.values.tolist()]

    return train, test

def main():
    print("load train test datasets")
    train, test = load_dataset()
    y_train_all = train['label']

    train.drop(['label'], axis=1, inplace=True)

    df_columns = train.columns.values
    print('train: {}, test: {}, feature count: {}, label 1:0 = {:.5f}'.format(
        train.shape[0], test.shape[0], len(df_columns), 1.0 * sum(y_train_all) / len(y_train_all)))

    xgb_params = {
        'eta': 0.01,
        'max_depth': 6,
        'subsample': 0.9,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'updater': 'grow_gpu',
        'gpu_id': 1,
        'nthread': -1,
        'silent': 1,
        'booster': 'gbtree',
    }

    dtest = xgb.DMatrix(test)

    pred_train_full = np.zeros(train.shape[0])
    pred_test_full = 0
    cv_scores = []
    roof_flod = 5

    kf = StratifiedKFold(n_splits=roof_flod, shuffle=True, random_state=42)
    for i, (dev_index, val_index) in enumerate(kf.split(train, y_train_all)):
        print('========== perform fold {}, train size: {}, validate size: {} =========='.format(i, len(dev_index),
                                                                                                len(val_index)))
        train_x, val_x = train.ix[dev_index], train.ix[val_index]
        train_y, val_y = y_train_all[dev_index], y_train_all[val_index]
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dval = xgb.DMatrix(val_x, label=val_y)

        model = xgb.train(xgb_params, dtrain,
                          evals=[(dtrain, 'train'), (dval, 'valid')],
                          verbose_eval=20,
                          early_stopping_rounds=100,
                          num_boost_round=4000)

        # predict validate
        predict_valid = model.predict(dval, ntree_limit=model.best_ntree_limit)
        valid_auc = log_loss(val_y, predict_valid, eps=1e-10)
        # predict test
        predict_test = model.predict(dtest, ntree_limit=model.best_ntree_limit)

        print('valid_logloss = {}'.format(valid_auc))
        cv_scores.append(valid_auc)

        # run-out-of-fold predict
        pred_train_full[val_index] = predict_valid
        pred_test_full += predict_test

    mean_cv_scores = np.mean(cv_scores)
    print('Mean cv logloss:', mean_cv_scores)

    print("saving test predictions for ensemble")
    pred_test_full = pred_test_full / float(roof_flod)
    test_pred_df = pd.DataFrame({'y_pre': pred_test_full})
    test_pred_df.to_csv("level2_stacking_{}_models_cv{}.csv".format(test.shape[1], mean_cv_scores),
        index=False, columns=['y_pre'], sep='\t')

    print('-------- predict and check  ------')
    print('test 1 count: {:.6f}, mean: {:.6f}'.format(np.sum(test_pred_df['y_pre']), np.mean(test_pred_df['y_pre'])))
    print('done.')


if __name__ == '__main__':
    print('========== xgboost stacking ==========')
    main()

