import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
import os


def get_processed_data():

    processed_train = pd.read_csv('processed_data/processed_train.csv')
    processed_test = pd.read_csv('processed_data/processed_test.csv')
    processed_train.fillna(0, inplace=True)
    processed_test.fillna(0, inplace=True)
    # processed_train.drop(['Unnamed:0', 'register_time'], axis=1, inplace=True)
    # processed_test.drop(['Unnamed:0', 'register_time'], axis=1, inplace=True)
    return processed_train, processed_test


def train_xgb(processed_train, processed_test):
    predict_data = processed_test['user_id'].copy()
    processed_train_x = processed_train.drop(['Unnamed: 0', 'user_id', 'register_time', 'prediction_pay_price'], axis=1)
    processed_test_x = processed_test.drop(['Unnamed: 0', 'user_id', 'register_time'], axis=1)

    train_dmatrix = xgb.DMatrix(processed_train_x, label=processed_train.prediction_pay_price)
    predict_dmatrix = xgb.DMatrix(processed_test_x)
    
    # xgboost模型训练
    params = {'booster': 'gbtree',
              'eval_metric': 'rmse',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.02,
              # 'tree_method': 'gpu_hist',
              # 'gpu_id': '1',
              # 'n_gpus': '-1',
              'seed': 0,
              'nthread': cpu_jobs,
              # 'predictor': 'gpu_predictor'
              }

    # 使用xgb.cv优化num_boost_round参数
    # cvresult = xgb.cv(params, train_dmatrix, num_boost_round=10000, nfold=2, metrics={'rmse'}, seed=0,
    #                   callbacks=[xgb.callback.print_evaluation(show_stdv=False),
    #                              xgb.callback.early_stop(30)])
    # num_round_best = cvresult.shape[0] - 1
    # print('Best round num: ', num_round_best)

    # 使用优化后的num_boost_round参数训练模型
    watchlist = [(train_dmatrix, 'train')]
    model = xgb.train(params, train_dmatrix, num_boost_round=200, evals=watchlist)

    model.save_model('model/xgb_model1')
    params['predictor'] = 'cpu_predictor'
    model = xgb.Booster(params)
    model.load_model('model/xgb_model1')

    test_predict = predict_data.copy()
    test_predict['label'] = model.predict(predict_dmatrix)

    test_predict.to_csv('predict/xgb_predict_1.csv', index=None, header=None)


if __name__ == '__main__':
    start = datetime.datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))

    cpu_jobs = os.cpu_count() - 1

    processed_train, processed_test = get_processed_data()
    train_xgb(processed_train, processed_test)

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %s s' % (datetime.datetime.now() - start).seconds)

