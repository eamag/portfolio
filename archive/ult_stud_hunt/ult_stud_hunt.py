# Fast solution for https://datahack.analyticsvidhya.com/contest/the-ultimate-student-hunt/
# 105/240, spend about 6-8 hours, a lot of visualisation was in temp .ipnb file 

import datetime
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from operator import itemgetter
import copy
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import ELU
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


def load_data():
    train = pd.read_csv('Train_xyqdbho.csv', parse_dates=["Date"])

    train['year'] = train.Date.dt.year
    train['month'] = train.Date.dt.month
    train['dayofyear'] = train.Date.dt.dayofyear
    train['dayofweek'] = train.Date.dt.dayofweek
    train['day'] = train.Date.dt.day
    train["Week"] = train.Date.dt.week
    # columns with a lot of nulls
    train = train.drop(['Date', 'ID', 'Average_Atmospheric_Pressure', 'Max_Atmospheric_Pressure',
                        'Min_Atmospheric_Pressure', 'Min_Ambient_Pollution', 'Max_Ambient_Pollution'], axis=1)
    # getting rid of nulls
    for column in train.columns:
        if train[column].isnull().any():
            train[column].fillna(train[column].mean(), inplace=True)

    train_cv = train[train['year'] > 1999]
    train = train[train['year'] <= 1999]

    target = copy.copy(train['Footfall'])
    target_cv = copy.copy(train_cv['Footfall'])
    train = train.drop('Footfall', axis=1)
    train_cv = train_cv.drop('Footfall', axis=1)

    test = pd.read_csv('Test_pyI9Owa.csv', parse_dates=["Date"])

    test['year'] = test.Date.dt.year
    test['month'] = test.Date.dt.month
    test['dayofyear'] = test.Date.dt.dayofyear
    test['dayofweek'] = test.Date.dt.dayofweek
    test['day'] = test.Date.dt.day
    test["Week"] = test.Date.dt.week
    test_id = copy.copy(test['ID'])
    test = test.drop(['Date', 'ID', 'Average_Atmospheric_Pressure', 'Max_Atmospheric_Pressure',
                      'Min_Atmospheric_Pressure', 'Min_Ambient_Pollution', 'Max_Ambient_Pollution'], axis=1)
    for column in test.columns:
        if test[column].isnull().any():
            test[column].fillna(test[column].mean(), inplace=True)  # getting rid of zeros

    return train, train_cv, target, target_cv, test, test_id


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def print_features_importance(imp):
    for i in range(len(imp)):
        print("# " + str(imp[i][1]))
        print('output.remove(\'' + imp[i][0] + '\')')


def run_fold(train, target, train_cv, target_cv, test, random_state=2016):
    eta = 0.009
    max_depth = 6
    subsample = 0.9
    colsample_bytree = 0.9

    params = {
        "objective": "count:poisson",
        "booster": "gbtree",
        "eval_metric": "rmse",
        "eta": eta,
        "tree_method": 'auto',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 6000
    early_stopping_rounds = 50

    features = train.columns
    dtrain = xgb.DMatrix(train, target)
    watchlist = [(dtrain, 'train')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(train_cv), ntree_limit=gbm.best_iteration + 1)
    rms = np.sqrt(mean_squared_error(target_cv, check))
    print('Check error value: {:.6f}'.format(rms))

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test), ntree_limit=gbm.best_iteration + 1)

    imp = get_importance(gbm=gbm, features=features)
    print_features_importance(imp=imp)
    return test_prediction


def create_sub(test_id, predicted):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('ID,Footfall\n')
    total = 0
    for id in test_id:
        str1 = str(id) + ',' + str(int(predicted[total][0]))
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def get_net():
    model = Sequential()
    model.add(Dense(32, input_dim=16, activation=ELU(), init='he_normal'))
    model.add(Dense(64, activation=ELU(), init='he_normal'))
    model.add(Dense(128, activation=ELU(), init='he_normal'))
    model.add(Dense(64, activation=ELU(), init='he_normal'))
    model.add(Dense(32, activation=ELU(), init='he_normal'))
    model.add(Dense(64, activation=ELU(), init='he_normal'))
    model.add(Dense(128, activation=ELU(), init='he_normal'))
    model.add(Dense(64, activation=ELU(), init='he_normal'))
    model.add(Dense(16, activation=ELU(), init='he_normal'))
    model.add(Dense(1, activation='linear', init='he_normal'))
    model.compile(loss=rmse, optimizer=Adam(lr=9e-3))
    model.summary()
    return model


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


if __name__ == '__main__':
    train, train_cv, target, target_cv, test, test_id = load_data()

    neural = 1
    if neural:
        model = get_net()
        model_checkpoint = ModelCheckpoint('dense.hdf5', monitor='loss', save_best_only=True,
                                           save_weights_only=True)
        model.fit(train.values, target.values, batch_size=4096*2, nb_epoch=85, verbose=1, shuffle=False,
                  callbacks=[model_checkpoint], validation_data=(train_cv.values, target_cv.values))
        predicted = model.predict(test.values, batch_size=256, verbose=1)
        print(predicted)
        create_sub(test_id, predicted)
    else:
        predicted = run_fold(train, train_cv, test)
        create_sub(test_id, predicted)
