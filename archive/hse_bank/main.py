# Small task for HSE course

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from pylightgbm.models import GBMClassifier
import datetime
import numpy as np
import random
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

random.seed(2016)
np.random.seed(2016)


def load_data():
    x = pd.read_csv('train_data.csv')
    y = pd.read_csv('train_target.csv', header=None)
    y = y.values.ravel()
    x_test = pd.read_csv('test_data.csv')

    all_data = pd.concat([x, x_test])
    for column in all_data:
        encoder = LabelEncoder()
        if all_data[column].dtype == 'object':
            all_data[column] = encoder.fit_transform(all_data[column])
    all_data.drop(['Unnamed: 0'], axis=1, inplace=True)
    train = all_data[:x.shape[0]]
    test = all_data[train.shape[0]:]

    return train, y, test


def simple_logreg(X, y, verbose=True):
    cv = KFold(n_splits=7, shuffle=True)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    gs = GridSearchCV(cv=cv, estimator=LogisticRegression(n_jobs=7), param_grid=param_grid)
    gs.fit(X, y)
    if verbose:
        print(gs.score(X, y))
        print(roc_auc_score(y, gs.predict(X)))


def lgbm(X, Y, verbose=True, predict=False, x_test=None, y_test=None, kf=True):
    """
Install this: https://github.com/Microsoft/LightGBM/wiki/Installation-Guide and https://github.com/ArdalanM/pyLightGBM
    """
    X = X.values

    exec = "~/LightGBM/lightgbm"  # full path to lightgbm executable (on Windows include .exe)

    clf = GBMClassifier(num_iterations=20, exec_path=exec, learning_rate=0.2,min_data_in_leaf=140, verbose=verbose,
                        max_bin=440)

    if kf:
        kf = KFold(n_splits=2, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train, test_data=[(X_test, y_test)])
            if verbose:
                y_pred = clf.predict_proba(X_test)[:, 1]
                print('\nScore for another fold: ', roc_auc_score(y_test, y_pred))

    if predict:
        clf.fit(X, Y)
        print('\nFinal result for train set is ', roc_auc_score(Y, clf.predict_proba(X)[:, 1]))
        return clf.predict_proba(x_test.values)[:, 1]


def xgboosting(X, Y, verbose=True, predict=False, x_test=None, kf=True):
    X = X.values
    model = XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=1000)

    if kf:
        kf = KFold(n_splits=2, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train, verbose=verbose)
            if verbose:
                print("Validating...")
                check = model.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, check)
                print('Check error value: {:.6f}'.format(score))

    if predict:
        model.fit(X, Y)
        print('\nFinal result for train set is ', roc_auc_score(Y, model.predict_proba(X)[:, 1]))
        print("Predict test set...")
        return model.predict_proba(x_test)[:, 1]


def submission(prediction):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('Id,Prediction\n')
    total = 0
    for id in pd.read_csv('test_data.csv').iloc[:, 0]:
        str1 = str(id) + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def find_alpha(res1, res2, y_test):
    scores = []
    for alpha in np.linspace(0.01, 0.99, 99):
        p = res1 * alpha + res2 * (1 - alpha)
        score = roc_auc_score(y_test, p)
        scores.append((score, alpha))

    print(max(scores))
    alpha = max(scores)[1]
    print(alpha)
    return alpha


def blend(l, x, alpha):
    final = l * alpha + x * (1 - alpha)
    return final


if __name__ == '__main__':
    x, y, x_test = load_data()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # simple_logreg(x, y)
    result_l = lgbm(X_train, y_train, verbose=False, predict=True, x_test=X_test, y_test=y_test, kf=False)
    result_x = xgboosting(X_train, y_train, verbose=True, predict=True, x_test=X_test, kf=False)
    print(roc_auc_score(y_test, result_l), roc_auc_score(y_test, result_x))
    alpha = find_alpha(result_l, result_x, y_test)

    resl = lgbm(x, y, verbose=False, predict=True, x_test=x_test, kf=False)
    resx = xgboosting(x, y, verbose=True, predict=True, x_test=x_test, kf=False)
    submission(blend(resl, resx, alpha))
