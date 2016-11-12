# Small task for HSE course, collaboration with ml beginner
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from pylightgbm.models import GBMRegressor, GBMClassifier
import datetime


def load_data():
    x = pd.read_csv('train_data.csv')
    y = pd.read_csv('train_target.csv', header=None)
    x_test = pd.read_csv('test_data.csv')
    for i in range(1, 10):
        x[x.columns[i]] = x[x.columns[i]].str.lower()
        x[x.columns[i]].replace('[^a-zA-Z0-9]', ' ', regex=True)
        x[x.columns[i]].fillna('nan', inplace=True)
        x[x.columns[i]] = pd.get_dummies(x[x.columns[i]])
        x_test[x_test.columns[i + 1]] = x_test[x_test.columns[i + 1]].str.lower()
        x_test[x_test.columns[i + 1]].replace('[^a-zA-Z0-9]', ' ', regex=True)
        x_test[x_test.columns[i + 1]].fillna('nan', inplace=True)
        x_test[x_test.columns[i + 1]] = pd.get_dummies(x_test[x_test.columns[i + 1]])
    x[x.columns[14]].fillna('nan', inplace=True)
    x[x.columns[14]] = x[x.columns[14]].str.lower()
    x[x.columns[14]].replace('[^a-zA-Z0-9]', ' ', regex=True)
    x[x.columns[14]] = pd.get_dummies(x[x.columns[14]])
    x_test[x_test.columns[15]].fillna('nan', inplace=True)
    x_test[x_test.columns[15]] = x_test[x_test.columns[15]].str.lower()
    x_test[x_test.columns[15]].replace('[^a-zA-Z0-9]', ' ', regex=True)
    x_test[x_test.columns[15]] = pd.get_dummies(x_test[x_test.columns[15]])
    y = y.values.ravel()
    return x, y, x_test


def simple_logreg(X, y, verbose=True):
    cv = KFold(n_splits=7, shuffle=True)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    gs = GridSearchCV(cv=cv, estimator=LogisticRegression(n_jobs=7), param_grid=param_grid)
    gs.fit(X, y)
    if verbose:
        print(gs.score(X, y))
        print(roc_auc_score(y, gs.predict(X)))


def lgbm(X, Y, verbose=True, predict=False, x_test=None):
    """
Install this: https://github.com/Microsoft/LightGBM/wiki/Installation-Guide and https://github.com/ArdalanM/pyLightGBM
    """

    X = X.values

    exec = "~/LightGBM/lightgbm"  # full path to lightgbm executable (on Windows include .exe)
    # clf = GBMClassifier(num_iterations=30, exec_path=exec, min_data_in_leaf=1, verbose=verbose)
    clf = GBMRegressor(exec_path=exec, num_iterations=100, early_stopping_round=10,
                       num_leaves=10, min_data_in_leaf=10, verbose=verbose)

    kf = KFold(n_splits=7, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train, test_data=[(X_test, y_test)])
        y_pred = clf.predict(X_test)
        if verbose:
            print('\nScore for another fold: ', roc_auc_score(y_test, y_pred))

    print(X, Y)
    print('\nFinal result for train set is ', roc_auc_score(Y, clf.predict(X)))
    if predict:
        return clf.predict(x_test)


def submission(test, prediction):
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('Id,Prediction\n')
    total = 0
    for id in test.iloc[:, 0]:
        str1 = str(id) + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


if __name__ == '__main__':
    x, y, x_test = load_data()
    # simple_logreg(x, y)
    result = lgbm(x, y, verbose=False, predict=True, x_test=x_test)
    submission(x_test, result)