import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from pylightgbm.models import GBMClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def load_d():
    x = pd.DataFrame.from_csv("user_activity.csv", index_col=None)
    targets = pd.DataFrame.from_csv("targets.csv", index_col=None)
    x_test = pd.DataFrame.from_csv("user_activity_test.csv", index_col=None)
    Y = targets.passed.values

    all_data = pd.concat([x, x_test])
    for column in all_data:
        if all_data[column].dtype == 'object':
            all_data = pd.concat([all_data, pd.get_dummies(all_data[column])], axis=1)
            all_data.drop(column, axis=1, inplace=True)
    all_data.drop('step_id', axis=1, inplace=True)

    train = all_data[:x.shape[0]]
    test = all_data[train.shape[0]:]
    train.sort_values(by=['user_id', 'time'], inplace=True)
    test.sort_values(by=['user_id', 'time'], inplace=True)

    train, ind_tr = feat_extract(train)
    test, ind_test = feat_extract(test)

    return train, ind_tr, test, ind_test, Y


def feat_extract(df):
    grouped = df.groupby('user_id')
    new_df = grouped.sum()
    index = new_df.index
    new_df.drop('time', axis=1, inplace=True)
    new_df = pd.concat([new_df, grouped['time'].agg([np.ptp])], axis=1)
    return new_df, index


def lgbm(X, Y, verbose=True, predict=False, x_test=None, kf=True, feat_imp=True):
    feature_list = X.columns
    feature_dict = dict(zip(range(len(feature_list)), feature_list))
    X = X.values

    exec = "~/LightGBM/lightgbm"  # full path to lightgbm executable (on Windows include .exe)
    clf = GBMClassifier(num_iterations=200, exec_path=exec, learning_rate=0.2, min_data_in_leaf=140, verbose=0,
                        max_bin=255)

    if kf:
        kf = KFold(n_splits=7
                   , shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            clf.fit(X_train, y_train, test_data=[(X_test, y_test)])
            if verbose:
                y_pred = clf.predict(X_test)
                print('\nScore for another fold: ', f1_score(y_test, y_pred))

    if predict:
        clf.fit(X, Y)
        print('\nFinal result for train set is ', f1_score(Y, clf.predict(X)))
        if feat_imp:
            df_fi = pd.DataFrame(list(clf.feature_importance().items()), columns=['feature', 'importance'])
            df_fi = df_fi.replace({"feature": feature_dict}).sort_values('importance', ascending=False)
            print(df_fi)
            plt.figure()
            df_fi.head(10).plot(kind='barh',
                                x='feature',
                                y='importance',
                                sort_columns=False,
                                legend=False,
                                figsize=(10, 6),
                                facecolor='#1DE9B6',
                                edgecolor='white')

            plt.title('XGBoost Feature Importance')
            plt.xlabel('relative importance')
            plt.show()
        return clf.predict(x_test.values)


def create_submission(X, name):
    np.savetxt('%s.csv' % name, X, delimiter=',', fmt="%d", header='user_id,passed', comments='')


if __name__ == '__main__':
    train, ind_tr, test, ind_test, Y = load_d()
    result = lgbm(train, Y, verbose=True, predict=True, x_test=test, kf=False)
    submit = np.concatenate((np.asarray(ind_test, dtype=int).reshape(-1, 1),
                             np.asarray(result, dtype=int).reshape(-1, 1)), axis=1)
    # create_submission(submit, 'lgbm')
