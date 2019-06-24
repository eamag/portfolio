import numpy as np
import pandas as pd
import random
from sklearn.metrics import r2_score
from sklearn import linear_model, metrics
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc
from tqdm import *

plt.style.use('ggplot')
pd.options.mode.chained_assignment = None


def r_score(y_true, y_pred, sample_weight=None, multioutput=None):
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight,
                  multioutput=multioutput)
    r = (np.sign(r2) * np.sqrt(np.abs(r2)))
    if r <= -1:
        return -1
    else:
        return r


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


import kagglegym

env = kagglegym.make()
o = env.reset()
train = o.train

# train = pd.read_hdf('../input/train.h5')

low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
train = train.loc[y_is_within_cut, :]

d_mean = train.mean()
train["nbnulls"] = train.isnull().sum(axis=1)

rnd = 17

# columns kept for evolution from one month to another (best selected by the tree algorithms)
add_diff_ft = True
diff_cols = ['technical_17', 'technical_30', 'technical_33',
             'technical_11', 'technical_20', 'technical_21',
             'technical_2', 'technical_24', 'technical_41', 'technical_3',
             'technical_19', 'technical_40', 'technical_27', 'technical_6',
             'technical_35', 'technical_1', 'technical_31', 'fundamental_44']


# homemade class used to infer randomly on the way the model learns
# so it basically create output of linear model as new feature
class createLinearFeatures:
    def __init__(self, n_neighbours=1, max_elts=None, verbose=True, random_state=None):
        self.rnd = random_state
        self.n = n_neighbours
        self.max_elts = max_elts
        self.verbose = verbose
        self.neighbours = []
        self.clfs = []

    def fit(self, train, y):
        if self.rnd != None:
            random.seed(self.rnd)
        if self.max_elts == None:
            self.max_elts = len(train.columns)
        # list_vars = list(train.columns)
        list_vars = list(['technical_30_d', 'technical_17', 'technical_30', 'technical_33',
       'technical_11', 'technical_20', 'technical_21', 'technical_20_d',
       'technical_2', 'technical_24', 'technical_41', 'technical_3',
       'technical_19', 'technical_40', 'technical_27', 'technical_6',
       'technical_35', 'technical_1', 'technical_31', 'fundamental_44',
       'technical_44', 'technical_36', 'technical_0', 'fundamental_62',
       'technical_28', 'technical_5', 'technical_25', 'fundamental_0',
       'fundamental_42', 'technical_7'])
        random.shuffle(list_vars)

        lastscores = np.zeros(self.n) + 1e15

        for elt in list_vars[:self.n]:
            self.neighbours.append([elt])
        list_vars = list_vars[self.n:]

        for elt in tqdm(list_vars):  # for each feature
            indice = 0
            scores = []
            for elt2 in self.neighbours:  # for each second feature
                if len(elt2) < self.max_elts:  # if not enough features for regression
                    clf = linear_model.LinearRegression(fit_intercept=False, copy_X=True)
                    clf.fit(train[elt2 + [elt]], y)  # fitting regression and saving score
                    scores.append(r2_score(y, clf.predict(train[elt2 + [elt]])))
                    indice += 1
                else:
                    scores.append(lastscores[indice])
                    indice += 1
                gc.collect()
            gains = lastscores - scores
            if gains.max() > 0:
                temp = gains.argmax()
                lastscores[temp] = scores[temp]
                self.neighbours[temp].append(elt)

        indice = 0
        for elt in self.neighbours:
            clf = linear_model.LinearRegression(fit_intercept=False, copy_X=True, n_jobs=-1)
            clf.fit(train[elt], y)
            self.clfs.append(clf)
            if self.verbose:
                print(indice, lastscores[indice], elt)
            indice += 1

    def transform(self, train):
        indice = 0
        for elt in self.neighbours:
            # this line generates a warning. Could be avoided by working and returning
            # with a copy of train.
            # kept this way for memory management
            train['neighbour' + str(indice)] = self.clfs[indice].predict(train[elt])
            indice += 1
        return train

    def fit_transform(self, train, y):
        self.fit(train, y)
        return self.transform(train)


if add_diff_ft:
    train = train.sort_values(by=['id', 'timestamp'])
    for elt in diff_cols:
        # a quick way to obtain deltas from one month to another but it is false on the first
        # month of each id
        train[elt + "_d"] = train[elt].rolling(2).apply(lambda x: x[1] - x[0]).fillna(0)
        gc.collect()
    # removing month 0 to reduce the impact of erroneous deltas
    train = train[train.timestamp != 0]

bad_cols = ['technical_32', 'fundamental_3', 'fundamental_9', 'technical_34',
            'fundamental_27', 'technical_10', 'technical_18', 'fundamental_28',
            'technical_39', 'fundamental_63', 'fundamental_26',
            'fundamental_61', 'fundamental_1', 'technical_22', 'fundamental_38',
            'fundamental_6', 'technical_9', 'fundamental_57']
more_bad_cols = ['fundamental_27', 'technical_29', 'technical_39', 'fundamental_28',
                 'technical_34', 'neighbour6', 'fundamental_63', 'fundamental_38',
                 'fundamental_1', 'technical_10', 'technical_18', 'technical_16',
                 'technical_34_d', 'technical_22_d', 'fundamental_26',
                 'fundamental_6', 'fundamental_57', 'technical_22', 'technical_9',
                 'fundamental_61']
cols = [x for x in train.columns if x not in (['id', 'timestamp', 'y', 'sample'] + bad_cols + more_bad_cols)]

train = train.fillna(d_mean)

# adding all trees generated by a tree regressor
print("adding new features")
featureexpander = createLinearFeatures(n_neighbours=10, max_elts=5, verbose=True, random_state=rnd)
featureexpander.fit(train[cols], train['y'])
trainer = featureexpander.transform(train[cols])
treecols = trainer.columns

lastvalues = train[train.timestamp == 905][['id'] + diff_cols].copy()

verbose = 1
plot = 0
submit = 0

params = {
    'task': 'train',
    'boosting_type': 'dart',
    'objective': 'regression',
    'verbose': 0,
    # 'num_leaves': 63,
    'learning_rate': 0.005,
    # 'feature_fraction': 0.8,
    # 'bagging_fraction': 0.7,
    # 'bagging_freq': 10,
    'max_bin': 127,
    'lambda_l2': 1,
    'max_depth': 4
}

if not submit:
    X_train, X_test, y_train, y_test = train_test_split(trainer, train.y, test_size=0.2, random_state=0)
    train_data = lgb.Dataset(X_train, y_train)
    gc.collect()
    bst = lgb.train(params, train_data, feval=r_score, verbose_eval=verbose)

    df_fi = pd.DataFrame(bst.feature_name(), columns=['feature'])
    df_fi['importance'] = list(bst.feature_importance('gain'))
    df_fi.sort_values('importance', ascending=False, inplace=True)
    print_full(df_fi)
    if plot:
        plt.figure()
        df_fi.head(10).plot(kind='barh',
                            x='feature',
                            y='importance',
                            sort_columns=False,
                            legend=False,
                            figsize=(10, 6),
                            facecolor='#1DE9B6',
                            edgecolor='white')

        plt.title('LightGBM Feature Importance')
        plt.xlabel('relative importance')
        plt.show()

    if verbose:
        y_pred = bst.predict(X_test)
        print('\nScore for another fold: ', r_score(y_test, y_pred))
else:
    train_data = lgb.Dataset(trainer, train.y)
    gc.collect()
    bst = lgb.train(params, train_data, feval=r_score)

print("end of training, now predicting")
print('indice, countplus, reward, np.mean(rewards), info["public_score"]')
indice = 0
countplus = 0
rewards = []
while True:
    indice += 1
    test = o.features
    test["nbnulls"] = test.isnull().sum(axis=1)
    test = test.fillna(d_mean)

    pred = o.target
    if add_diff_ft:
        # creating deltas from lastvalues
        indexcommun = list(set(lastvalues.id) & set(test.id))
        lastvalues = pd.concat([test[test.id.isin(indexcommun)]['id'],
                                pd.DataFrame(test[diff_cols][test.id.isin(indexcommun)].values - lastvalues[diff_cols][
                                    lastvalues.id.isin(indexcommun)].values,
                                             columns=diff_cols, index=test[test.id.isin(indexcommun)].index)],
                               axis=1)
        # adding them to test data
        test = test.merge(right=lastvalues, how='left', on='id', suffixes=('', '_d')).fillna(0)
        # storing new lastvalues
        lastvalues = test[['id'] + diff_cols].copy()

    testid = test.id
    test = featureexpander.transform(test[cols])

    pred['y'] = bst.predict(test.loc[:, treecols])

    indexbase = pred.index
    pred.index = testid
    oldpred = pred['y']
    pred.index = indexbase

    o, reward, done, info = env.step(pred)
    rewards.append(reward)
    if reward > 0:
        countplus += 1

    if indice % 100 == 0:
        print(indice, countplus, reward, np.mean(rewards))

    if done:
        print(info["public_score"])
        break

# todo:cluster ids and columns
# create linear for:
# y                 1.000000
# derived_1         0.619237
# technical_24      0.563498
# fundamental_17    0.533610
# derived_2         0.450020
# fundamental_61    0.449296
# fundamental_51    0.375551
# technical_3       0.371798
# id                0.327293
# timestamp         0.283616
# fundamental_27    0.278856
# fundamental_11    0.270736
# fundamental_9     0.210841
# technical_0       0.200569
# fundamental_56    0.192394
# fundamental_25    0.189558
# fundamental_49    0.186698
# technical_37      0.183247
# fundamental_55    0.177027
# fundamental_53    0.164005
# fundamental_38    0.161427
# fundamental_5     0.160513
# technical_41      0.147678
# fundamental_59    0.138768
# fundamental_6     0.137772
# fundamental_7     0.118249
# fundamental_2     0.115978
# technical_12      0.114265
# fundamental_43    0.103725
# fundamental_10    0.101221
