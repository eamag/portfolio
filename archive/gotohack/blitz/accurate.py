# top 10 fastest users, reached 24 points

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from pylightgbm.models import GBMClassifier
from sklearn.metrics import f1_score

events = pd.DataFrame.from_csv("course-217-events.csv", index_col=None)
struc =  pd.DataFrame.from_csv("course-217-structure.csv", index_col=None)


def top10():
    new_t = events.merge(struc[['step_id', 'step_cost']], on='step_id')
    del new_t['action']

    min_time = new_t.groupby('user_id').time.min()
    min_time = pd.DataFrame(min_time)
    min_time['user_id'] = min_time.index

    new_t.sort_values(by=['user_id', 'time'], inplace=True)
    new_t = new_t.drop_duplicates(subset=['step_id', 'user_id'])
    new_t = new_t[new_t['step_cost'] == 1 ]
    k = []
    for i in new_t.groupby('user_id'):
        if i[1].iloc[:, 3].sum() >23:
            k.append([i[0], i[1].iloc[23, 2]])
        k = pd.DataFrame(k, columns=['user_id', 'time'])

    res = pd.merge(k, min_time, on='user_id', how='inner')
    res['time'] = res['time_x'] - res['time_y']
    print(res.sort_values(by=['time']).iloc[0:10, 0].values)

if __name__ == '__main__':
    top10()
