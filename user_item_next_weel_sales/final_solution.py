import pandas as pd
import numpy as np
import catboost as ctb
from sklearn import metrics

def add_neg_sampling(train):
    t_len = 1561 * 10
    for i in range(49):
        neg_sample_data = np.stack([np.random.randint(0, 2000, t_len), np.random.randint(0, 40, t_len),
                                    np.full(t_len, i), np.zeros(t_len), np.zeros(t_len)]).T
        train = train.append(pd.DataFrame(neg_sample_data,
                                          columns=['i', 'j', 't', 'price', 'advertised']), ignore_index=True,
                             sort=False)

    train = train.drop_duplicates(['i', 'j', 't'], keep='first').sort_values(['t', 'i']).reset_index(drop=True)
    return train
def create_val_ds(train):
    val = train[train.t == train.t.max()]
    empty_data = (np.stack([np.arange(0, 2000).repeat(40), np.tile(np.arange(0, 40), 2000),
                            np.full(80_000, 48), np.zeros(80_000), np.zeros(80_000)]).T)
    empty_df = pd.DataFrame(empty_data, columns=val.columns)
    val = val.append(empty_df, ignore_index=True, sort=False).drop_duplicates(['i', 'j', 't'], keep='first')
    val = val.sort_values(['t', 'i', 'j']).reset_index(drop=True)
    return val

def fit_val_model(train, val):
    train = train[train.t != 48]

    cat_features_idxs = [0, 1, 3]

    train.price = (train.price > 0).astype(int)
    val.price = (val.price > 0).astype(int)

    train_pool = ctb.Pool(train.drop(columns=['price']), train.price, cat_features_idxs)
    validation_pool = ctb.Pool(val.drop(columns=['price']), val.price, cat_features_idxs)

    params = {
        'loss_function': 'Logloss',
        'iterations': 300,
        'early_stopping_rounds': 50,
        'custom_metric': ['Accuracy']
    }
    model = ctb.CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=validation_pool, verbose=True, plot=False)

    res = model.predict_proba(val.drop(columns=['price']))[:, 1]
    print(f'roc auc is{metrics.roc_auc_score(val.price.tolist(), res)}')
    return model

def fit_prod_model(train):
    cat_features_idxs = [0, 1, 3]  # rerun train val part before
    train.price = (train.price > 0).astype(int)
    train_pool = ctb.Pool(train.drop(columns=['price']), train.price, cat_features_idxs)
    params = {
        'loss_function': 'Logloss',
        'iterations': 26,
        'custom_metric': ['Accuracy']
    }
    model = ctb.CatBoostClassifier(**params)
    model.fit(train_pool, verbose=True, plot=False)
    return model
def save_pred_file(val, model, name_prefix):
    val = val.sort_values(['i', 'j']).reset_index(drop=True)
    val.t = 49
    val.loc[val.j == 24, 'advertised'] = 1
    val = val.drop(columns=['price'])
    res = model.predict(val)
    val['prediction'] = res
    val[['i', 'j', 'prediction']].to_csv(f'{name_prefix}_prediction_roc_0.83.csv')
def create_pred_file():
    train = pd.read_csv('train.csv')
    train = add_neg_sampling(train)
    val = create_val_ds(train)
    model_val = fit_val_model(train, val)
    model = fit_prod_model(train)
    save_pred_file(val, model, 'prod')
    save_pred_file(val, model_val, 'val')

if __name__ == '__main__':
    create_pred_file()