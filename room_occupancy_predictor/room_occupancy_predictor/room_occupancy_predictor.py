import pandas as pd
import numpy as np
import re
import sys
import catboost as ctb
from sklearn import metrics
from catboost.utils import select_threshold
from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class RoomOccupancyPredictor:
    def __init__(self, data_path):
        self.models = self.init_models(data_path)

    def init_models(self, data_path):
        df = pd.read_csv(data_path, parse_dates=["time"])
        models = {}
        for device in df.device.unique():
            df2 = df[df.device == device]
            train, y_train, val, y_val = self.create_train_val_sets(df2)
            models[device] = self.fit_model(train, y_train, val, y_val)
        return models

    @staticmethod
    def add_datepart(df, fldname, drop=True):
        """add_datepart converts a column of df from a datetime64 to many columns containing
        the information from the date. This applies changes inplace.

        Parameters:
        -----------
        df: A pandas data frame. df gain several new columns.
        fldname: A string that is the name of the date column you wish to expand.
            If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
        drop: If true then the original date column will be removed.
        time: If true time features: Hour, Minute, Second will be added.

        """
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
        targ_pre = re.sub("[Dd]ate$", "", fldname)
        attr = [
            "Is_month_end",
            "Is_month_start",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
            "DayOfWeek",
            "Hour",
        ]
        for n in attr:
            df[targ_pre + n] = getattr(fld.dt, n.lower())
        if drop:
            df.drop(fldname, axis=1, inplace=True)

    def create_train_val_sets(self, df2):
        df2 = (
            df2.set_index("time")
            .groupby(pd.Grouper(freq="H"))["device_activated"]
            .sum()
            .reset_index()
        )
        df2.device_activated = df2.device_activated.map(lambda x: 1 if x > 0 else 0)
        split_point = -len(df2) // 10 * 2
        train = df2[:split_point]
        val = df2[split_point:]
        self.add_datepart(train, "time")
        self.add_datepart(val, "time")
        y_train = train["device_activated"].copy()
        train = train.drop(columns=["device_activated"])
        y_val = val["device_activated"].copy()
        val = val.drop(columns=["device_activated"])
        return train, y_train, val, y_val

    @staticmethod
    def fit_model(train, y_train, val, y_val):
        categorical_features_indices = np.where(train.dtypes != np.float)[0]
        train_pool = ctb.Pool(train, y_train, categorical_features_indices)
        validation_pool = ctb.Pool(val, y_val, categorical_features_indices)
        params = {
            "loss_function": "Logloss",
            "iterations": 500,
            "early_stopping_rounds": 50,
            "custom_metric": ["AUC", "Accuracy"],
            "cat_features": categorical_features_indices,
        }
        model = ctb.CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=validation_pool, verbose=False)
        print(
            f"Model roc_auc: {metrics.roc_auc_score(y_val.tolist(), model.predict_proba(val)[:, 1])}"
        )
        # We don't want to turn off the heating when people are in the office, so we set False Negative Rate to 5%
        threshold = select_threshold(model, validation_pool, FNR=0.05)
        return model, threshold

    # noinspection PyUnresolvedReferences
    def predict_24_hours(self, current_time, model, threshold):
        next_24_hours = pd.date_range(current_time, periods=24, freq="H").ceil("H")
        test = pd.DataFrame(next_24_hours, columns=["time"])
        self.add_datepart(test, "time")
        pred = model.predict_proba(test)[:, 1]
        test = pd.DataFrame(next_24_hours, columns=["time"])
        test["activation_predicted"] = pred
        test["activation_predicted"] = (
            test["activation_predicted"] > threshold
        ).astype(int)
        return test

    def output_df_single_device(self, df2, timestamp):
        train, y_train, val, y_val = self.create_train_val_sets(df2)
        model, threshold = self.fit_model(train, y_train, val, y_val)
        out_df_single = self.predict_24_hours(timestamp, model, threshold)
        return out_df_single

    def create_output_df(self, timestamp):
        out_df = pd.DataFrame(columns=["time", "activation_predicted", "device"])
        for device, (model, threshold) in self.models.items():
            out_df_single = self.predict_24_hours(timestamp, model, threshold)
            out_df_single["device"] = device
            out_df = pd.concat([out_df, out_df_single], ignore_index=True)
        return out_df.sort_values(["time", "device"]).reset_index(drop=True)


if __name__ == "__main__":
    timestamp, in_file_path, out_file_path = sys.argv[1:]
    out_df = RoomOccupancyPredictor(in_file_path).create_output_df(timestamp)
    out_df.to_csv(out_file_path, index=False)
