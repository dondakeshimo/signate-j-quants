"""stockに関するビジネスルール
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from .feature_interface import FeatureInterface

FRIDAY = 4


@dataclass
class StockConfig:
    quote_column: str = "EndOfDayQuote ExchangeOfficialClose"
    buffer_day: int = 90  # 計算用に確保するstart_dtからのバッファ日数
    calc_days: List[int] = field(default_factory=list)
    start_dt: str = "2019-02-01"
    fin_columns: List[str] = field(default_factory=list)

    def load_config(self, path: str):
        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.quote_column = config["quote_column"]
        self.buffer_day = config["buffer_day"]
        self.calc_days = config["calc_days"]
        self.fin_columns = config["fin_columns"]


class Stock(FeatureInterface):
    def __init__(self, config: StockConfig) -> None:
        self._df: pd.DataFrame = None
        self.conf: StockConfig = config

    def load_data(self, inputs: Dict[str, str]) -> None:
        self.list_df = pd.read_csv(inputs["stock_list"])
        self.price_df = pd.read_csv(inputs["stock_price"])
        self.fin_df = pd.read_csv(inputs["stock_fin"])
        self.fin_price_df = pd.read_csv(inputs["stock_fin_price"])
        self.labels_df = pd.read_csv(inputs["stock_labels"])

        # NOTE: コピー要るかわからないけどチュートリアルを踏襲
        stock_list = self.list_df.copy()
        self._codes = stock_list[stock_list["universe_comp2"] ==
                                 True]["Local Code"].values
        self._codes_for_train = stock_list[stock_list["prediction_target"] == True]["Local Code"].values

    def preprocess(self) -> None:
        self._set_date_index_all_df()

    def extract_feature(self) -> pd.DataFrame:
        buff = []
        for code in self.codes:
            buff.append(self._extract_feature_weekofday_by_code(code))

        self._df = pd.concat(buff)
        return self._df

    def get_features_and_label(self, label: str, train_end: str, val_start: str, val_end: str,
                               test_start: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                    pd.DataFrame, pd.DataFrame, pd.DataFrame):
        buff = []
        for code in self._codes_for_train:
            buff.append(self._extract_feature_for_train_by_code(code))

        self._df = pd.concat(buff)

        trains_X, vals_X, tests_X = [], [], []
        trains_y, vals_y, tests_y = [], [], []
        for code in self._codes_for_train:
            feats_df = self._df[self._df["code"] == code]
            labels_df = self.labels_df[self.labels_df["Local Code"] == code]
            labels_df = labels_df[label].copy()
            labels_df.dropna(inplace=True)

            if feats_df.shape[0] == 0 or labels_df.shape[0] == 0:
                continue

            labels_df = labels_df.loc[labels_df.index.isin(feats_df.index)]
            feats_df = feats_df.loc[feats_df.index.isin(labels_df.index)]
            labels_df.index = feats_df.index

            _train_X = feats_df[:train_end]
            _val_X = feats_df[val_start:val_end]
            _test_X = feats_df[test_start:]
            trains_X.append(_train_X)
            vals_X.append(_val_X)
            tests_X.append(_test_X)

            _train_y = labels_df[:train_end]
            _val_y = labels_df[val_start:val_end]
            _test_y = labels_df[test_start:]
            trains_y.append(_train_y)
            vals_y.append(_val_y)
            tests_y.append(_test_y)

        return (pd.concat(trains_X), pd.concat(trains_y),
                pd.concat(vals_X), pd.concat(vals_y),
                pd.concat(tests_X), pd.concat(tests_y))

    @property
    def codes(self) -> List[int]:
        return self._codes

    def get_fundamental_columns(self) -> List[str]:
        fundamental_cols = self.fin_df.select_dtypes("float64").columns
        fundamental_cols = fundamental_cols[
            fundamental_cols != "Result_Dividend DividendPayableDate"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Local Code"]
        return list(fundamental_cols)

    def get_technical_columns(self) -> List[str]:
        technical_cols = [
            x for x in self._df
            if (x not in self.get_fundamental_columns()) and (x != "code")
        ]
        return list(technical_cols)

    def get_feature_columns(self) -> List[str]:
        return self.get_fundamental_columns() + self.get_technical_columns()

    def _set_date_index(self, df: pd.DataFrame, column: str) -> None:
        df.loc[:, "datetime"] = pd.to_datetime(df.loc[:, column])
        df.set_index("datetime", inplace=True)

    def _set_date_index_all_df(self) -> None:
        self._set_date_index(self.price_df, "EndOfDayQuote Date")
        self._set_date_index(self.fin_df, "base_date")
        self._set_date_index(self.fin_price_df, "base_date")
        self._set_date_index(self.labels_df, "base_date")

    def _calc_return(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df[self.conf.quote_column].pct_change(n)

    def _calc_volatility(self, df: pd.DataFrame, n: int) -> np.ndarray:
        return np.log(df[self.conf.quote_column]).diff().rolling(n).std()

    def _calc_ma_gap(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df[self.conf.quote_column] / df[self.conf.quote_column].rolling(
            n).mean()

    def _extract_feature_weekofday_by_code(self, code: int) -> pd.DataFrame:
        feats = self.fin_price_df[self.fin_price_df["Local Code"] == code]
        feats = feats.loc[pd.Timestamp(self.conf.start_dt) -
                          pd.offsets.BDay(self.conf.buffer_day):]
        feats = feats.loc[:, self.conf.fin_columns +
                          [self.conf.quote_column]].copy()
        feats = feats.fillna(0)

        for i, d in enumerate(self.conf.calc_days, 1):
            feats[f"return_{i}month"] = self._calc_return(feats, d)
            feats[f"volatility_{i}month"] = self._calc_volatility(feats, d)
            feats[f"MA_gap_{i}month"] = self._calc_ma_gap(feats, d)

        feats = feats.fillna(0)
        feats = feats.drop([self.conf.quote_column], axis=1)

        feats = feats.resample("B").ffill()
        feats = feats.loc[feats.index.dayofweek == FRIDAY]
        feats = feats.loc[pd.Timestamp(self.conf.start_dt):]

        feats = feats.replace([np.inf, -np.inf], 0)

        feats["code"] = code
        return feats

    def _extract_feature_for_train_by_code(self, code: int) -> pd.DataFrame:
        fin_df = self.fin_df[self.fin_df["Local Code"] == code]
        fin_df = fin_df.loc[pd.Timestamp(self.conf.start_dt) -
                            pd.offsets.BDay(self.conf.buffer_day):]
        fin_df = fin_df.select_dtypes(include=["float64"])
        fin_df = fin_df.fillna(0)

        price_df = self.price_df[self.price_df["Local Code"] == code]
        price_df = price_df.loc[pd.Timestamp(self.conf.start_dt) -
                                pd.offsets.BDay(self.conf.buffer_day):]
        price_df = price_df.loc[:, [self.conf.quote_column]].copy()

        for i, d in enumerate(self.conf.calc_days, 1):
            price_df[f"return_{i}month"] = self._calc_return(price_df, d)
            price_df[f"volatility_{i}month"] = self._calc_volatility(price_df, d)
            price_df[f"MA_gap_{i}month"] = self._calc_ma_gap(price_df, d)

        price_df = price_df.fillna(0)
        price_df = price_df.drop([self.conf.quote_column], axis=1)

        price_df = price_df.loc[price_df.index.isin(fin_df.index)]
        fin_df = fin_df.loc[fin_df.index.isin(price_df.index)]

        feats_df = pd.concat([price_df, fin_df], axis=1).dropna()

        feats_df = feats_df.replace([np.inf, -np.inf], 0)

        feats_df["code"] = code
        return feats_df
