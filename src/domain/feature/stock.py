"""stockに関するビジネスルール
"""

from dataclasses import dataclass, field
from .feature_interface import FeatureInterface
from typing import List
import pandas as pd
import numpy as np

FRIDAY = 4


@dataclass
class StockConfig:
    quote_column: str = "EndOfDayQuote ExchangeOfficialClose"
    buffer_day: int = 90  # 計算用に確保するstart_dtからのバッファ日数
    calc_days: List[int] = field(default_factory=list)
    start_dt: str = "2019-02-01"
    fin_columns: List[str] = field(default_factory=list)


class Stock(FeatureInterface):
    def __init__(self, config: StockConfig) -> None:
        self._df: pd.DataFrame = None
        self.conf: StockConfig = config

    def load_data(self, inputs: List[str]) -> None:
        self.list_df = pd.read_csv(inputs["stock_list"])
        self.fin_df = pd.read_csv(inputs["stock_fin"])
        self.fin_price_df = pd.read_csv(inputs["stock_fin_price"])

        # NOTE: コピー要るかわからないけどチュートリアルを踏襲
        stock_list = self.list_df.copy()
        self._codes = stock_list[stock_list["universe_comp2"] == True]["Local Code"].values

    def preprocess(self) -> None:
        self._set_date_index_all_df()

    def extract_feature(self) -> None:
        buff = []
        for code in self.codes:
            buff.append(self._extract_feature_by_code(code))

        self._df = pd.concat(buff)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def codes(self) -> List[int]:
        return self._codes

    def get_fundamental_columns(self) -> List[str]:
        fundamental_cols = self.fin_df.select_dtypes("float64").columns
        fundamental_cols = fundamental_cols[
            fundamental_cols != "Result_Dividend DividendPayableDate"
        ]
        fundamental_cols = fundamental_cols[fundamental_cols != "Local Code"]
        return list(fundamental_cols)

    def get_technical_columns(self) -> List[str]:
        technical_cols = [
            x for x in self._df if (x not in self.get_fundamental_columns()) and (x != "code")
        ]
        return list(technical_cols)

    def get_feature_columns(self) -> List[str]:
        return self.get_fundamental_columns() + self.get_technical_columns()

    def _set_date_index(self, df: pd.DataFrame, column: str) -> None:
        df.loc[:, "datetime"] = pd.to_datetime(df.loc[:, column])
        df.set_index("datetime", inplace=True)

    def _set_date_index_all_df(self) -> None:
        self._set_date_index(self.fin_df, "base_date")
        self._set_date_index(self.fin_price_df, "base_date")

    def _calc_return(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df[self.conf.quote_column].pct_change(n)

    def _calc_volatility(self, df: pd.DataFrame, n: int) -> np.ndarray:
        return np.log(df[self.conf.quote_column]).diff().rolling(n).std()

    def _calc_ma_gap(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df[self.conf.quote_column] / df[self.conf.quote_column].rolling(n).mean()

    def _extract_feature_by_code(self, code: int) -> pd.DataFrame:
        feats = self.fin_price_df[self.fin_price_df["Local Code"] == code]
        feats = feats.loc[pd.Timestamp(self.conf.start_dt) - pd.offsets.BDay(self.conf.buffer_day):]
        feats = feats.loc[:, self.conf.fin_columns + [self.conf.quote_column]].copy()
        feats = feats.fillna(0)

        # HACK: column order unified
        for i, d in enumerate(self.conf.calc_days, 1):
            feats[f"return_{i}month"] = self._calc_return(feats, d)
        for i, d in enumerate(self.conf.calc_days, 1):
            feats[f"volatility_{i}month"] = self._calc_volatility(feats, d)
        for i, d in enumerate(self.conf.calc_days, 1):
            feats[f"MA_gap_{i}month"] = self._calc_ma_gap(feats, d)

        feats = feats.fillna(0)
        feats = feats.drop([self.conf.quote_column], axis=1)

        feats = feats.resample("B").ffill()
        feats = feats.loc[feats.index.dayofweek == FRIDAY]
        feats = feats.loc[pd.Timestamp(self.conf.start_dt):]

        feats = feats.replace([np.inf, -np.inf], 0)

        feats["code"] = code
        return feats
