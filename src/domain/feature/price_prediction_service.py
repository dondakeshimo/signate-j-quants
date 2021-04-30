"""値の予測モデル

実態はpickleに保存されたモデルを呼び出しているだけ
"""

import pickle
from dataclasses import dataclass
import pandas as pd
from .feature_interface import FeatureInterface


@dataclass
class PricePredictionRequest:
    target_label: str
    feature_df: pd.DataFrame


class PricePredictor(FeatureInterface):
    def __init__(self, request: PricePredictionRequest) -> None:
        self._df = request.feature_df
        self.target_label: str = request.target_label

    def load_data(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def preprocess(self) -> None:
        pass

    def extract_feature(self) -> None:
        self._df[self.target_label] = self.model.predict(self._df)

    @property
    def df(self) -> pd.DataFrame:
        return self._df
