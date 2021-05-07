"""価格予測モデルのインタフェース
"""

from abc import ABC, abstractmethod

import pandas as pd


class PricePredictorInterface(ABC):
    @abstractmethod
    def fit(self, train_X: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        pass
