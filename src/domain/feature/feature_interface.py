"""featureモジュールのクラスのインタフェースを規定する

    featureモジュール内のクラスはこのモジュールのクラスを継承している
"""

from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class FeatureInterface(ABC):
    _df: pd.DataFrame

    @abstractmethod
    def load_data(self, path: List[str]) -> None:
        pass

    @abstractmethod
    def preprocess(self) -> None:
        pass

    @abstractmethod
    def extract_feature(self) -> None:
        pass

    @property
    def df(self) -> pd.DataFrame:
        return self._df
