"""featureモジュールのクラスのインタフェースを規定する

    featureモジュール内のクラスはこのモジュールのクラスを継承している
"""

from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class FeatureInterface(ABC):
    @abstractmethod
    def load_data(self, path: List[str]) -> None:
        pass

    @abstractmethod
    def preprocess(self) -> None:
        pass

    @abstractmethod
    def extract_feature(self) -> pd.DataFrame:
        pass
