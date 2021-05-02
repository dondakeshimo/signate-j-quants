"""tdnetに関するビジネスルール
"""

from typing import Dict

import pandas as pd

from .feature_interface import FeatureInterface


class Tdnet(FeatureInterface):
    def load_data(self, inputs: Dict[str, str]) -> None:
        self._df = pd.read_csv(inputs["tdnet"])

    def preprocess(self) -> None:
        pass

    def extract_feature(self) -> pd.DataFrame:
        return self._df
