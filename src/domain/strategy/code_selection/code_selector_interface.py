"""code_selectionモジュールのクラスのインタフェースを規定する
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class CodeSelectorRequest:
    stock_df: pd.DataFrame
    tdnet_df: pd.DataFrame


class CodeSelectorABC(ABC):
    @abstractmethod
    def select(self, request: CodeSelectorRequest) -> pd.DataFrame:
        """stock_dfから選ばれた銘柄分だけ抽出して返す
        """
        pass
