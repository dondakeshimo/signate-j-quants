"""code_selectionモジュールのクラスのインタフェースを規定する
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class CodeSelectorRequest:
    stock_df: pd.DataFrame
    tdnet_df: pd.DataFrame
    nikkei_df: pd.DataFrame
    code_num: int  #選択銘柄数
    heuristic=False #特別損失や決算大赤字を除外する場合True 


class CodeSelectorABC(ABC):
    @abstractmethod
    def select(self, request: CodeSelectorRequest) -> pd.DataFrame:
        """stock_dfから選ばれた銘柄分だけ抽出して返す
        """
        pass
