"""budget_adjustmentモジュールのクラスのインタフェースを規定する
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod

import pandas as pd


@dataclass
class BudgetAdjustorRequest:
    stock_df: pd.DataFrame
    sentiments_df: pd.DataFrame
    dist_start_dt: str = "2019-06-29"
    dist_end_dt: str = "2020-09-25"
    use_start_dt: str = "2020-10-02"


class BudgetAdjustorABC(ABC):
    @abstractmethod
    def adjust(self, request: BudgetAdjustorRequest) -> pd.DataFrame:
        """stock_dfにbudgetカラムを追加したdfを返す
        """
        pass
