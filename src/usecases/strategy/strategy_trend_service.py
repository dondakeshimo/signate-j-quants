"""全然抽象化されていないけどとりあえず
"""

import pandas as pd
from .strategy_service_abc import StrategyService
from domain.strategy.trend import Trend


class StrategyTrendService(StrategyService):
    def __init__(self, stock_df: pd.DataFrame, sentiments_df: pd.DataFrame,
                 tdnet_df: pd.DataFrame, strategy_id: int) -> None:
        self._df = None
        self.stock_df = stock_df
        self.sentiments_df = sentiments_df
        self.tdnet_df = tdnet_df
        self.strategy_id = strategy_id
        self.strategy = Trend()

    def decide_budget(self) -> None:
        cash_df = self.strategy.get_cash_ratio(self.sentiments_df)
        decision_columns = ["code", "label_high_20", "label_low_20"]
        self._df = self.stock_df.loc[:, decision_columns].copy()

        cash = 50000
        self._df.loc[:, "budget"] = cash
        for s in cash_df.index:
            t = cash_df.loc[cash_df.index == s, "risk"][0]
            if t == 10:
                cash = 40000
            elif t == 20:
                cash = 30000
            elif t == 30:
                cash = 20000
            else:
                cash = 50000
            self._df.loc[self._df.index == s, "budget"] = cash

    def select_code(self) -> None:
        self._df = self.strategy.select_code(self.strategy_id, self._df, self.tdnet_df)

    def adjust_ratio(self) -> None:
        pass

    @property
    def df(self) -> pd.DataFrame:
        return self._df
