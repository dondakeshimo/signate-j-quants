"""全然抽象化されていないけどとりあえず
"""

import pandas as pd

from domain.strategy.budget_adjustment.budget_adjustor_interface import BudgetAdjustorRequest
from domain.strategy.budget_adjustment.considering_sentiment_risk import BudgetAdjustor
from domain.strategy.code_selection.code_selector_interface import CodeSelectorRequest
from domain.strategy.code_selection.trend import CodeSelector

from .strategy_service_abc import StrategyService


class StrategyTrendService(StrategyService):
    def __init__(self, stock_df: pd.DataFrame, sentiments_df: pd.DataFrame,
                 tdnet_df: pd.DataFrame) -> None:
        self._df = None
        self.stock_df = stock_df
        self.sentiments_df = sentiments_df
        self.tdnet_df = tdnet_df
        self.code_selector = CodeSelector()
        self.budget_adjustor = BudgetAdjustor()

    def execute(self) -> pd.DataFrame:
        ba_req = BudgetAdjustorRequest(self.stock_df, self.sentiments_df)
        df = self.budget_adjustor.adjust(ba_req)

        cs_req = CodeSelectorRequest(stock_df=df, tdnet_df=self.tdnet_df)
        return self.code_selector.select(cs_req)
