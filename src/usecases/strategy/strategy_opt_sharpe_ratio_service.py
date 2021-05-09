"""全然抽象化されていないけどとりあえず
"""

import pandas as pd

from domain.strategy.budget_adjustment.budget_adjustor_interface import \
    BudgetAdjustorRequest
from domain.strategy.budget_adjustment.optimize_sharpratio import \
    BudgetAdjustor
from domain.strategy.code_selection.code_selector_interface import \
    CodeSelectorRequest
from domain.strategy.code_selection.top_n import CodeSelector

from .strategy_service_abc import StrategyService


class StrategyOptSharpeRatioService(StrategyService):
    def __init__(self, stock_df: pd.DataFrame, tdnet_df: pd.DataFrame,
                 labels_df: pd.DataFrame) -> None:
        self._df = None
        self.stock_df = stock_df
        self.tdnet_df = tdnet_df
        self.labels_df = labels_df
        self.purchase_code_num = 30
        self.code_selector = CodeSelector()
        self.budget_adjustor = BudgetAdjustor()

    def execute(self) -> pd.DataFrame:
        cs_req = CodeSelectorRequest(stock_df=self.stock_df,
                                     tdnet_df=self.tdnet_df,
                                     code_num=self.purchase_code_num)
        df = self.code_selector.select(cs_req)

        ba_req = BudgetAdjustorRequest(stock_df=df,
                                       labels_df=self.labels_df,
                                       code_num=self.purchase_code_num)
        return self.budget_adjustor.adjust(ba_req)
