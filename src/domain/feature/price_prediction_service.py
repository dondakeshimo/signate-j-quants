"""値の予測モデル

実態はpickleに保存されたモデルを呼び出しているだけ
"""

import pickle
from dataclasses import dataclass

import pandas as pd

from .feature_interface import FeatureInterface


@dataclass
class PricePredictionConfig:
    target_label: str


class PricePredictor(FeatureInterface):
    def __init__(self, config: PricePredictionConfig) -> None:
        self.target_label: str = config.target_label
        self.feature_columns = [
            'Result_FinancialStatement FiscalYear',
            'Result_FinancialStatement NetSales',
            'Result_FinancialStatement OperatingIncome',
            'Result_FinancialStatement OrdinaryIncome',
            'Result_FinancialStatement NetIncome',
            'Result_FinancialStatement TotalAssets',
            'Result_FinancialStatement NetAssets',
            'Result_FinancialStatement CashFlowsFromOperatingActivities',
            'Result_FinancialStatement CashFlowsFromFinancingActivities',
            'Result_FinancialStatement CashFlowsFromInvestingActivities',
            'Forecast_FinancialStatement FiscalYear',
            'Forecast_FinancialStatement NetSales',
            'Forecast_FinancialStatement OperatingIncome',
            'Forecast_FinancialStatement OrdinaryIncome',
            'Forecast_FinancialStatement NetIncome',
            'Result_Dividend FiscalYear',
            'Result_Dividend QuarterlyDividendPerShare',
            'Result_Dividend AnnualDividendPerShare',
            'Forecast_Dividend FiscalYear',
            'Forecast_Dividend QuarterlyDividendPerShare',
            'Forecast_Dividend AnnualDividendPerShare', 'return_1month',
            'return_2month', 'return_3month', 'volatility_1month',
            'volatility_2month', 'volatility_3month', 'MA_gap_1month',
            'MA_gap_2month', 'MA_gap_3month'
        ]

    def load_data(self, pkl_path: str) -> None:
        with open(pkl_path, "rb") as f:
            self.model = pickle.load(f)

    def preprocess(self) -> None:
        pass

    def extract_feature(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        stock_df = stock_df.reindex(columns=self.feature_columns)
        stock_df[self.target_label] = self.model.predict(stock_df)
        return stock_df
