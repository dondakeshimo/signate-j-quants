"""LightGBMを用いた価格予測モデル
"""

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .price_predictor_interface import PricePredictorInterface


class RFPricePredictor(PricePredictorInterface):
    def __init__(self):
        self.regressor = RandomForestRegressor(random_state=0)
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

    def fit(self, train_X: pd.DataFrame, train_y: pd.DataFrame) -> None:
        train_X = train_X.reindex(columns=self.feature_columns)
        self.regressor.fit(train_X, train_y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X.reindex(columns=self.feature_columns)
        return self.regressor.predict(X)

    def save_model(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
