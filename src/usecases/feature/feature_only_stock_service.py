"""2021/04/29 stockの確認
"""

from .feature_service_abc import FeatureService
from domain.feature.stock import Stock, StockConfig


class FeatureOnlyStockService(FeatureService):
    def __init__(self, inputs, start_dt):
        stock_config = StockConfig()

        # TODO: read from yaml
        stock_config.start_dt = start_dt
        stock_config.calc_days = [20, 40, 60]
        stock_config.fin_columns = [
            "Result_FinancialStatement FiscalYear",
            "Result_FinancialStatement NetSales",
            "Result_FinancialStatement OperatingIncome",
            "Result_FinancialStatement OrdinaryIncome",
            "Result_FinancialStatement NetIncome",
            "Result_FinancialStatement TotalAssets",
            "Result_FinancialStatement NetAssets",
            "Result_FinancialStatement CashFlowsFromOperatingActivities",
            "Result_FinancialStatement CashFlowsFromFinancingActivities",
            "Result_FinancialStatement CashFlowsFromInvestingActivities",
            "Forecast_FinancialStatement FiscalYear",
            "Forecast_FinancialStatement NetSales",
            "Forecast_FinancialStatement OperatingIncome",
            "Forecast_FinancialStatement OrdinaryIncome",
            "Forecast_FinancialStatement NetIncome",
            "Result_Dividend FiscalYear",
            "Result_Dividend QuarterlyDividendPerShare",
            "Result_Dividend AnnualDividendPerShare",
            "Forecast_Dividend FiscalYear",
            "Forecast_Dividend QuarterlyDividendPerShare",
            "Forecast_Dividend AnnualDividendPerShare",
        ]

        self.stock = Stock(stock_config)
        self.stock.load_data(inputs)

    def preprocess(self):
        self.stock.preprocess()

    def extract_feature(self):
        self.stock.extract_feature()
        return self.stock.df

    def get_codes(self):
        return self.stock.codes
