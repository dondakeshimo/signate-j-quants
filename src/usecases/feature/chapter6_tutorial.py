"""チュートリアルの処理
"""

import os
from .feature_service_abc import FeatureService
from domain.feature.stock import Stock, StockConfig
from domain.feature.price_prediction_service import PricePredictionRequest, PricePredictor
from domain.feature.news import News, NewsConfig
from domain.feature.tdnet import Tdnet


class Chapter6Tutorial(FeatureService):
    def __init__(self, inputs, model_path, start_dt):
        self.model_path = model_path
        inputs["model_path"] = model_path

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

        inputs["sentiment_dist"] = f"{model_path}/headline_features/LSTM_sentiment.pkl"
        news_config = NewsConfig(start_dt, "2020-09-25", ["headline"])
        self.news = News(news_config)
        self.news.load_data(inputs)

        self.tdnet = Tdnet()
        self.tdnet.load_data(inputs)

    def preprocess(self):
        self.stock.preprocess()

    def extract_feature(self):
        self.stock.extract_feature()
        stock_df = self.stock.df
        feature_columns = self.stock.get_feature_columns()

        label = "label_high_20"
        hith_price_req = PricePredictionRequest(label, self.stock.df[feature_columns])
        high_price_predictor = PricePredictor(hith_price_req)
        high_price_predictor.load_data(os.path.join(self.model_path, "my_model_label_high_20.pkl"))
        high_price_predictor.extract_feature()
        stock_df[label] = high_price_predictor.df[label]

        label = "label_low_20"
        low_price_req = PricePredictionRequest(label, self.stock.df[feature_columns])
        low_price_predictor = PricePredictor(low_price_req)
        low_price_predictor.load_data(os.path.join(self.model_path, "my_model_label_low_20.pkl"))
        low_price_predictor.extract_feature()
        stock_df[label] = low_price_predictor.df[label]

        self.news.extract_feature()

        return {"stock": stock_df, "sentiments": self.news.df, "tdnet": self.tdnet.df}

    def get_codes(self):
        return self.stock.codes
