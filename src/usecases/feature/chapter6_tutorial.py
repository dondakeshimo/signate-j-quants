"""チュートリアルの処理
"""

import os

from domain.feature.news import News, NewsConfig
from domain.feature.price_prediction_service import (PricePredictionRequest,
                                                     PricePredictor)
from domain.feature.stock import Stock, StockConfig
from domain.feature.tdnet import Tdnet

from .feature_service_abc import FeatureService


class Chapter6Tutorial(FeatureService):
    def __init__(self, inputs, model_path, start_dt):
        self.model_path = model_path
        inputs["model_path"] = model_path
        inputs[
            "sentiment_dist"] = f"{model_path}/headline_features/LSTM_sentiment.pkl"

        stock_config = StockConfig()
        stock_config.load_config("./src/config/stock_base.yaml")
        stock_config.start_dt = start_dt
        self.stock = Stock(stock_config)
        self.stock.load_data(inputs)

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

        label = "label_high_20"
        hith_price_req = PricePredictionRequest(label, self.stock.df)
        high_price_predictor = PricePredictor(hith_price_req)
        high_price_predictor.load_data(
            os.path.join(self.model_path, "my_model_label_high_20.pkl"))
        high_price_predictor.extract_feature()
        stock_df[label] = high_price_predictor.df[label]

        label = "label_low_20"
        low_price_req = PricePredictionRequest(label, self.stock.df)
        low_price_predictor = PricePredictor(low_price_req)
        low_price_predictor.load_data(
            os.path.join(self.model_path, "my_model_label_low_20.pkl"))
        low_price_predictor.extract_feature()
        stock_df[label] = low_price_predictor.df[label]

        self.news.extract_feature()

        return {
            "stock": stock_df,
            "sentiments": self.news.df,
            "tdnet": self.tdnet.df
        }

    def get_codes(self):
        return self.stock.codes
