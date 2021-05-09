"""チュートリアルの処理
"""

import os
import pathlib
from typing import Dict

import pandas as pd

from domain.feature.news import News, NewsConfig
from domain.feature.price_prediction_service import (PricePredictionConfig,
                                                     PricePredictor)
from domain.feature.stock import Stock, StockConfig
from domain.feature.tdnet import Tdnet

from .feature_service_abc import FeatureService


class LGBMEstimation(FeatureService):
    def __init__(self, inputs, model_path, start_dt) -> None:
        self.model_path = model_path
        inputs["model_path"] = model_path
        inputs[
            "sentiment_dist"] = f"{model_path}/headline_features/LSTM_sentiment.pkl"

        config_dir = pathlib.Path(__file__).parent.parent.parent / "config"
        yaml_path = config_dir / "stock_base.yaml"

        stock_config = StockConfig()
        stock_config.load_config(str(yaml_path.resolve()))
        stock_config.start_dt = start_dt
        self.stock = Stock(stock_config)
        self.stock.load_data(inputs)

        self.tdnet = Tdnet()
        self.tdnet.load_data(inputs)

        self.high_label = "label_high_20"
        hith_price_conf = PricePredictionConfig(self.high_label)
        self.high_price_predictor = PricePredictor(hith_price_conf)
        self.high_price_predictor.load_data(
            os.path.join(self.model_path, "lgbm_label_high_20.pkl"))

        self.low_label = "label_low_20"
        low_price_conf = PricePredictionConfig(self.low_label)
        self.low_price_predictor = PricePredictor(low_price_conf)
        self.low_price_predictor.load_data(
            os.path.join(self.model_path, "my_model_label_low_20.pkl"))

    def preprocess(self) -> None:
        self.stock.preprocess()

    def extract_feature(self) -> Dict[str, pd.DataFrame]:
        stock_df = self.stock.extract_feature()
        stock_df[self.high_label] = self.high_price_predictor.extract_feature(
            stock_df)[self.high_label]
        stock_df[self.low_label] = self.low_price_predictor.extract_feature(
            stock_df)[self.low_label]

        tdnet_df = self.tdnet.extract_feature()
        labels_df = self.stock.get_stock_labels().copy()

        return {"stock": stock_df, "tdnet": tdnet_df, "target": labels_df}

    def get_codes(self):
        return self.stock.codes
