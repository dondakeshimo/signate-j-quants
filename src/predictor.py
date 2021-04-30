# -*- coding: utf-8 -*-
import io
import os
import pickle

import numpy as np
import pandas as pd
from usecases.feature.feature_stock_price_news_service import FeatureStockPriceNewsService
from usecases.strategy.strategy_trend_service import StrategyTrendService
from scipy.stats import zscore


class ScoringService(object):
    # テスト期間開始日
    TEST_START = "2021-02-01"
    # 目的変数
    TARGET_LABELS = ["label_high_20", "label_low_20"]
    # データをこの変数に読み込む
    dfs = None
    # モデルをこの変数に読み込む
    models = None
    # 対象の銘柄コードをこの変数に読み込む
    codes = None
    # センチメントの分布をこの変数に読み込む
    df_sentiment_dist = None
    # モデルを保存しているディレクトリへのパス
    model_path = "../model"

    @classmethod
    def get_dataset(cls, inputs, load_data):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            # 必要なデータのみ読み込みます
            if k not in load_data:
                continue
            cls.dfs[k] = pd.read_csv(v)
            # DataFrameのindexを設定します。
            if k == "stock_price":
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "EndOfDayQuote Date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
            elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "base_date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
        return cls.dfs

    @classmethod
    def get_model(cls, model_path="../model", labels=None):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            labels (arrayt): list of prediction target labels

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        cls.model_path = model_path

        return True

    @classmethod
    def predict(
        cls,
        inputs,
        labels=None,
        codes=None,
        start_dt=TEST_START,
        load_data=[
            "stock_list",
            "stock_fin",
            "stock_fin_price",
            "stock_price",
            "tdnet",
            "purchase_date",
        ],
        fin_columns=None,
        strategy_id=5,
    ):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify target purchase date
            load_data (list[str]): list of data to load
            fin_columns (list[str]): list of columns to use as features
            strategy_id (int): specify strategy to use
        Returns:
            str: Inference for the given input.
        """
        # データ読み込み
        if cls.dfs is None:
            print("[+] load data")
            cls.get_dataset(inputs, load_data)

        # purchase_date が存在する場合は予測対象日を上書き
        if "purchase_date" in cls.dfs.keys():
            # purchase_dateの最も古い日付を設定
            start_dt = cls.dfs["purchase_date"].sort_values("Purchase Date").iloc[0, 0]

        # 日付型に変換
        start_dt = pd.Timestamp(start_dt)
        # 予測対象日の月曜日日付が指定されているため
        # 特徴量の抽出に使用する1週間前の日付に変換します
        start_dt -= pd.Timedelta("7D")
        # 文字列型に戻す
        start_dt = start_dt.strftime("%Y-%m-%d")

        feature_service = FeatureStockPriceNewsService(inputs, cls.model_path, start_dt)
        feature_service.preprocess()

        # 特徴量を作成
        print("[+] generate feature")
        features_dict = feature_service.extract_feature()
        feats = features_dict["stock"]
        df_sentiments = features_dict["sentiments"]

        strategy_service = StrategyTrendService(feats, df_sentiments, cls.dfs["tdnet"], strategy_id)
        strategy_service.decide_budget()
        strategy_service.select_code()
        df = strategy_service.df

        # 日付順に並び替え
        df.sort_index(kind="mergesort", inplace=True)
        # 月曜日日付に変更
        df.index = df.index + pd.Timedelta("3D")
        # 出力用に調整
        df.index.name = "date"
        df.rename(columns={"code": "Local Code"}, inplace=True)
        df.reset_index(inplace=True)

        # 出力対象列を定義
        output_columns = ["date", "Local Code", "budget"]

        out = io.StringIO()
        df.to_csv(out, header=True, index=False, columns=output_columns)

        return out.getvalue()
