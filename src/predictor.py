# -*- coding: utf-8 -*-
import io

import pandas as pd

from usecases.feature import *  # noqa:F401, F403
from usecases.strategy import *  # noqa:F401, F403


class ScoringService(object):
    # モデルを保存しているディレクトリへのパス
    model_path = "../model"

    @classmethod
    def get_model(cls, model_path="../model"):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        cls.model_path = model_path

        return True

    @classmethod
    def predict(
            cls,
            inputs,
            feature_service="chapter6_tutorial.Chapter6Tutorial",
            strategy_service="strategy_trend_service.StrategyTrendService"):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
        Returns:
            str: Inference for the given input.
        """
        start_dt = cls._load_purchase_date(inputs)  # noqa: F841

        feature_service = eval(
            f"{feature_service}(inputs, cls.model_path, start_dt)")
        feature_service.preprocess()

        print("[+] generate feature")
        features_dict = feature_service.extract_feature()  # noqa: F841

        strategy_service = eval(
            f"{strategy_service}(features_dict['stock'], features_dict['sentiments'],features_dict['tdnet'])"
        )
        df = strategy_service.execute()

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

    @classmethod
    def _load_purchase_date(cls, inputs):
        purchase_date_df = pd.read_csv(inputs["purchase_date"])
        start_dt = purchase_date_df.sort_values("Purchase Date").iloc[0, 0]
        # 日付型に変換
        start_dt = pd.Timestamp(start_dt)
        # 予測対象日の月曜日日付が指定されているため
        # 特徴量の抽出に使用する1週間前の日付に変換します
        start_dt -= pd.Timedelta("7D")
        # 文字列型に戻す
        start_dt = start_dt.strftime("%Y-%m-%d")
        return start_dt
