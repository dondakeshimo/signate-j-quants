"""newsに関するビジネスルール
"""

from dataclasses import dataclass, field
from .feature_interface import FeatureInterface
from .news_analysis_service import SentimentGenerator
from typing import Dict, List
import pandas as pd


@dataclass
class NewsConfig:
    start_dt: str = "2019-02-01"
    dist_end_dt: str = "2020-09-25"
    target_feature_types: List[str] = field(default_factory=list)


class News(FeatureInterface):
    DIST_END_DT = "2020-09-25"

    def __init__(self, config: NewsConfig) -> None:
        self._df: pd.DataFrame = None
        self.conf: NewsConfig = config

    def load_data(self, inputs: Dict[str, str]) -> None:
        """
        inputsに以下のキーが必要
            - model_path
            - sentiment_dist
            - nikkei_article
        """
        SentimentGenerator.initialize(inputs["model_path"])
        self._load_analyzed_sentiments(inputs["sentiment_dist"])
        self.article_path = inputs["nikkei_article"]

    def preprocess(self) -> None:
        pass

    def extract_feature(self) -> None:
        self._df = pd.concat([self.analyzed_sentiments_df, self._analyze_sentiment(self.article_path)])

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _analyze_sentiment(self, article_path: str) -> pd.DataFrame:
        df = SentimentGenerator.generate_lstm_features(
            article_path,
            start_dt=self.conf.start_dt,
            target_feature_types=self.conf.target_feature_types,
        )["headline_features"]
        df = self._arrange_sentiments_df(df)
        return df

    def _load_analyzed_sentiments(self, path) -> None:
        df = pd.read_pickle(path)
        df = self._arrange_sentiments_df(df)
        df = df.loc[:self.conf.dist_end_dt]
        self.analyzed_sentiments_df = df

    def _arrange_sentiments_df(self, df) -> pd.DataFrame:
        df.loc[:, "index"] = df.index.map(
            lambda x: self._transform_yearweek_to_monday(x[0], x[1]))
        df.set_index("index", inplace=True)
        df.rename(columns={0: "headline_m2_sentiment_0"}, inplace=True)
        df.index = df.index + pd.Timedelta("4D")
        return df

    def _transform_yearweek_to_monday(self, year, week):
        """
        ニュースから抽出した特徴量データのindexは (year, week) なので、
        (year, week) => YYYY-MM-DD 形式(月曜日) に変換します。
        """
        for s in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
            if s.week == week:
                # to return Monday of the first week of the year
                # e.g. "2020-01-01" => "2019-12-30"
                return s - pd.Timedelta(f"{s.dayofweek}D")
