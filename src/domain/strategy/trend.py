"""trendのstrategy

次のstrategyを思いついたタイミングで抽象化を行う
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore


class Trend:

    def get_exclude(
        cls,
        df_tdnet,  # tdnetのデータ
        start_dt=None,  # データ取得対象の開始日、Noneの場合は制限なし
        end_dt=None,  # データ取得対象の終了日、Noneの場合は制限なし
        lookback=7,  # 除外考慮期間 (days)
        target_day_of_week=4,  # 起点となる曜日
    ):
        # 特別損失のレコードを取得
        special_loss = df_tdnet[df_tdnet["disclosureItems"].str.contains('201"')].copy()
        # 日付型を調整
        special_loss["date"] = pd.to_datetime(special_loss["disclosedDate"])
        # 処理対象開始日が設定されていない場合はデータの最初の日付を取得
        if start_dt is None:
            start_dt = special_loss["date"].iloc[0]
        # 処理対象終了日が設定されていない場合はデータの最後の日付を取得
        if end_dt is None:
            end_dt = special_loss["date"].iloc[-1]
        #  処理対象日で絞り込み
        special_loss = special_loss[
            (start_dt <= special_loss["date"]) & (special_loss["date"] <= end_dt)
        ]
        # 出力用にカラムを調整
        res = special_loss[["code", "disclosedDate", "date"]].copy()
        # 銘柄コードを4桁にする
        res["code"] = res["code"].astype(str).str[:-1]
        # 予測の基準となる金曜日の日付にするために調整
        res["remain"] = (target_day_of_week - res["date"].dt.dayofweek) % 7
        res["start_dt"] = res["date"] + pd.to_timedelta(res["remain"], unit="d")
        res["end_dt"] = res["start_dt"] + pd.Timedelta(days=lookback)
        # 出力するカラムを指定
        columns = ["code", "date", "start_dt", "end_dt"]
        return res[columns].reset_index(drop=True)

    def select_code(cls, strategy_id, df, df_tdnet):
        df = df.copy()
        # 銘柄選択方法選択
        if strategy_id in [1, 4]:
            # 最高値モデル +　最安値モデル
            df.loc[:, "pred"] = df.loc[:, "label_high_20"] + df.loc[:, "label_low_20"]
        elif strategy_id in [2, 5]:
            # 最高値モデル
            df.loc[:, "pred"] = df.loc[:, "label_high_20"]
        elif strategy_id in [3, 6]:
            # 最高値モデル
            df.loc[:, "pred"] = df.loc[:, "label_low_20"]
        else:
            raise ValueError("no strategy_id selected")

        # 特別損失を除外する場合
        if strategy_id in [4, 5, 6]:
            # 特別損失が発生した銘柄一覧を取得
            df_exclude = cls.get_exclude(df_tdnet)
            # 除外用にユニークな列を作成します。
            df_exclude.loc[:, "date-code_lastweek"] = (
                df_exclude.loc[:, "start_dt"].dt.strftime("%Y-%m-%d-") + df_exclude.loc[:, "code"]
            )
            df_exclude.loc[:, "date-code_thisweek"] = (
                df_exclude.loc[:, "end_dt"].dt.strftime("%Y-%m-%d-") + df_exclude.loc[:, "code"]
            )
            #
            df.loc[:, "date-code_lastweek"] = (df.index - pd.Timedelta("7D")).strftime(
                "%Y-%m-%d-"
            ) + df.loc[:, "code"].astype(str)
            df.loc[:, "date-code_thisweek"] = df.index.strftime("%Y-%m-%d-") + df.loc[
                :, "code"
            ].astype(str)
            # 特別損失銘柄を除外
            df = df.loc[
                ~df.loc[:, "date-code_lastweek"].isin(
                    df_exclude.loc[:, "date-code_lastweek"]
                )
            ]
            df = df.loc[
                ~df.loc[:, "date-code_thisweek"].isin(
                    df_exclude.loc[:, "date-code_thisweek"]
                )
            ]

        # 予測出力を降順に並び替え
        df = df.sort_values("pred", ascending=False)
        # 予測出力の大きいものを取得
        df = df.groupby("datetime").head(30)

        return df

    def get_cash_ratio(cls, df_sentiment):
        """
        headline_m2_sentimentの値が低い時はリスクとみなし
        現金保有量を多くする。

        入力:
        センチメント (DataFrame): センチメント情報
        出力:
        現金比率 (Dataframe): 入力値に現金比率を追加したもの
        """
        # リスク値マッピング用の分布取得期間
        DIST_START_DT = "2020-06-29"
        DIST_END_DT = "2020-09-25"
        # リスク値を計算する期間
        USE_START_DT = "2020-10-02"

        # 出力用にコピー
        df_sentiment = df_sentiment.copy()
        # headline_m2_sentiment_0の値が高いほどポジティブなので符号反転させる
        sentiment_dist = sorted(
            df_sentiment.loc[
                DIST_START_DT:DIST_END_DT, "headline_m2_sentiment_0"
            ].values * -1
        )
        sentiment_use = (
            df_sentiment.loc[USE_START_DT:, "headline_m2_sentiment_0"].values * -1
        )

        # DIST_START_DT:DIST_END_DTの分布を使用してリスク判定する
        z = zscore(sentiment_dist)
        # 閾値を決定
        p = np.percentile(z, [25, 50, 75])
        # リスク値を計算する
        u = zscore(sentiment_use)
        # 分布から現金比率の割合を決定
        d = np.digitize(u, p)
        # 出力用に整形して 0, 10, 20, 30 のいずれか返すようにする
        df_sentiment.loc[USE_START_DT:, "risk"] = d * 10

        return df_sentiment
