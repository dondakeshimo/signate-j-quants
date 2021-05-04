"""headline_m2_sentimentの値が低い時はリスクとみなし現金保有量を多くする
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore

from .budget_adjustor_interface import BudgetAdjustorABC, BudgetAdjustorRequest


class BudgetAdjustor(BudgetAdjustorABC):
    def adjust(self, request: BudgetAdjustorRequest) -> None:
        """stock_dfにbudgetカラムを追加する
        """
        cash_df = self._calc_risk(request.sentiments_df, request.dist_start_dt,
                                  request.dist_end_dt, request.use_start_dt)
        df = request.stock_df.copy()

        cash = 50000
        df.loc[:, "budget"] = cash
        for s in cash_df.index:
            t = cash_df.loc[cash_df.index == s, "risk"][0]
            if t == 10:
                cash = 40000
            elif t == 20:
                cash = 30000
            elif t == 30:
                cash = 20000
            else:
                cash = 50000
            df.loc[df.index == s, "budget"] = cash

        return df

    def _calc_risk(self, sentiments_df: pd.DataFrame, dist_start_dt: str,
                   dist_end_dt: str, use_start_dt: str) -> pd.DataFrame:
        """リクエストのsentiments_dfにriskカラムを追加して返す
        """
        # headline_m2_sentiment_0の値が高いほどポジティブなので符号反転させる
        sentiment_dist = sorted(
            sentiments_df.loc[dist_start_dt:dist_end_dt,
                              "headline_m2_sentiment_0"].values * -1)
        sentiment_use = (sentiments_df.loc[use_start_dt:,
                                           "headline_m2_sentiment_0"].values *
                         -1)

        # DIST_START_DT:DIST_END_DTの分布を使用してリスク判定する
        z = zscore(sentiment_dist)
        # 閾値を決定
        p = np.percentile(z, [25, 50, 75])
        # リスク値を計算する
        u = zscore(sentiment_use)
        # 分布から現金比率の割合を決定
        d = np.digitize(u, p)
        # 出力用に整形して 0, 10, 20, 30 のいずれか返すようにする
        sentiments_df.loc[use_start_dt:, "risk"] = d * 10

        return sentiments_df
