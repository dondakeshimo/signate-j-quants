"""headline_m2_sentimentの値が低い時はリスクとみなし現金保有量を多くする
"""

import numpy as np
import pandas as pd
from pypfopt import CLA, expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from scipy.stats import zscore

from .budget_adjustor_interface import BudgetAdjustorABC, BudgetAdjustorRequest


class BudgetAdjustor(BudgetAdjustorABC):
    def adjust(self, request: BudgetAdjustorRequest) -> None:
        """stock_dfにbudgetカラムを追加して比率を最適化する。
        """
        digit = 1000000
        MIN_BET = 10000
        labels_df = request.labels_df.copy()
        stock_df = request.stock_df.copy()

        # 5銘柄以上購入する必要があるので、30銘柄1万円は必ず購入する。
        stock_df.loc[:, "budget"] = MIN_BET
        # ポートフォリオ最適化で配分する総予算
        total_budget = 1000000 - MIN_BET * request.code_num
        purchase_code_list = stock_df.loc[:, "code"].unique()

        # Pyportfolioの入力形式にdfを変更する。
        # indexにcode追加してmulti indexにする
        labels_df = labels_df.set_index(["Local Code"],
                                        append=True).sort_index()
        # 対象の変化率だけ抜き取り、銘柄コードを列名に。
        labels_df = labels_df.loc[:, "label_high_5"].unstack("Local Code")

        #　期間指定
        end_date = stock_df.index.unique()[0]
        start_date = end_date - pd.offsets.BDay(90)
        labels_df = labels_df[start_date:end_date]
        # 銘柄指定
        labels_df = labels_df.loc[:, purchase_code_list]

        # 期待リターン
        mu = labels_df.mean()
        # 標本分散共分散行列
        std = labels_df.dropna(how='all').cov()
        cla = CLA(mu, std)

        # 最適化
        try:
            portfolio = cla.max_sharpe()
        except Exception as e:
            return stock_df

        # budgetを最適値で更新
        for code in purchase_code_list:
            # 最適化計算時の数値誤差で1を超える場合があるので、四捨五入する。
            ceiled_ratio = np.round(portfolio[code] * digit) / digit
            stock_df.loc[stock_df["code"] == code,
                            "budget"] += int(total_budget * ceiled_ratio)
        stock_df.loc[:, "budget"] = stock_df["budget"].astype(int)
        return stock_df
