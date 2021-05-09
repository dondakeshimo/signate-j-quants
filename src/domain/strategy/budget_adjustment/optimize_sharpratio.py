"""headline_m2_sentimentの値が低い時はリスクとみなし現金保有量を多くする
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier

from .budget_adjustor_interface import BudgetAdjustorABC, BudgetAdjustorRequest


class BudgetAdjustor(BudgetAdjustorABC):
    def adjust(self, request: BudgetAdjustorRequest) -> None:
        """stock_dfにbudgetカラムを追加して比率を最適化する。
        """
        labels_df = request.labels_df.copy()
        purchase_df = request.stock_df.copy()
        # 購入金額
        TOTAL_BUDGET = 1000000        
        # 最適化できない場合もあるので、初期値として、均等に予算を割り当てる
        cash = TOTAL_BUDGET/request.code_num
        purchase_df.loc[:, "budget"] = cash
        purchase_code_list = purchase_df.loc[:, "code"].unique()

        # Pyportfolioの入力形式にdfを変更する。
        # indexにcode追加してmulti indexにする
        labels_df = labels_df.set_index(["Local Code"], append=True).sort_index()
        # 対象の変化率だけ抜き取り、銘柄コードを列名に。
        labels_df = labels_df.loc[:, "label_high_20"].unstack("Local Code")
        #　期間指定
        end_date = purchase_df.index.unique()[0]
        start_date = end_date - pd.offsets.BDay(90)
        labels_df = labels_df[start_data:end_date]
        # 銘柄指定
        labels_df = labels_df.loc[:,purchase_code_list]
        # 期待リターン
        mu = labels_df.mean()
        # 標本分散共分散行列
        # std = risk_models.sample_cov(df)
        std = labels_df.dropna(how='all').cov()
        # 下方半分散
        # std = risk_models.semicovariance(df, benchmark=0)
        ef = EfficientFrontier(mu, std)
        # 最適化
        try:
            portfolio = ef.max_sharpe(risk_free_rate=0.001)
        except:
            return purchase_df
        # budgetを最適値で更新
        for code in purchase_code_list:
            # 最適化計算時の数値誤差で1を超える場合があるので、小数点の切り上げ・切り捨てする。
            ceiled_ratio = np.ceil(portfolio[code]*100) / 100
            purchase_df.loc[df_purchase["code"]==code, "budget"] = TOTAL_BUDGET * ceiled_ratio

        return purchase_df