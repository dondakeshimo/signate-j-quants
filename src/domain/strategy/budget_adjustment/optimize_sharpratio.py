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
        df = request.stock_df_allperiod.copy()
        df_purchase = request.stock_df.copy()
        # 購入金額
        TOTAL_BUDGET = 1000000        
        # 最適化できない場合もあるので、初期値として、均等に予算を割り当てる
        cash = TOTAL_BUDGET/code_num
        df_purchase.loc[:, "budget"] = cash
        purchase_code_list = df_purchase.loc[:, "code"].unique()

        # Pyportfolioの入力形式にdfを変更する。
        # indexにcode追加してmulti indexにする
        df = df.set_index(['code'], append=True).sort_index()
        # 対象の変化率だけ抜き取り、銘柄コードを列名に。
        df = df.loc[:, "return_1month"].unstack("code")
        #　期間指定
        # start_date = 
        # end_date = 
        # df = df[start_data:end_date]
        # 銘柄指定
        df = df.loc[:,purchase_code_list]
        # 期待リターン
        mu = df.mean()
        # 標本分散共分散行列
        # std = risk_models.sample_cov(df)
        std = df.dropna(how='all').cov()
        # 下方半分散
        # std = risk_models.semicovariance(df, benchmark=0)
        ef = EfficientFrontier(mu, std)
        # 最適化
        try:
            portfolio = ef.max_sharpe(risk_free_rate=0.001)
        except:
            return df_purchase
        # budgetを最適値で更新
        for code in purchase_code_list:
            # 最適化計算時の数値誤差で1を超える場合があるので、小数点の切り上げ・切り捨てする。
            ceiled_ratio = np.ceil(portfolio[code]*100) / 100
            df_purchase.loc[df_purchase["code"]==code, "budget"] = TOTAL_BUDGET * ceiled_ratio

        return df_purchase