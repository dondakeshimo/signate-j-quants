"""trendのstrategy
"""

import pandas as pd

from .code_selector_interface import CodeSelectorABC, CodeSelectorRequest


class CodeSelector(CodeSelectorABC):
    def select(
        self, 
        request: CodeSelectorRequest):
        code_num = request.code_num  # 銘柄の数
        decision_columns = ["code", "label_high_20"]
        df = request.stock_df.loc[:, decision_columns].copy()
        df.loc[:, "pred"] = df.loc[:, "label_high_20"]

        # 特別損失や決算大赤字を除外する場合
        if request.heuristic == True:
            # 特別損失が発生した銘柄一覧を取得
            df_exclude = self._get_exclude(request.tdnet_df)
            # 除外用にユニークな列を作成します。
            df_exclude.loc[:, "date-code_lastweek"] = (
                df_exclude.loc[:, "start_dt"].dt.strftime("%Y-%m-%d-") +
                df_exclude.loc[:, "code"])
            df_exclude.loc[:, "date-code_thisweek"] = (
                df_exclude.loc[:, "end_dt"].dt.strftime("%Y-%m-%d-") +
                df_exclude.loc[:, "code"])
            #
            df.loc[:, "date-code_lastweek"] = (df.index - pd.Timedelta(
                "7D")).strftime("%Y-%m-%d-") + df.loc[:, "code"].astype(str)
            df.loc[:, "date-code_thisweek"] = df.index.strftime(
                "%Y-%m-%d-") + df.loc[:, "code"].astype(str)
            # 特別損失銘柄を除外
            df = df.loc[~df.loc[:, "date-code_lastweek"].
                        isin(df_exclude.loc[:, "date-code_lastweek"])]
            df = df.loc[~df.loc[:, "date-code_thisweek"].
                        isin(df_exclude.loc[:, "date-code_thisweek"])]

        # 予測出力を降順に並び替え
        df = df.sort_values("pred", ascending=False)
        # 予測出力の大きいものを取得
        df = df.groupby("datetime").head(code_num)

        return df

    def _get_exclude(
            self,
            df_tdnet,  # tdnetのデータ
            start_dt=None,  # データ取得対象の開始日、Noneの場合は制限なし
            end_dt=None,  # データ取得対象の終了日、Noneの場合は制限なし
            lookback=7,  # 除外考慮期間 (days)
            target_day_of_week=4,  # 起点となる曜日
    ):
        # 特別損失のレコードを取得
        special_loss = df_tdnet[df_tdnet["disclosureItems"].str.contains(
            '201"')].copy()
        # 日付型を調整
        special_loss["date"] = pd.to_datetime(special_loss["disclosedDate"])
        # 処理対象開始日が設定されていない場合はデータの最初の日付を取得
        if start_dt is None:
            start_dt = special_loss["date"].iloc[0]
        # 処理対象終了日が設定されていない場合はデータの最後の日付を取得
        if end_dt is None:
            end_dt = special_loss["date"].iloc[-1]
        #  処理対象日で絞り込み
        special_loss = special_loss[(start_dt <= special_loss["date"])
                                    & (special_loss["date"] <= end_dt)]
        # 出力用にカラムを調整
        res = special_loss[["code", "disclosedDate", "date"]].copy()
        # 銘柄コードを4桁にする
        res["code"] = res["code"].astype(str).str[:-1]
        # 予測の基準となる金曜日の日付にするために調整
        res["remain"] = (target_day_of_week - res["date"].dt.dayofweek) % 7
        res["start_dt"] = res["date"] + pd.to_timedelta(res["remain"],
                                                        unit="d")
        res["end_dt"] = res["start_dt"] + pd.Timedelta(days=lookback)
        # 出力するカラムを指定
        columns = ["code", "date", "start_dt", "end_dt"]
        return res[columns].reset_index(drop=True)
