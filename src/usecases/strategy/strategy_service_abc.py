"""StrategyServiceの抽象クラスを定義するモジュール

    - StrategyServiceは投資戦略における手順を記載する
    - 具体的なデータの操作や生成についてはdomain層で行う
    - ハイパーパラメタなどの設定についてもdomain層で解釈する
"""

from abc import ABC, abstractmethod

import pandas as pd


class StrategyService(ABC):
    """投資戦略の手順を記載するクラスの親クラス
    """
    @abstractmethod
    def execute(self) -> pd.DataFrame:
        pass
