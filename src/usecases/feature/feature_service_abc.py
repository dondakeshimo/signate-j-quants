"""FeatureServiceの抽象クラスを定義するモジュール

    - FeatureServiceは特徴量抽出における手順を記載する
    - 具体的なデータの操作や生成についてはdomain層で行う
    - ハイパーパラメタなどの設定についてもdomain層で解釈する
"""

from abc import ABC, abstractmethod


class FeatureService(ABC):
    """特徴量抽出の手順を記載するクラスの親クラス
    """

    @abstractmethod
    def preprocess(self) -> None:
        pass

    @abstractmethod
    def extract_feature(self) -> None:
        pass
