import argparse
import pathlib

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

from domain.feature.rf_price_predictor import RFPricePredictor
from domain.feature.stock import Stock, StockConfig

DATASET_DIR = "./data"
MODEL_PATH = "./model"
output_path = "."

# 訓練期間開始日
TRAIN_START = "2016-01-04"
# 訓練期間終了日
TRAIN_END = "2018-12-31"
# 評価期間開始日
VAL_START = "2019-02-01"
# 評価期間終了日
VAL_END = "2019-12-01"
# テスト期間開始日
TEST_START = "2020-01-01"
# 目的変数
TARGET_LABEL = "label_high_5"

INPUTS = {
    "stock_list": f"{DATASET_DIR}/stock_list.csv.gz",
    "stock_price": f"{DATASET_DIR}/stock_price.csv.gz",
    "stock_fin": f"{DATASET_DIR}/stock_fin.csv.gz",
    "stock_fin_price": f"{DATASET_DIR}/stock_fin_price.csv.gz",
    # ニュースデータ
    "tdnet": f"{DATASET_DIR}/tdnet.csv.gz",
    "disclosureItems": f"{DATASET_DIR}/disclosureItems.csv.gz",
    "nikkei_article": f"{DATASET_DIR}/nikkei_article.csv.gz",
    "article": f"{DATASET_DIR}/article.csv.gz",
    "industry": f"{DATASET_DIR}/industry.csv.gz",
    "industry2": f"{DATASET_DIR}/industry2.csv.gz",
    "region": f"{DATASET_DIR}/region.csv.gz",
    "theme": f"{DATASET_DIR}/theme.csv.gz",
    # 目的変数データ
    "stock_labels": f"{DATASET_DIR}/stock_labels.csv.gz",
    # 購入日指定データ
    "purchase_date": f"{DATASET_DIR}/purchase_date.csv",
    "model_path": MODEL_PATH,
}


def main(args: argparse.Namespace) -> None:
    config_dir = pathlib.Path(__file__).parent / "config"
    yaml_path = config_dir / "stock_base.yaml"

    stock_config = StockConfig()
    stock_config.load_config(str(yaml_path.resolve()))
    stock_config.start_dt = TRAIN_START
    stock = Stock(stock_config)
    stock.load_data(INPUTS)

    stock.preprocess()

    train_X, train_y, val_X, val_y, _, _ = stock.get_features_and_label(
        TARGET_LABEL, TRAIN_END, VAL_START, VAL_END, TEST_START)

    predictor = RFPricePredictor()
    predictor.fit(train_X, train_y)
    predictor.save_model(f"{MODEL_PATH}/{args.model_filename}")

    pred_y = predictor.predict(val_X)
    spearman = spearmanr(val_y, pred_y)
    accuracy = accuracy_score(np.sign(val_y), np.sign(pred_y))
    print("\n=== result ===")
    print(f"model: {args.model_filename}\nconfig_path: {args.config_path}")
    print(f"spearman: {spearman}\naccuracy: {accuracy}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_filename")
    parser.add_argument("--config_path")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
