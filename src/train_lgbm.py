import pathlib
from domain.feature.stock import Stock, StockConfig
from domain.feature.lgbm_price_predictor import LGBMPricePredictor


dataset_dir = "./data"
model_path = "./model"
output_path = "."

# 訓練期間終了日
TRAIN_END = "2018-12-31"
# 評価期間開始日
VAL_START = "2017-02-01"
# 評価期間終了日
VAL_END = "2017-12-01"
# テスト期間開始日
TEST_START = "2020-01-01"
# 目的変数
TARGET_LABEL = "label_high_20"

inputs = {
    "stock_list": f"{dataset_dir}/stock_list.csv.gz",
    "stock_price": f"{dataset_dir}/stock_price.csv.gz",
    "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
    "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv.gz",
    # ニュースデータ
    "tdnet": f"{dataset_dir}/tdnet.csv.gz",
    "disclosureItems": f"{dataset_dir}/disclosureItems.csv.gz",
    "nikkei_article": f"{dataset_dir}/nikkei_article.csv.gz",
    "article": f"{dataset_dir}/article.csv.gz",
    "industry": f"{dataset_dir}/industry.csv.gz",
    "industry2": f"{dataset_dir}/industry2.csv.gz",
    "region": f"{dataset_dir}/region.csv.gz",
    "theme": f"{dataset_dir}/theme.csv.gz",
    # 目的変数データ
    "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
    # 購入日指定データ
    "purchase_date": f"{dataset_dir}/purchase_date.csv"
}

inputs["model_path"] = model_path

config_dir = pathlib.Path(__file__).parent / "config"
yaml_path = config_dir / "stock_base.yaml"

stock_config = StockConfig()
stock_config.load_config(str(yaml_path.resolve()))
stock_config.start_dt = "2016-01-04"
stock = Stock(stock_config)
stock.load_data(inputs)

stock.preprocess()

train_X, train_y, _, _, _, _ = stock.get_features_and_label(
    TARGET_LABEL, TRAIN_END, VAL_START, VAL_END, TEST_START)

predictor = LGBMPricePredictor()
predictor.fit(train_X, train_y)
predictor.save_model(f"{model_path}/lgbm_label_high_20.pkl")
