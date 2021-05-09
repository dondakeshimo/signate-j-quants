import argparse

from predictor import ScoringService

DATASET_DIR = "./data"
MODEL_PATH = "./model"
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
    "purchase_date": f"{DATASET_DIR}/purchase_date.csv"
}


def main(args: argparse.Namespace) -> None:
    ScoringService.get_model(MODEL_PATH)
    ret = ScoringService.predict(INPUTS,
                                 feature_service=args.feature_service,
                                 strategy_service=args.strategy_service)

    print("\n== 出力データの確認 ==")
    print("\n".join(ret.split("\n")[:10]))

    with open(args.output_path, mode="w") as f:
        f.write(ret)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path")
    parser.add_argument("--feature_service",
                        default="chapter6_tutorial.Chapter6Tutorial")
    parser.add_argument(
        "--strategy_service",
        default=
        "strategy_opt_sharpe_ratio_service.StrategyOptSharpeRatioService")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
