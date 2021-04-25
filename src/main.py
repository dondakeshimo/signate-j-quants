from predictor import ScoringService

dataset_dir = "./data"
model_path = "./model"
output_path = "."

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

ScoringService.get_model(model_path)
ret = ScoringService.predict(inputs)
print("\n".join(ret.split("\n")[:10]))

with open(f"{output_path}/chapter06-tutorial-1.csv", mode="w") as f:
    f.write(ret)
