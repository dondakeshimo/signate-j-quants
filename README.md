# signate-j-quants
https://signate.jp/competitions/443


# 準備

## docker image

```
$ docker build . -t dondakeshimo/signate-j-quants
```

## data
https://signate.jp/competitions/443/data

上記URLの拡張子が `csv.gz` , `csv` のものをダウンロードして `data/` に配置する。

## model
https://signate.jp/competitions/443/data

上記URLの

- headline\_features.zip
- keywords\_features.zip
- Chapter02\_models.zip

をダウンロードし、 `model/` 配下で解凍する。

`Chapter02_models.zip` については余分な階層がうまれてしまっているので `pkl` ファイルを `model` 直下に移動させる

```
$ mv model/Chapter02_models/*.pkl model/
```

BERTに使用する学習済みモデルをダウンロードする

```
$ docker run --rm -it -v $(pwd):/opt/ml dondakeshimo/signate-j-quants python hack/download_bert_model.py
```

## 買付日
買付日の指定を行う。とりあえずはチュートリアルで指定していた日付。

```
$ echo "Purchase Date" > data/purchase_date.csv
$ echo "2020-12-28" >> data/purchase_date.csv
```


# 実行方法

### GPUあり

```
$ docker run --gpus all --rm -it -v $(pwd):/opt/ml dondakeshimo/signate-j-quants python src/main.py submission.csv
```

### GPUなし

```
$ docker run --rm -it -v $(pwd):/opt/ml dondakeshimo/signate-j-quants python src/main.py submission.csv
```

### LGBMの訓練

```
$ docker run --gpus all --rm -it -v $(pwd):/opt/ml dondakeshimo/signate-j-quants python src/train_lgbm.py lgbm_label_high_20.pkl --config_path ./src/config/lgbm_base.yaml
```

### その他

一応docker-composeでも立ち上げられる。
```
(deprecated)
$ docker-compose up
```


# Jupyter実行方法
```
$ docker run --rm --name tutorial --shm-size=2G -v ${PWD}:/notebook -p8888:8888 --rm -it dondakeshimo/signate-j-quants jupyter notebook --ip 0.0.0.0 --allow-root --no-browser --no-mathjax --NotebookApp.disable_check_xsrf=True --NotebookApp.token='' --NotebookApp.password='' /notebook
```
ipynbは `ipynb` ディレクトリ配下に作成すること。


# 提出方法
成果物を下記コマンドで生成する。

```
$ ./hack/make_submit.sh 20210425_submission.zip
```

下記コマンドでできた `20210425_submission.zip` を https://signate.jp/competitions/443/submissions に提出する。
