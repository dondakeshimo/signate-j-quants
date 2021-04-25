# signate-j-quants
https://signate.jp/competitions/443


# 準備

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

## 買付日
買付日の指定を行う。とりあえずはチュートリアルで指定していた日付。

```
$ echo "Purchase Date" > data/purchase_date.csv
$ echo "2020-12-28" >> data/purchase_date.csv
```


# 実行方法

```
$ docker-compose up
```


# 提出方法
成果物を下記コマンドで生成する。

```
$ ./hack/make_submit.sh 20210425_submission.zip
```

下記コマンドでできた `20210425_submission.zip` を https://signate.jp/competitions/443/submissions に提出する。
