#!/bin/bash

# current directoryを親ディレクトリに移動
cd `dirname $0`/../

# 第一引数が成果物の名前 (default: submit.zip)
SUBMIT_NAME=${1:-submit.zip}
TMP_DIR=tmp

# tmpディレクトリがなければ作る
if [ ! -d $TMP_DIR ]; then mkdir $TMP_DIR; fi

# 対象をコピー
cp -r model $TMP_DIR/
cp -r src $TMP_DIR/
cp requirements.txt $TMP_DIR/

# zipにまとめる
zip -r $SUBMIT_NAME $TMP_DIR
