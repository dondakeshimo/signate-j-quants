# project rootから実行すること

import sys
sys.path.append("./src")

from module import SentimentGenerator

model_path = "./model"

SentimentGenerator.load_feature_extractor(model_path, download=True, save_local=True)
SentimentGenerator.load_bert_tokenizer(model_path, download=True, save_local=True)
