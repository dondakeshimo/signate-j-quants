"""ニュース解析モデルを訓練する
"""

import pandas as pd

from domain.feature.news_analysis_service import (FeatureCombinerHandler,
                                                  SentimentGenerator)

boundary_week = (2020, 26)
model_path = "./model"
data_path = "./data"

headline_features = pd.read_pickle(f'{model_path}/headline_features.pkl')
keywords_features = pd.read_pickle(f'{model_path}/keywords_features.pkl')
stock_price = pd.read_csv(f"{data_path}/stock_price.csv.gz")
stock_list = pd.read_csv(f"{data_path}/stock_list.csv.gz")

stock_price = stock_price[[
    'EndOfDayQuote Date', 'Local Code', "EndOfDayQuote Open",
    "EndOfDayQuote ExchangeOfficialClose"
]]

stock_price = stock_price.rename(
    columns={
        'EndOfDayQuote Date': 'date',
        'Local Code': 'asset',
        'EndOfDayQuote Open': 'open',
        'EndOfDayQuote ExchangeOfficialClose': 'close',
    })

stock_price['date'] = pd.to_datetime(stock_price['date'])
stock_price['date'] = pd.DatetimeIndex(stock_price['date']).tz_localize(
    headline_features.index.tz)

stock_price = stock_price.set_index(['date', 'asset']).sort_index()
stock_price = stock_price.unstack()
stock_price = stock_price['2020-01-01':]

stock_list = stock_list[['Local Code', 'IssuedShareEquityQuote IssuedShare']]

stock_list = stock_list.rename(columns={
    'Local Code': 'asset',
    'IssuedShareEquityQuote IssuedShare': 'shares'
})

shares = stock_list.set_index('asset')['shares']

last_date = stock_price.index[-1]
universe_condition_1 = stock_price.xs(last_date)["close"].dropna().index
marketcap = (stock_price.xs(last_date)['close'] * shares)
universe_condition_2 = marketcap[marketcap >= 20000000000].index
universe = universe_condition_1 & universe_condition_2
universe.to_series().rename('universe').to_frame().to_csv('universe.csv')
stock_price = stock_price[[
    column for column in stock_price.columns if column[-1] in universe
]]

SentimentGenerator.headline_feature_combiner_handler = FeatureCombinerHandler(
    feature_combiner_params={
        "input_size": 768,
        "hidden_size": 128
    },
    store_dir=f'{model_path}/my_headline_features')
SentimentGenerator.keywords_feature_combiner_handler = FeatureCombinerHandler(
    feature_combiner_params={
        "input_size": 768,
        "hidden_size": 128
    },
    store_dir=f'{model_path}/my_keywords_features')

for features, feature_type in [(headline_features, 'headline_features'),
                               (keywords_features, 'keywords_features')]:
    # feature_typeに合致するfeature_combiner_handlerをSentimentGeneratorから取得する。
    feature_combiner_handler = {
        'headline_features':
        SentimentGenerator.headline_feature_combiner_handler,
        'keywords_features':
        SentimentGenerator.keywords_feature_combiner_handler,
    }[feature_type]

    # 学習及び、validationに用いる、データをビルドする
    weekly_features = SentimentGenerator.build_weekly_features(
        features, boundary_week)
    weekly_labels = SentimentGenerator.build_weekly_labels(
        stock_price, boundary_week)

    # train dataloaderをsetする。
    # このとき、batch_sizeを4にすることで、4つのデータを並列に学習し、
    # num_workersを2にすることでdataloaderはcpu 2coreを用いて、並列的にロードされる。
    feature_combiner_handler.set_train_dataloader(
        dataloader_params={
            "batch_size": 4,
            "num_workers": 2,
        },
        weekly_features=weekly_features['train'],
        weekly_labels=weekly_labels['train'],
        max_sequence_length=1000)

    # validation dataloaderをsetする。
    feature_combiner_handler.set_val_dataloader(
        dataloader_params={
            "batch_size": 4,
            "num_workers": 2,
        },
        weekly_features=weekly_features['test'],
        weekly_labels=weekly_labels['test'],
        max_sequence_length=1000)

    # 学習
    feature_combiner_handler.train(n_epoch=20)
