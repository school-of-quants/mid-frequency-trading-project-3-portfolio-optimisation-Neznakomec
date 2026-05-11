import json
import os
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

from equity_project.src.qpi import QP
from equity_project.src.utils import load_config, three_barrier

warnings.filterwarnings("ignore")

project_path = Path(__file__).parent.parent


def generate_features(data):
    """Generate some features based on data

    Args:
        data (pd.DataFrame): raw OHLC dataset

    Returns:
        pd.DataFrame: features dataset
    """
    X = data.copy()

    close_col = "Close"
    high_col = "High"
    low_col = "Low"

    # dealing with multiindex
    tickers = X[close_col].columns

    # price deviation from moving averages
    X[[(("dev5"), ticker) for ticker in tickers]] = (
        X[close_col] - X[close_col].rolling(5).mean()
    ) / X[close_col]
    X[[(("dev22"), ticker) for ticker in tickers]] = (
        X[close_col] - X[close_col].rolling(22).mean()
    ) / X[close_col]
    X[[(("dev252"), ticker) for ticker in tickers]] = (
        X[close_col] - X[close_col].rolling(252).mean()
    ) / X[close_col]
    X[[(("ma200vs50"), ticker) for ticker in tickers]] = (
        X[close_col].rolling(200).mean() - X[close_col].rolling(50).mean()
    ) / X[close_col]

    # price momentum
    X[[(("mom5"), ticker) for ticker in tickers]] = (
        X[close_col].pct_change(5).rank(axis=1)
    )
    X[[(("mom22"), ticker) for ticker in tickers]] = (
        X[close_col].pct_change(22).rank(axis=1)
    )
    X[[(("mom252"), ticker) for ticker in tickers]] = (
        X[close_col].pct_change(252).rank(axis=1)
    )

    # volatility
    X[[(("vol5"), ticker) for ticker in tickers]] = (X[close_col].rolling(5).std()) / X[
        close_col
    ].rolling(5).mean()
    X[[(("vol22"), ticker) for ticker in tickers]] = (
        X[close_col].rolling(22).std()
    ) / X[close_col].rolling(22).mean()
    X[[(("vol252"), ticker) for ticker in tickers]] = (
        X[close_col].rolling(252).std()
    ) / X[close_col].rolling(252).mean()

    for ticker in tickers:
        X[[("qpi", ticker)]] = (
            QP(target_high=X[(high_col, ticker)].to_numpy(), target_close=X[(close_col, ticker)].to_numpy(), target_low=X[(low_col, ticker)].to_numpy())
        )

    # drop unnecessary сols
    X.drop(columns=["Close", "High", "Low", "Open", "Volume"], inplace=True)

    # avoid forward-looking
    X = X.shift(1)

    # avoid cold start
    X = X.iloc[260:, :]

    return X


def get_label(train_data):
    """Создаем разметку для ML модели на основе тройного барьерного метода. Его параметры захардкожены, но при желании вы можете вынести их в конфиг

    Args:
        train_data (pd.DataFrame): raw OHLC dataset

    Returns:
        pd.DataFrame: target dataset
    """
    # from pandarallel import pandarallel
    # pandarallel.initialize()
    target = (train_data.Close.diff(-30) > 0).astype(int) #parallel_apply(three_barrier)

    # target = train_data.Close.apply(three_barrier)

    return target


def get_raw_data():
    """Скачиваем OHLC данные, а также для каждого тикера определяем дату его первого вхождения в индекс

    Returns:
        pd.DataFrame: OHLC данные для всех акций, когда либо входивших в индекс S&P500
        dict: Словарь, где ключ - тикер, значение - дата первого вхождения в индекс
    """
    cfg = load_config(project_path.parent.as_posix() + "/config.yaml")
    TRAIN_START_DATE = cfg["train_start_date"]
    BACKTEST_END_DATE = cfg["backtest_end_date"]

    # будем включать в выборку бумаг не только актуальных состав индекса, но и все исторические вхождения
    # это должно убрать ошибку выжившего из датасета
    historical_components = pd.read_csv(
        project_path.as_posix() + "/data/pony/S&P_500_Historical_Components.csv",
        index_col=0,
    )

    historical_components[
        (historical_components.index >= TRAIN_START_DATE)
        & (historical_components.index <= BACKTEST_END_DATE)
    ]

    first_appearance_dict = {}

    for index, row in historical_components.iterrows():
        for ticker in row[0].split(","):
            if ticker not in first_appearance_dict:
                first_appearance_dict[ticker] = index

    # поправляем название некоторых тикеров, чтобы yfinance их распознал
    first_appearance_dict["BF-B"] = first_appearance_dict.pop("BF.B")
    first_appearance_dict["BRK-B"] = first_appearance_dict.pop("BRK.B")

    # для этих акций yfinance предоставлет битые данные (нулевые или околонулевые цены для некоторых периодов в прошлом, которые ломают алгоритм)
    # можете изучить их котировки и если yfinance цены истинны, то оставить тикеры в выборке, пока же мы их удалим
    for trash_ticker in ("DEC", "USBC", "CPWR", "TNB", "APP", "BMC", "SBNY"):
        first_appearance_dict.pop(trash_ticker)

    data = pd.read_parquet(project_path.as_posix() + "/data/raw/all_data.parquet")
    with open(project_path.as_posix() + "/data/raw/first_appearance_dict.json") as json_file:
        first_appearance_dict = json.load(json_file)

    data.index = pd.to_datetime(data.index)
    data = data.astype(float)

    return (
        data,
        first_appearance_dict,
    )


def get_data():
    """Скачиваем сырые данные, строим на их основе фичасеты и таргеты для наших тикеров и сохраняем сырые и обработанные данные"""
    cfg = load_config(project_path.parent.as_posix() + "/config.yaml")

    (
        data,
        first_appearance_dict,
    ) = get_raw_data()

    # генерируем фичи для ML модели
    X = generate_features(data)

    # генерируем столбец таргета
    y = get_label(data)

    X = X.stack(level=1)

    TRAIN_START_DATE = cfg["train_start_date"]
    TRAIN_END_DATE = cfg["train_end_date"]
    BACKTEST_START_DATE = cfg["backtest_start_date"]
    BACKTEST_END_DATE = cfg["backtest_end_date"]

    # для каждого тикера определяем дату первого вхождения в индекс
    print("start cleaning features matrix")
    for ticker, first_appearance_dt in first_appearance_dict.items():
        condition_to_drop = (X.index.get_level_values("Date") < first_appearance_dt) & (
            X.index.get_level_values("Ticker") == ticker
        )
        X = X[~condition_to_drop]
    print("features matrix successfully cleaned")

    y = y.stack(level=0).loc[X.index]
    y.name = "target"

    # разбиваем исходные данные на трейн и бэктест
    train_data = data[
        (data.index.get_level_values("Date") <= TRAIN_END_DATE)
        & (data.index.get_level_values("Date") >= TRAIN_START_DATE)
    ]
    train_data.to_parquet(project_path.as_posix() + "/data/raw/train_data.parquet")

    backtest_data = data[
        (data.index.get_level_values("Date") <= BACKTEST_END_DATE)
        & (data.index.get_level_values("Date") >= BACKTEST_START_DATE)
    ]

    backtest_data.to_parquet(
        project_path.as_posix() + "/data/raw/backtest_data.parquet", engine="pyarrow"
    )

    # разбиваем фичасеты и таргеты ML модели на трейн и бэктест
    X_train = X[
        (X.index.get_level_values("Date") <= TRAIN_END_DATE)
        & (X.index.get_level_values("Date") >= TRAIN_START_DATE)
    ]
    os.makedirs(project_path.as_posix() + "/data/processed/", exist_ok=True)
    X_train.to_parquet(project_path.as_posix() + "/data/processed/X_train.parquet")

    y_train = y.to_frame()[
        (y.index.get_level_values("Date") <= TRAIN_END_DATE)
        & (y.index.get_level_values("Date") >= TRAIN_START_DATE)
    ]
    y_train.to_parquet(project_path.as_posix() + "/data/processed/y_train.parquet")

    X_backtest = X[
        (X.index.get_level_values("Date") <= BACKTEST_END_DATE)
        & (X.index.get_level_values("Date") >= BACKTEST_START_DATE)
    ]
    X_backtest.to_parquet(
        project_path.as_posix() + "/data/processed/X_backtest.parquet"
    )

    y_backtest = y.to_frame()[
        (y.index.get_level_values("Date") <= BACKTEST_END_DATE)
        & (y.index.get_level_values("Date") >= BACKTEST_START_DATE)
    ]
    y_backtest.to_parquet(
        project_path.as_posix() + "/data/processed/y_backtest.parquet"
    )


if __name__ == "__main__":
    get_data()
