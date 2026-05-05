import os
from pathlib import Path

import joblib
import pandas as pd
import vectorbt as vbt

from equity_project.src.utils import load_config, save_dict

project_path = Path(__file__).parent.parent


def generate_weights(preds):
    """Превращаем скоры ML модели в веса бумаг в портфеле

    Args:
        preds (pd.DataFrame): Датафрейм скоров ML модели

    Returns:
        pd.DataFrame: Веса бумаг в портфеле
    """
    preds_unstack = preds.unstack(level=1)

    # считаем разницу между вероятностью сигнала на лонг и вероятностью шорт сигнала
    long_prob_minus_short_prob = preds_unstack[0] - preds_unstack[1]

    # считаем ранги данного фактора. У бумаги с наибольшим фактором самый большой ранг
    signals_rank = long_prob_minus_short_prob.rank(axis=1, ascending=False, pct=False)

    # конструируем веса на основе рангов
    weights = signals_rank

    # отсеиваем бумаги, у которых вероятность падения больше вероятности роста
    weights[long_prob_minus_short_prob < 0] = 0

    # делим ранг на сумму рангов бумаг, таким образом в портфеле дается больше веса тем бумагам, которые имеют бОльшую вероятность роста (пропорционально рангу)
    weights = (weights.T / weights.sum(axis=1)).T
    weights = weights.fillna(0)
    return weights


def run_backtest():
    """
    Запускает бэктест на бэктестовых данных
    Сохраняет:
        - Основные бэктестовые метрики в /artifacts/backtest_metrics.json
        - График PnL стратегии в /artifacts/pnl.png
    """

    os.makedirs(project_path.as_posix() + "/artifacts/plots", exist_ok=True)
    os.makedirs(project_path.as_posix() + "/artifacts/metrics", exist_ok=True)

    cfg = load_config(project_path.parent.as_posix() + "/config.yaml")

    # считываем бэктестовые данные и ML модель
    X_backtest = pd.read_parquet(
        project_path.as_posix() + "/data/processed/X_backtest.parquet"
    )

    backtest_data = pd.read_parquet(
        project_path.as_posix() + "/data/raw/backtest_data.parquet", engine="pyarrow"
    )

    # производим инференс модели
    model = joblib.load(project_path.as_posix() + "/models/model.joblib")
    preds = model.predict_proba(X_backtest)
    preds = pd.DataFrame(preds, index=X_backtest.index)

    # избавляемся от полностью пустых колонок котировок
    close = backtest_data.Close.dropna(axis=1, how="all")
    size = generate_weights(preds)
    price = backtest_data.shift(-1).Open[list(set(close.columns) & set(size.columns))]
    close = close[price.columns]
    size = size[price.columns]

    # формируем портфель на основе сигналов
    init_cash = cfg["init_cash"]
    fees = cfg["fees"]

    pf = vbt.Portfolio.from_orders(
        close=close,
        price=price,
        size=size,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        freq="1d",
        init_cash=init_cash,
        fees=fees,
    )

    # сохраняем PnL график
    pf.plot().write_image(project_path.as_posix() + "/artifacts/plots/pnl.png")

    # сохраняем метрики бэктеста
    backtest_metrics = pf.stats().to_dict()
    save_dict(
        backtest_metrics,
        project_path.as_posix() + "/artifacts/metrics/backtest_metrics.json",
    )


if __name__ == "__main__":
    run_backtest()
