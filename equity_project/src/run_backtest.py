import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import vectorbt as vbt

from equity_project.src.utils import load_config, save_dict

project_path = Path(__file__).parent.parent

def calculate_ipc_dynamics(weights_df, close_prices, name):
    """
    Calculation of IPC dynamics based on a weight matrix.

    Args:
        weights_df (pd.DataFrame): Weight matrix (Index: Date, Columns: Ticker).
        close_prices (pd.DataFrame): Close prices of all assets.
        name (str): Name for saving the file (portfolio or snp500).
    """
    print(f"Calculating IPC for {name}...")
    returns = close_prices.pct_change().dropna(axis=0, how="all")
    window = 90
    results = []

    for date in weights_df.index:
        # Take tickers that have a weight on the current date
        row_w = weights_df.loc[date]
        active_tickers = row_w[row_w > 0].index

        # Filter tickers that actually have price data on this date to avoid NaNs
        active_tickers = close_prices.loc[date, active_tickers].dropna().index

        n = len(active_tickers)
        if n < 2:
            continue

        # Slice historical returns for the lookback period
        start_win = date - pd.Timedelta(days=window)
        win_ret = returns.loc[start_win:date, active_tickers].fillna(0)

        if win_ret.empty:
            continue

        # Apply the IPC formula
        corr_matrix = win_ret.corr().values

        # Calculate sum of all correlations minus the diagonal
        total_sum = np.sum(corr_matrix)
        diagonal_sum = np.trace(corr_matrix)

        ipc_val = (total_sum - diagonal_sum) / (n * (n - 1))

        results.append({'Date': date, 'IPC': ipc_val})

    # Final aggregation and metric calculation
    ipc_df = pd.DataFrame(results).set_index('Date')
    mean_val = float(ipc_df['IPC'].mean())
    print(f"Done: {name} IPC mean = {mean_val:.4f}")
    return mean_val


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

def stabilize_size_for_month(size):
    """Выравниваем веса акций в портфеле,
    чтобы они были стабильными с начала по конец каждого месяца

    Args:
        size (pd.DataFrame): Размеры позиций по каждой акции на каждый день

    Returns:
        pd.DataFrame: Обновленные веса бумаг в портфеле
    """
    # Resample to get the first value of each month
    monthly_firsts = size.resample('MS').first()

    # Reindex back to the original index and forward fill
    size_stabilized = monthly_firsts.reindex(size.index, method='ffill')

    return size_stabilized

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
    size = stabilize_size_for_month(size)
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

    # сохраняем метрики Intra-portfolio correlation (IPC)
    ipc = calculate_ipc_dynamics(size, backtest_data.Close, name="portfolio")

    with open(project_path.as_posix() + "/artifacts/metrics/portfolio_ipc.json", 'w') as f:
        json.dump({"mean_snp500_ipc": round(ipc, 4)}, f, indent=4)



if __name__ == "__main__":
    run_backtest()
