import numpy as np
from numba import jit
from scipy import stats


# https://www.quantitativo.com/p/a-mean-reversion-strategy-from-first?utm_source=publication-search

# https://newsletter.huntgathertrade.com/p/the-qp-a-python-interpretation-of

def MyPercentRankAnalog(target_value, target_array):
    return stats.percentileofscore(target_array, target_value)

# @jit(nopython=True)
def PercentRank(target_value, target_array):
    n = len(target_array)
    sorted_indices = np.argsort(target_array)
    sorted_array = target_array[sorted_indices]
    rank = np.searchsorted(sorted_array, target_value, side='right')
    percent_rank = (rank - 1) / (n - 1) * 100 if n > 1 else 100.0

    return percent_rank

# @jit(nopython=True)
def atr(high_prices, low_prices, close_prices, period=252, use_log=True):
	# Calculate the logarithmic returns or standard differences
    if use_log:
        high_log = np.log(high_prices[1:] / high_prices[:-1])
        low_log = np.log(low_prices[1:] / low_prices[:-1])
        close_log = np.log(close_prices[1:] / close_prices[:-1])

        tr1 = high_log - low_log
        tr2 = np.abs(high_log - close_log)
        tr3 = np.abs(low_log - close_log)
    else:
        tr1 = high_prices[1:] - low_prices[1:]
        tr2 = np.abs(high_prices[1:] - close_prices[:-1])
        tr3 = np.abs(low_prices[1:] - close_prices[:-1])

    # Calculate the True Range as the maximum of tr1, tr2, tr3
    true_ranges = np.maximum(np.maximum(tr1, tr2), tr3)

    # Calculate the ATR value as the mean of the last 'period' true ranges
    atr_value = np.mean(true_ranges[-period:])

    return atr_value

# @jit(nopython=True)
def QP(target_high, target_low, target_close, window=3, lookback=1260):
    """
    Calculate the Quantitativo's Probability (QP) indicator for the most recent data point using log returns.

    Parameters:
    - target_high (np.ndarray): Array of high prices for the target security.
    - target_low (np.ndarray): Array of low prices for the target security.
    - target_close (np.ndarray): Array of close prices for the target security.
    - window (int): The number of days for the price change window (default is 3).
    - lookback (int): The number of days to look back for the histogram (default is 1260).

    Returns:
    - float: The QP indicator value for the most recent price.
    """

    n = len(target_close)

    # Replace zeros in target_close with epsilon
    target_close = np.where(target_close == 0, 1.e-20, target_close)

    # Calculate the ATR value for normalization
    atr252 = atr(target_high, target_low, target_close, period=252)

    # Calculate daily log returns and normalize by ATR
    daily_returns = np.log(target_close[1:] / target_close[:-1]) / atr252

    # Calculate the 3-day cumulative log returns using a rolling window sum
    returns = np.convolve(daily_returns, np.ones(window), 'valid')

    # Reverse the array to have the most recent return at index [0]
    returns = returns[::-1]

    # Extract the most recent return
    recent_return = returns[0]

    # Calculate the QP value based on the most recent return
    if recent_return <= 0:
        # Count the number of negative returns within the lookback period
        neg_returns_count = np.sum(returns[:lookback] <= 0)
        total_count = len(returns[:lookback])

        # Area as the ratio of the count of negative returns to the total count
        area_left_of_zero = neg_returns_count / total_count if total_count != 0 else 1.e-20

        # Calculate Percentile Rank
        pctrank = PercentRank(recent_return, returns[:lookback])

        # Normalize by dividing by the "area"
        raw_qp = pctrank / area_left_of_zero
    else:
        # Count the number of positive returns within the lookback period
        pos_returns_count = np.sum(returns[:lookback] > 0)
        total_count = len(returns[:lookback])

        # Area as the ratio of the count of positive returns to the total count
        area_right_of_zero = pos_returns_count / total_count if total_count != 0 else 1.e-20

        # Calculate Percentile Rank
        pctrank = PercentRank(recent_return, returns[:lookback])

        # Normalize by dividing by the "area"
        raw_qp = (100 - pctrank) / area_right_of_zero

    return raw_qp

# print(PercentRank(target_value=1, target_array=np.array([1,3,5])))
# print(MyPercentRankAnalog(target_value=0.99, target_array=np.array([1,3,5])))