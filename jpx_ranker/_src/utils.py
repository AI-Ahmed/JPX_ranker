import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from loguru import logger

from IPython.display import clear_output, display


def cal_target(df):
    """
    Calculate target returns based on given periods (1-day return, 2-day returns).

    This function calculates target returns for each security in the provided DataFrame.
    It computes the percentage change in the closing price between the current day and the next day,
    and between the current day and the day after next, representing 1-day and 2-day returns respectively.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data for multiple securities.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated target returns for each security.
    """
    df_ = pd.DataFrame()
    lst_secs = df.SecuritiesCode.unique().tolist()
    prg_bar = tqdm(enumerate(lst_secs), total=len(lst_secs),
                   desc=f'Compute Target Returns')
    for _, sec in prg_bar:
        sec_df = df[df['SecuritiesCode'] == sec].reset_index(drop=True)
        sec_df['Date'] = pd.to_datetime(sec_df['Date'])
        sec_df = sec_df.set_index('Date')

        # ref: https://www.kaggle.com/code/chumajin/easy-to-understand-the-competition?scriptVersionId=94143164&cellId=17
        sec_df["Close_shift1"] = sec_df["Close"].shift(-1)
        sec_df["Close_shift2"] = sec_df["Close"].shift(-2)
        sec_df["Target"] = (sec_df["Close_shift2"] - sec_df["Close_shift1"]) / sec_df["Close_shift1"]
        sec_df = sec_df[['SecuritiesCode', 'Close', 'Volume', 'Target']]

        sec_df = sec_df.dropna().reset_index()
        df_ = pd.concat([df_, sec_df])
    df_ = df_.sort_values(by='Date').reset_index(drop=True)
    return df_


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    Compute Volume-Weighted Average Price (VWAP) and related metrics for
    each security in a DataFrame.

    This function calculates VWAP (Volume-Weighted Average Price) and additional metrics
    for each security in the provided DataFrame. It iterates over each unique ``SecuritiesCode``
    and computes VWAP, one-day backward shifted VWAP, two-day backward shifted VWAP,
    and VWAP rate of change.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data for multiple securities.

    Returns
    -------
    pd.DataFrame
        DataFrame containing VWAP and related metrics for each security.
    """
    df_ = pd.DataFrame()
    lst_sec = df.SecuritiesCode.unique().tolist()

    for sec in tqdm(lst_sec, desc='Compute VWAP'):
        sec_df = df[df['SecuritiesCode'] == sec].reset_index(drop=True)
        close = sec_df.Close
        volume = sec_df.Volume
        sec_df_ = sec_df.copy()
        sec_df_['VWAP'] = (close * volume).cumsum() / volume.cumsum()
        sec_df_['bkwd_sh_1d'] = sec_df_.VWAP.shift(-1) # shift one day backward
        sec_df_['bkwd_sh_2d'] = sec_df_.VWAP.shift(-2) # shift two day backward
        sec_df_['vwap_rate'] = (sec_df_['bkwd_sh_2d'] - sec_df_['bkwd_sh_1d']) / sec_df_['bkwd_sh_1d']
        sec_df_ = sec_df_[['Date', 'SecuritiesCode', 'Close', 'VWAP', 'Target', 'vwap_rate']].dropna().reset_index(drop=True)
        df_ = pd.concat([df_, sec_df_])
    return df_.sort_values(by='Date').reset_index(drop=True)


def compute_feature_eng(df: pd.DataFrame,
                        fast_period: int,
                        slow_period: int
) -> pd.DataFrame:
    """
    Compute feature engineering for each security in a DataFrame.

    This function computes various features for each security in the provided DataFrame,
    including moving averages, volatility measures, and autocorrelation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data for multiple securities.
    fast_period : int
        Number of periods for the fast moving average.
    slow_period : int
        Number of periods for the slow moving average.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the computed features for each security.
        The DataFrame includes additional columns for volatility measures and autocorrelation.
    """
    df_ = pd.DataFrame()
    lst_sec = df.SecuritiesCode.unique().tolist()
    prg_bar = tqdm(enumerate(lst_sec), total= len(lst_sec), desc='Compute Securities Features')
    for idx, sec in prg_bar:
        sec_df = df[df['SecuritiesCode'] == sec].reset_index(drop=True)
        sec_df['Date'] = pd.to_datetime(sec_df['Date'])
        sec_df = sec_df.set_index('Date')

        close = sec_df.Close
        fast_ma = close.rolling(fast_period, min_periods=1).mean()
        slow_ma = close.rolling(slow_period, min_periods=1).mean()

        long_signal = (fast_ma <= slow_ma)
        short_signal = (fast_ma > slow_ma)

        sec_df.loc[long_signal, 'side'] = 1
        sec_df.loc[short_signal, 'side'] = -1

        sec_df['vol5'] =  sec_df['Target'].rolling(5, min_periods=1).std()
        sec_df['vol10'] = sec_df['Target'].rolling(10, min_periods=1).std()
        sec_df['vol15'] = sec_df['Target'].rolling(15, min_periods=1).std()
        sec_df['vol21'] = sec_df['Target'].rolling(21, min_periods=1).std()

        sec_df['autocorr-10'] = sec_df['Target'].rolling(10, min_periods=1).apply(lambda x: pd.Series(x).autocorr(lag=1))
        sec_df['autocorr-15'] = sec_df['Target'].rolling(15, min_periods=1).apply(lambda x: pd.Series(x).autocorr(lag=1))
        sec_df['autocorr-21'] = sec_df['Target'].rolling(21, min_periods=1).apply(lambda x: pd.Series(x).autocorr(lag=1))

        sec_df = sec_df.dropna().reset_index()
        df_ = pd.concat([df_, sec_df])

        if idx % 5 == 0 and hasattr(prg_bar, 'container'):
            clear_output(wait=True)
            display(prg_bar.container)

    df_ = df_.sort_values(by='Date').reset_index(drop=True)
    return df_


def _calc_spread_return_per_day(df: pd.DataFrame, 
                                 portfolio_size: int = 200, 
                                 toprank_weight_ratio: float = 2) -> float:
    """
    Calculate the spread return for a single day.
    
    This is a shared utility function used across multiple modules for computing
    daily spread returns in ranking strategies.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame for a single date containing 'Rank' and 'Target' columns.
    portfolio_size : int, optional
        Number of equities to buy/sell (default: 200).
    toprank_weight_ratio : float, optional
        The relative weight of the most highly ranked stock compared to the least (default: 2).
    
    Returns
    -------
    float
        Spread return for the day (long portfolio return - short portfolio return).
        
    Notes
    -----
    This function validates that ranks start from 0 and are contiguous up to N-1.
    Weights decrease linearly from toprank_weight_ratio to 1 across the portfolio.
    """
    # Convert to scalar safely for pandarallel compatibility
    rank_min = df['Rank'].min()
    rank_max = df['Rank'].max()
    
    # Convert to Python int - works with pandas Series, numpy scalars, and Python types
    try:
        rank_min = int(rank_min)
        rank_max = int(rank_max)
    except (TypeError, ValueError):
        # Fallback for edge cases in parallel processing
        rank_min = int(np.asarray(rank_min).flat[0])
        rank_max = int(np.asarray(rank_max).flat[0])
    
    expected_max = len(df) - 1
    
    assert rank_min == 0, f"Rank min should be 0, got {rank_min}"
    assert rank_max == expected_max, f"Rank max should be {expected_max}, got {rank_max}"
    
    # Adjust portfolio size if there are fewer stocks than requested
    actual_portfolio_size = min(portfolio_size, len(df))
    
    # Generate weights based on actual portfolio size
    weights = np.linspace(start=toprank_weight_ratio, stop=1, num=actual_portfolio_size)
    
    # Calculate long (purchase) and short positions
    purchase = (df.sort_values(by='Rank')['Target'][:actual_portfolio_size] * weights).sum() / weights.mean()
    short = (df.sort_values(by='Rank', ascending=False)['Target'][:actual_portfolio_size] * weights).sum() / weights.mean()
    return purchase - short


def calc_spread_return_sharpe(df: pd.DataFrame,
                              portfolio_size: int = 200,
                              toprank_weight_ratio: float = 2) -> float:
    """
    Calculate the Sharpe ratio of a long-short spread return strategy.
    
    This function implements a market-neutral strategy that goes long on the top-ranked
    securities and short on the bottom-ranked securities, with linearly decreasing weights.
    The Sharpe ratio is computed across all dates to measure risk-adjusted performance.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing predicted results with columns 'Date', 'Rank', and 'Target'.
        Must have ranks starting from 0 to n-1 for each date group.
    portfolio_size : int, optional
        Number of equities to buy (long) and sell (short) in the portfolio (default: 200).
    toprank_weight_ratio : float, optional
        The relative weight of the most highly ranked stock compared to the least ranked
        stock in the portfolio. Weights decrease linearly from this ratio to 1 (default: 2).
    
    Returns
    -------
    float
        Sharpe ratio of the spread return strategy, calculated as mean daily spread return
        divided by standard deviation of daily spread returns.
    
    Notes
    -----
    The spread return for each day is calculated as:
    spread_return = weighted_long_return - weighted_short_return
    
    Where weights are linearly spaced from toprank_weight_ratio to 1.
    """
    try:
        # Try parallel processing first
        buf = df.groupby('Date').parallel_apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    except (AttributeError, RuntimeError) as e:
        # Fallback to regular apply if parallel processing fails (e.g., on Kaggle)
        logger.warning(f"parallel_apply failed ({e}), falling back to regular apply")
        buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio


def calc_spread_return_sharpe_scorer(actual_df: pd.DataFrame, predicted_df: pd.DataFrame,
                                     portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Compute the difference in spread return Sharpe ratio between actual and predicted rankings.
    
    This function evaluates the quality of predicted rankings by comparing the Sharpe ratio
    of spread returns (long-short portfolio returns) between actual and predicted rankings.
    The spread return is calculated by taking a long position in the top-ranked stocks and
    a short position in the bottom-ranked stocks, with linearly decreasing weights.
    
    Parameters
    ----------
    actual_df : pd.DataFrame
        DataFrame containing actual rankings and target returns. Must include columns:
        - 'Date': Trading date
        - 'Rank': Actual ranking (0 to N-1, where 0 is best)
        - 'Target': Actual target returns
    predicted_df : pd.DataFrame
        DataFrame containing predicted rankings and target returns. Must include columns:
        - 'Date': Trading date
        - 'Rank': Predicted ranking (0 to N-1, where 0 is best)
        - 'Target': Actual target returns (same as actual_df)
    portfolio_size : int, optional
        Number of equities to include in each side (long/short) of the portfolio.
        Default is 200.
    toprank_weight_ratio : float, optional
        The relative weight of the most highly ranked stock compared to the least
        highly ranked stock in the portfolio. Weights decrease linearly from
        toprank_weight_ratio to 1.0. Default is 2.0.
    
    Returns
    -------
    float
        Absolute difference in spread return Sharpe ratios between actual and predicted
        rankings. Lower values indicate better prediction quality.
    
    Notes
    -----
    The spread return for each day is calculated as:
    
    .. math::
        SR_t = \\frac{\\sum_{i=1}^{N} w_i \\cdot r_{i,long}}{\\bar{w}} - \\frac{\\sum_{i=1}^{N} w_i \\cdot r_{i,short}}{\\bar{w}}
    
    where :math:`w_i` are linearly decreasing weights, :math:`r_{i,long}` are returns of
    top-ranked stocks, :math:`r_{i,short}` are returns of bottom-ranked stocks, and
    :math:`\\bar{w}` is the mean weight.
    
    The Sharpe ratio is then computed as:
    
    .. math::
        Sharpe = \\frac{\\mu(SR)}{\\sigma(SR)}
    
    where :math:`\\mu(SR)` is the mean spread return and :math:`\\sigma(SR)` is the
    standard deviation of spread returns across all dates.
    
    Examples
    --------
    >>> actual_df = pd.DataFrame({
    ...     'Date': ['2021-01-01', '2021-01-01', '2021-01-02', '2021-01-02'],
    ...     'Rank': [0, 1, 0, 1],
    ...     'Target': [0.05, -0.02, 0.03, -0.01]
    ... })
    >>> predicted_df = pd.DataFrame({
    ...     'Date': ['2021-01-01', '2021-01-01', '2021-01-02', '2021-01-02'],
    ...     'Rank': [0, 1, 1, 0],
    ...     'Target': [0.05, -0.02, 0.03, -0.01]
    ... })
    >>> score = calc_spread_return_sharpe_scorer(actual_df, predicted_df, portfolio_size=1)
    >>> print(f"Sharpe difference: {score:.4f}")
    """
    # Calculate spread return Sharpe ratio for actual and predicted rankings
    try:
        # Try parallel processing first
        actual_buf = actual_df.groupby('Date').parallel_apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
        predicted_buf = predicted_df.groupby('Date').parallel_apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    except (AttributeError, RuntimeError) as e:
        # Fallback to regular apply if parallel processing fails (e.g., on Kaggle)
        logger.warning(f"parallel_apply failed ({e}), falling back to regular apply")
        actual_buf = actual_df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
        predicted_buf = predicted_df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)

    actual_sharpe_ratio = actual_buf.mean() / actual_buf.std()
    predicted_sharpe_ratio = predicted_buf.mean() / predicted_buf.std()

    # Return absolute difference in Sharpe ratios
    return abs(actual_sharpe_ratio - predicted_sharpe_ratio)
