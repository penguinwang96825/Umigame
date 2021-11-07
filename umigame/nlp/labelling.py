import math
import numpy as np
import pandas as pd


def fixed_time_horizon(df, column='close', lookback=20):
    """
    Fixed-time Horizon
    As it relates to finance, virtually all ML papers label observations using the fixed-time horizon method.
    Fixed-time horizon is presented as one of the main procedures to label data when it comes to processing 
    financial time series for machine learning.

    Parameters
    ----------
    df: pd.DataFrame
    column: str
        Choose from "open", "high", "low", and "close."
    lookahead: str
        The number of days to look ahead.

    References
    ----------
    1. https://mlfinlab.readthedocs.io/en/latest/labeling/labeling_fixed_time_horizon.html
    2. https://arxiv.org/pdf/1603.08604.pdf
    3. https://quantdare.com/4-simple-ways-to-label-financial-data-for-machine-learning/
    4. De Prado, Advances in financial machine learning, 2018
    5. Dixon et al., Classification-based financial markets prediction using deep neural networks, 2017
    """
    price = df[column]
    label = (price.shift(-lookback) / price > 1).astype(int)
    return label


def triple_barrier(df, column='close', ub=0.07, lb=0.03, lookback=20, binary_classification=True):
    """
    Triple Barrier
    The idea is to consider the full dynamics of a trading strategy and not a simple performance proxy. 
    The rationale for this extension is that often money managers implement P&L triggers that cash in 
    when gains are sufficient or opt out to stop their losses. Upon inception of the strategy, 
    three barriers are fixed (De Prado, 2018).
    
    Parameters
    ----------
    df: pd.DataFrame
    column: str
        Choose from "open", "high", "low", and "close."
    ub: float
        It stands for upper bound, e.g. 0.07 is a 7% profit taking.
    lb: float
        It stands for lower bound, e.g. 0.03 is a 3% stop loss.
    lookback: str
        Maximum holding time.

    References
    ----------
    1. https://www.finlab.tw/generate-labels-stop-loss-stop-profit/
    2. http://www.mlfactor.com/Data.html#the-triple-barrier-method
    3. https://chrisconlan.com/calculating-triple-barrier-labels-from-advances-in-financial-machine-learning/
    4. https://towardsdatascience.com/financial-machine-learning-part-1-labels-7eeed050f32e
    5. De Prado, Advances in financial machine learning, 2018
    """
    ub = 1 + ub
    lb = 1- lb

    def end_price(s):
        return np.append(s[(s / s[0] > ub) | (s / s[0] < lb)], s[-1])[0]/s[0]

    r = np.array(range(lookback))

    def end_time(s):
        return np.append(r[(s / s[0] > ub) | (s / s[0] < lb)], lookback-1)[0]

    price = df[column]
    p = price.rolling(lookback).apply(end_price, raw=True).shift(-lookback+1)
    t = price.rolling(lookback).apply(end_time, raw=True).shift(-lookback+1)
    t = pd.Series(
        [t.index[int(k+i)] if not math.isnan(k+i) else np.datetime64('NaT') 
        for i, k in enumerate(t)], index=t.index
    ).dropna()

    label = pd.Series(0, p.index)
    label.loc[p > ub] = 1
    label.loc[p < lb] = -1
    if binary_classification:
        label = np.where(label == 1, 1, 0)

    return pd.Series(label, index=price.index)


def get_continuous_trading_signals(df, column='close', lookahead=5):
    """
    Continuous Trading Signal
    A hybrid stock trading framework integrating technical analysis with machine learning techniques.

    Parameters
    ----------
    df: pd.DataFrame
    column: str
        Choose from "open", "high", "low", and "close."
    lookahead: str
        The number of days to look ahead.
        
    References
    ----------
    1. https://translateyar.ir/wp-content/uploads/2020/05/1-s2.0-S2405918815300179-main-1.pdf
    2. Dash and Dash, A hybrid stock trading framework integrating technical analysis with machine learning techniques, 2016
    """
    price = df.data[column]
    OTr = []
    trends = []
    for idx in range(len(price)-lookahead+1):
        arr_window = price[idx:(idx+lookahead)]
        if price[idx+lookahead-1] > price[idx]:
            coef = (price[idx+lookahead-1]-min(arr_window)) / (max(arr_window)-min(arr_window))
            y_t = coef * 0.5 + 0.5
        elif price[idx+lookahead-1] <= price[idx]:
            coef = (price[idx+lookahead-1]-min(arr_window)) / (max(arr_window)-min(arr_window))
            y_t = coef * 0.5
        OTr.append(y_t)
    OTr = np.append(OTr, np.zeros(shape=(len(price)-len(OTr))))
    trends = (OTr >= np.mean(OTr)).astype(int)
    return pd.Series(OTr, index=price.index), pd.Series(trends, index=price.index)