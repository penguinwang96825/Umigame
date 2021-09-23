import numpy as np
import pandas as pd
import empyrical as ep


EPSILON = 0.0000001
TRADING_DAYS = 252
RISK_FREE_RATE = 0.01


def sharpe_ratio(returns):
    return returns.mean() / (returns.std()+EPSILON) * (TRADING_DAYS ** 0.5)


def sortino_ratio(returns):
    mean_ = returns.mean() * TRADING_DAYS - RISK_FREE_RATE
    std_ = returns[returns<0].std() * (252 ** 0.5)
    return mean_ / std_


def calmars_ratio(returns):
    return returns.mean() * TRADING_DAYS / abs(max_drawdown(returns))


def max_drawdown(returns):
    comp = (returns+1).cumprod()
    peak = comp.expanding(min_periods=1).max()
    dd = comp / peak - 1
    return dd.min()


def drawdown(returns):

    def _compute_drawdown_duration_peaks(dd: pd.Series):
        iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
        iloc = pd.Series(iloc, index=dd.index[iloc])
        df = iloc.to_frame('iloc').assign(prev=iloc.shift())
        df = df[df['iloc'] > df['prev'] + 1].astype(int)
        # If no drawdown since no trade, avoid below for pandas sake and return nan series
        if not len(df):
            return (dd.replace(0, np.nan),) * 2
        df['duration'] = df['iloc'].map(dd.index.__getitem__) - df['prev'].map(dd.index.__getitem__)
        df['peak_dd'] = df.apply(lambda row: dd.iloc[row['prev']:row['iloc'] + 1].max(), axis=1)
        df = df.reindex(dd.index)
        return df['duration'], df['peak_dd']

    def _data_period(index):
        """Return data index period as pd.Timedelta"""
        values = pd.Series(index[-100:])
        return values.diff().median()

    def _round_timedelta(value, _period=_data_period(returns.index)):
        if not isinstance(value, pd.Timedelta):
            return value
        resolution = getattr(_period, 'resolution_string', None) or _period.resolution
        return value.ceil(resolution)

    dd = 1 - returns / np.maximum.accumulate(returns)
    dd_dur, dd_peaks = _compute_drawdown_duration_peaks(pd.Series(dd, index=returns.index))
    max_dd = -np.nan_to_num(dd.max()) * 100
    mean_dd = -dd_peaks.mean() * 100
    max_dd_duration = _round_timedelta(dd_dur.max())
    mean_dd_duration = _round_timedelta(dd_dur.mean())
    return max_dd, mean_dd, max_dd_duration, mean_dd_duration


def annualise_returns(returns, periods_per_year=252):
    """
    https://www.investopedia.com/terms/a/annualized-total-return.asp
    """
    compounded_growth = (1+returns).prod()
    n_periods = returns.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualise_volatility(returns, periods_per_year=252):
    """
    https://breakingdownfinance.com/finance-topics/finance-basics/annualize-volatility/
    """
    return returns.std()*(periods_per_year**0.5)