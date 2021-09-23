import numpy as np
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