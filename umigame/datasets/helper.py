import pandas as pd


def fill_and_cut(series, start=None, method="ffill"):
    f_index = pd.date_range(series.index[0], series.index[-1])
    n_index = pd.date_range(start or series.index[0], series.index[-1])
    return series.reindex(f_index, method=method).reindex(n_index)
