import pandas as pd
import numpy as np
from .breakthrough import BreakthroughStrategy
from ..utils import crossover, crossunder


class FourierStrategy(BreakthroughStrategy):

    def __init__(self, 
            ticker, 
            capital=10000, 
            fees=0.001, 
            fast_period=10, 
            slow_period=20, 
            start="2018-08-25", 
            end="2021-08-25", 
            show_progress=True
        ):
        self.ticker = ticker
        self.capital = capital
        self.fees = fees
        self.slow_period = slow_period
        self.fast_period = fast_period
        self.start = start
        self.end = end
        self.show_progress = show_progress
        self.initialise_local()

    def generate_signals(self, universe):
        universe["fast"] = fourier_denoise(universe["close"], window=self.fast_period)
        universe["slow"] = fourier_denoise(universe["close"], window=self.slow_period)
        entry = crossover(universe["fast"], universe['slow']).astype(int)
        exit = crossunder(universe["fast"], universe['slow']).astype(int)
        entry[entry==0], exit[exit==0] = np.nan, np.nan
        entry, exit = entry[~np.isnan(entry)], exit[~np.isnan(exit)]
        return entry, exit


def fourier_denoise(array, window=10):

    def fft(array):
        array = pd.Series(array)
        n = len(array)
        fhat = np.fft.fft(array, n)
        psd = fhat * np.conj(fhat) / n
        threshold = np.sort(psd)[::-1][n//2]
        fhat = fhat * (psd > threshold)
        ffilt = np.fft.ifft(fhat)
        return np.mean(ffilt)

    return array.rolling(window).apply(fft)