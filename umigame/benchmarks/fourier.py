import pandas as pd
import numpy as np


def fourier_denoise(array, window=10):

    def fft(array):
        array = pd.Series(array)
        length = len(array)
        fhat = np.fft.fft(array, length)
        psd = fhat * np.conj(fhat) / length
        fhat = fhat * (psd > 100)
        ffilt = np.fft.ifft(fhat)
        return np.mean(ffilt)

    return array.rolling(window).apply(fft)