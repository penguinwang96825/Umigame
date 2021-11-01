import pandas as pd
import yfinance as yf
from .helper import fill_and_cut
from ..utils import column_name_lower


CRYPTO = {
    'BTC-USD': 'Bitcoin USD', 
    'ETH-USD': 'Ethereum USD', 
    'BNB-USD': 'Binance USD', 
    'DOGE-USD': 'Dogecoin USD', 
    'USDT-USD': 'Tether USD', 
	'LTC-USD': 'Litecoin USD', 
    'LINK-USD': 'Chainlink USD', 
    'USDC-USD': 'USDCoin USD', 
    'THETA-USD': 'ThETA USD', 
    'XMR-USD': 'Monero USD'
}
TODAY_DATE = pd.to_datetime('today').date()


def fetch_crypto(tickers=None, start="2012-01-01", end=TODAY_DATE, interval="1d", fill=True):
    """
    Parameters
    ----------
    tickers: list
    start: str
    end: str
    interval: str
    fill: bool

    Returns
    -------
    prices_dict: dict
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if tickers is None:
        tickers = list(CRYPTO.keys())

    # Download prices information from Yahoo Finance API
    format = '{l_bar}{bar:40}{r_bar}{bar:-40b}'
    prices_dict = {
        t: yf.download(t, start=start, end=end, interval=interval, progress=False) 
        for t in tickers
    }
    prices_dict = {
        t: column_name_lower(df)
        for t, df in prices_dict.items()
    }

    # Fill the prices' index with same start date and end date
    if fill:
        prices_dict = {
            ticker: fill_and_cut(price, start=start)
            for ticker, price in prices_dict.items()
        }

    return prices_dict