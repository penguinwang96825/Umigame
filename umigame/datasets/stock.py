import pandas as pd
import yfinance as yf
from os.path import dirname
from pathlib import Path
from ..utils import column_name_lower
from .helper import fill_and_cut


STOCK = [
    "AAPL", "MSFT", "AMZN", "FB", "GOOGL", 
    "GOOG", "TSLA", "BRK-B", "NVDA", "JPM", 
    "JNJ", "V", "UNH", "PYPL", "HD", 
    "PG", "MA", "DIS", "BAC", "ADBE", 
    "XOM", "CMCSA", "NFLX", "VZ", "INTC"
]
MODULE_PATH = Path(dirname(__file__))
TODAY_DATE = pd.to_datetime('today').date()


def fetch_usstock(tickers=None, csv_file=None, start="2012-01-01", end=TODAY_DATE, interval="1d", fill=True):
    """
    Parameters
    ----------
    tickers: list
    csv_file: str
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

    # Get constituents from file or default
    if tickers is None:
        if csv_file:
            constituents = pd.read_csv(MODULE_PATH / csv_file)
            constituents = column_name_lower(constituents)
            tickers = constituents["symbol"].tolist()
        else: 
            tickers = STOCK

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