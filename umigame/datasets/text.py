import os
import pandas as pd
from os.path import dirname
from pathlib import Path
from ..utils import column_name_lower


MODULE_PATH = Path(dirname(__file__))


def fetch_news():
    file_path = os.path.join(MODULE_PATH, "text", "news.csv")
    df = pd.read_csv(file_path)
    df = column_name_lower(df)
    return df


def fetch_twitter():
    file_path = os.path.join(MODULE_PATH, "text", "twitter.csv")
    df = pd.read_csv(file_path)
    df = column_name_lower(df)
    return df


def fetch_text(source=None):
    if source == "news":
        df = fetch_news()
    elif source == "twitter":
        df = fetch_twitter()
    else: 
        raise NotImplementedError(f"{source} is not supported.")
    return df