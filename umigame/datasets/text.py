import os
import pandas as pd
from os.path import dirname
from pathlib import Path
from ..utils import column_name_lower
from torch.utils.data import Dataset


MODULE_PATH = Path(dirname(__file__))


def load_dataframe(source=None):
    """
    Parameters
    ----------
    source: str
        Source of text dataframe. {'news', 'twitter', 'all'}
    """
    file_path = os.path.join(MODULE_PATH, "text", "acl2021_news_and_twitter_datasets.csv")
    text_df = pd.read_csv(file_path)
    text_df = column_name_lower(text_df)

    if source == 'news':
        text_df = text_df.query("source=='news'").reset_index(drop=True)
    elif source == 'twitter':
        text_df = text_df.query("source=='twitter'").reset_index(drop=True)
    elif source == 'all':
        text_df = text_df.reset_index(drop=True)
    else:
        print('Please choose between "news", "twitter", and "all".')

    return text_df


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