import os
import pandas as pd
import yfinance as yf
from os.path import dirname
from pathlib import Path
from ..utils import column_name_lower


MODULE_PATH = Path(dirname(__file__))


def load_dataset(name=None, split=None, split_time='2018'):
    file_path = os.path.join(MODULE_PATH, "text", "acl2021_news_and_twitter_datasets.csv")
    text_df = pd.read_csv(file_path)
    text_df = column_name_lower(text_df)
    if name == 'news':
        text_df = text_df.query("source=='news'").reset_index(drop=True)
    elif name == 'twitter':
        text_df = text_df.query("source=='twitter'").reset_index(drop=True)

    if split == 'train':
        train_text_df = text_df.loc[:split_time]
        return train_text_df
    elif split == 'test':
        test_text_df = text_df.loc[split_time:]
        return test_text_df
    else:
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