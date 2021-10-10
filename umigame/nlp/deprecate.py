import talib
import itertools
import pandas as pd


def generate_feature(price_df):
    high = price_df["high"].values
    low = price_df["low"].values
    close = price_df["close"].values

    original_columns = price_df.columns
    feature_df = price_df.copy()
    for t in range(7, 22):
        series = pd.Series(talib.ADX(high, low, close, timeperiod=t), name=f"ADX_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(7, 22):
        series = pd.Series(talib.ADXR(high, low, close, timeperiod=t), name=f"ADXR_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for f, s in itertools.product(range(10, 15), range(25, 30)):
        series = pd.Series(talib.APO(close, fastperiod=f, slowperiod=s, matype=0), name=f"APO_{f}_{s}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(7, 22):
        series = pd.Series(talib.AROONOSC(high, low, timeperiod=t), name=f"AROONOSC_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(7, 22): 
        series = pd.Series(talib.CCI(high, low, close, timeperiod=t), name=f"CCI_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(7, 22): 
        series = pd.Series(talib.CMO(close, timeperiod=t), name=f"CMO_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(7, 22):
        series = pd.Series(talib.DX(high, low, close, timeperiod=t), name=f"DX_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(7, 22):
        series = pd.Series(talib.MINUS_DI(high, low, close, timeperiod=t), name=f"MINUS_DI_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(7, 22):
        series = pd.Series(talib.MINUS_DM(high, low, timeperiod=t), name=f"MINUS_DM_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(7, 22):
        series = pd.Series(talib.MOM(close, timeperiod=t), name=f"MOM_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(7, 22):
        series = pd.Series(talib.PLUS_DI(high, low, close, timeperiod=t), name=f"PLUS_DI_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(7, 22):
        series = pd.Series(talib.PLUS_DM(high, low, timeperiod=t), name=f"PLUS_DM_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for f, s in zip(range(10, 15), range(25, 30)):
        series = pd.Series(talib.PPO(close, fastperiod=f, slowperiod=s, matype=0), name=f"PPO_{f}_{s}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(7, 22):
        series = pd.Series(talib.ROC(close, timeperiod=t), name=f"ROC_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(7, 22):
        series = pd.Series(talib.ROCP(close, timeperiod=t), name=f"ROCP_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(6, 22):
        series = pd.Series(talib.ROCR100(close, timeperiod=t), name=f"ROCR100_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(5, 22):
        series = pd.Series(talib.RSI(close, timeperiod=t), name=f"RSI_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t1, t2, t3 in itertools.product(range(5, 10), range(10, 15), range(25, 30)):
        series = pd.Series(talib.ULTOSC(high, low, close, timeperiod1=t1, timeperiod2=t2, timeperiod3=t3), name=f"ULTOSC_{t1}_{t2}_{t3}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    for t in range(5, 22):
        series = pd.Series(talib.WILLR(high, low, close, timeperiod=t), name=f"WILLR_{t}", index=feature_df.index)
        feature_df = pd.concat((feature_df, series), axis=1)
    feature_df = feature_df.bfill()

    # Exclude columns you don't want
    feature_df = feature_df[feature_df.columns[~feature_df.columns.isin(original_columns)]]

    return feature_df