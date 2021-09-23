import numbers
import pandas as pd


def column_name_lower(df):
    df.columns = df.columns.str.lower()
    return df


def crossover(s1, s2):
    if isinstance(s2, numbers.Number):
        return (s1 > s2) & (s1.shift() < s2)
    return (s1 > s2) & (s1.shift() < s2.shift())


def crossunder(s1, s2):
    if isinstance(s2, numbers.Number):
        return (s1 < s2) & (s1.shift() > s2)
    return (s1 < s2) & (s1.shift() > s2.shift())