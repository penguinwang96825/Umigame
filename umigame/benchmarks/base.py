import math
import pandas as pd
from empyrical import (
    sharpe_ratio, 
    sortino_ratio, 
    calmar_ratio, 
    max_drawdown, 
    annual_return, 
    annual_volatility
)
from copy import deepcopy
from abc import ABC, abstractmethod
from ..utils import column_name_lower
from ..plotting import *


import os
import sys
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)
from umigame.datasets.crypto import fetch_crypto


class BaseStrategy(ABC):

    def __init__(self):
        super(BaseStrategy, self).__init__()

    @abstractmethod
    def run(self):
        raise NotImplementedError

    def buy(self, date, size, price):
        amount = price * size
        self.position += size
        self.balance -= amount * (1+self.fees)
        self.equity = self.position * price + self.balance
        self.position_history.append(
            ("buy", date, price, size, amount, self.position, self.balance, self.equity)
        )

    def sell(self):
        raise ValueError("Short trading is not supported.")

    def close(self, date, size, price, trigger):
        amount = price * size
        self.balance += amount * (1-self.fees)
        self.equity = self.position * price + self.balance
        self.position = 0
        self.position_history.append(
            (trigger, date, price, size, amount, self.position, self.balance, self.equity)
        )

    def score(self, scoring="sharpe_ratio"):
        if not hasattr(self, "statements"):
            raise NotRunError("Strategy has not been run...")
        rets = self.universe["equity"].pct_change().values
        return metric_from_name(scoring)(rets)

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            fields = [field.name for field in self._meta.fields]
            for field in fields:
                if not getattr(self, field) == getattr(other, field):
                    return False
            return True
        else:
            raise TypeError('Comparing object is not of the same type.')


class BuyAndHoldStrategy(BaseStrategy):

    def __init__(
            self, 
            ticker, 
            capital=10000, 
            fees=0.001, 
            start="2018-08-25", 
            end="2021-08-25", 
            show_progress=True
        ):
        super().__init__()
        self.ticker = ticker
        self.start = start
        self.end = end
        self.show_progress = show_progress

        self.capital = capital
        self.balance = capital
        self.equity = capital
        self.fees = fees
        self.position = 0
        self.position_history = []
        self.statements = None
        self.universe = None
        self.pnl = None

    def _prepare_price_data(self):
        universe = fetch_crypto([self.ticker], self.start, self.end)[self.ticker]
        universe = column_name_lower(universe)
        return universe

    def _get_price_data(self):
        universe = self._prepare_price_data()
        return universe

    def run(self):
        universe = self._get_price_data()
        first_date = universe.index.min()
        last_date = universe.index.max()

        for idx, row in universe.iterrows():
            # Open the position
            if idx == first_date:
                buy_price = row["close"]
                buy_date = pd.to_datetime(idx).date()
                size = self.balance / buy_price
                self.buy(buy_date, size, buy_price)
                if self.show_progress:
                    print(f"Open position @ {buy_price:.4f} on {buy_date}")
            # Close the position
            elif idx == last_date:
                sell_price = row["close"]
                sell_date = pd.to_datetime(idx).date()
                size = self.position
                self.close(sell_date, size, sell_price, "sell")
                if self.show_progress:
                    print(f"Close position @ {sell_price:.4f} on {sell_date}")

        self.statements = pd.DataFrame(
            self.position_history, 
            columns=["trade", "date", "price", "size", "amount", "position", "balance", "equity"]
        )
        self.statements.set_index("date", drop=True, inplace=True)
        universe.loc[self.statements.index, "position"] = self.statements['position']
        universe.loc[self.statements.index, "balance"] = self.statements['balance']
        universe["position"] = universe["position"].ffill().fillna(0)
        universe["balance"] = universe["balance"].ffill().fillna(self.capital)
        universe["equity"] = universe["balance"] + universe["position"] * universe["close"]
        self.universe = deepcopy(universe)


def metric_from_name(name):
    """
    Return metrics from name.
    """
    metrics = (
        sharpe_ratio, 
        sortino_ratio, 
        calmar_ratio, 
        max_drawdown, 
        annual_return, 
        annual_volatility
    )
    return {m.__name__: m for m in metrics}[name]


class NotRunError(ValueError):
    pass