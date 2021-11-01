import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .order import Trade, Order
from abc import ABC, abstractmethod
pd.options.mode.chained_assignment = None


class Strategy(ABC):
    """
    Examples
    --------
    >>> price_df = pd.DataFrame({
    ...     "close": [10, 20, 30, 40, 20], 
    ...     "entry": [0, 1, 1, 0, 1], 
    ...     "exit": [0, 0, 0, 1, 0]
    ... }, index=["2021-08-25", "2021-08-26", "2021-08-27", "2021-08-28", "2021-08-29"])

    >>> strategy = Strategy(initial_cash=100, fees=0.001)
    >>> strategy.from_signals(price_df, price_df.entry, price_df.exit)
    """
    def __init__(self, initial_cash=100.0, fees=0.001):
        super(Strategy, self).__init__()
        self.initial_cash = initial_cash
        self.fees = fees
        self.trade = Trade(initial_cash=self.initial_cash, positions=0)

    def from_signals(self, asset, entries, exits, stop_loss=0.0, take_profit=0.0, verbose=1):
        self.asset = asset
        self.verbose = verbose
        self.entries = entries.astype(int)
        self.exits = exits.astype(int)
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.stop_loss_price = 0
        self.take_profit_price = 0
        self.asset['entry'] = self.entries
        self.asset['exit'] = self.exits
        self.backtest(self.asset)

    def backtest(self, asset: pd.DataFrame):
        asset.columns = asset.columns.str.lower()
        for date, row in asset.iterrows():
            self.logic(date, row)
    
    @abstractmethod
    def logic(self, date, row):
        raise NotImplementedError

    @property
    def statement(self):
        return self.trade.statement.set_index('date')

    @property
    def entry_record(self):
        return self.trade.entry_record

    @property
    def exit_record(self):
        return self.trade.exit_record

    def plot(self):
        equity = self.statement['equity']
        timeframe = self.statement.index
        entry_date = self.statement.loc[self.statement['entry']==1].index
        exit_date = self.statement.loc[self.statement['exit']==1].index

        plt.figure(figsize=(15, 6))
        plt.subplot(2, 1, 1)
        plt.plot(timeframe, equity)
        plt.title('Equity')
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.plot(timeframe, self.statement.price, label='close price')
        plt.scatter(entry_date, self.statement.price[entry_date], label="buy", marker="^", alpha=.7, color="tab:green")
        plt.scatter(exit_date, self.statement.price[exit_date], label="sell", marker="v", alpha=.7, color="tab:red")
        plt.legend()
        plt.title('Price')
        plt.grid()
        plt.tight_layout()
        plt.show()


class BaseStrategy(Strategy):

    def logic(self, date, row):
        current_price = row['close']
        current_date = date
        size = 1
        # Trigger entry signal
        if row["entry"] != 0:
            size = size * 1
            entry_fees = current_price * size * self.fees
            entry_oder = Order(size, current_date, current_price, entry_fees)
            self.trade.add_entry_order(entry_oder)
            self.trade.update_statement(current_date, current_price, 1, 0, 0, 0, size, entry_fees)
            if (self.stop_loss != 0.0) and (self.stop_loss is not None):
                self.stop_loss_price = current_price * (1-self.stop_loss)
            if (self.take_profit != 0.0) and (self.take_profit is not None):
                self.take_profit_price = current_price * (1+self.take_profit)
            if self.verbose:
                print(
                    f'{current_date}: Open position @ {current_price:.4f} with size {size} '
                    f'TP {self.take_profit_price} SL {self.stop_loss_price}'
                )
        # Trigger exit signal
        elif row["exit"] != 0:
            size = size * 1
            exit_fees = current_price * size * self.fees
            exit_order = Order(size, current_date, current_price, exit_fees)
            self.trade.add_exit_order(exit_order)
            self.trade.update_statement(current_date, current_price, 0, 1, 0, 0, -size, exit_fees)
            if self.verbose:
                print(f'{current_date}: Close position @ {current_price:.4f} with size {size}')
        elif (self.take_profit != 0.0) and (self.take_profit is not None) and (self.stop_loss != 0.0) and (self.stop_loss is not None):
            # Trigger take profit price
            if (current_price > self.take_profit_price) and (self.trade.positions > 0):
                if (self.take_profit != 0.0) and (self.take_profit is not None):
                    size = size * 1
                    exit_fees = current_price * size * self.fees
                    exit_order = Order(size, current_date, current_price, exit_fees)
                    self.trade.add_exit_order(exit_order)
                    self.trade.update_statement(current_date, current_price, 0, 0, 0, 1, -size, exit_fees)
                    if self.verbose:
                        print(f'{current_date}: Close position @ {current_price:.4f} with size {size} (TP)')
            # Trigger stop loss price
            elif (current_price < self.stop_loss_price) and (self.trade.positions > 0):
                if (self.stop_loss != 0.0) and (self.stop_loss is not None):
                    size = size * 1
                    exit_fees = current_price * size * self.fees
                    exit_order = Order(size, current_date, current_price, exit_fees)
                    self.trade.add_exit_order(exit_order)
                    self.trade.update_statement(current_date, current_price, 0, 0, 1, 0, -size, exit_fees)
                    if self.verbose:
                        print(f'{current_date}: Close position @ {current_price:.4f} with size {size} (SL)')
        else:
            self.trade.update_statement(current_date, current_price, 0, 0, 0, 0)

