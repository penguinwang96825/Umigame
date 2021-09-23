import math
import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .base import BaseStrategy
from ..utils import column_name_lower, crossover, crossunder


class TurtleStrategy(BaseStrategy):

    def __init__(self, universe, window_up=20, window_down=10, window_atr=20):
        super().__init__()
        self.universe = universe
        self.initialise(window_up, window_down, window_atr)

    def initialise(self, window_up, window_down, window_atr):
        self.universe = column_name_lower(self.universe)
        # Compute returns
        self.universe['ret'] = self.universe["close"] / self.universe["close"].shift(1) - 1
        self.universe['ret'][0] = 0

        # Donchian Channel
        self.universe['dc_high'] = talib.MAX(self.universe["high"], timeperiod=window_up).shift(1)
        self.universe['dc_low'] = talib.MIN(self.universe["low"], timeperiod=window_down).shift(1)
        self.universe['atr'] = talib.ATR(
            self.universe["high"], 
            self.universe["low"], 
            self.universe["close"], 
            timeperiod=window_atr
        )
        
    def run(self, capital=10000, fee=0.001, position=0, quota=0.5, verbose=False):
        entry, exit = self.generate_signals(self.universe)
        self.trades = self.trade(self.universe, entry, exit, capital, fee, position, quota, verbose)

    def generate_signals(self, universe):
        # Compute signal based on Donchian Channel
        entry = crossover(universe["close"], universe['dc_high']).astype(int)
        exit = crossunder(universe["close"], universe['dc_low']).astype(int)
        entry[entry==0], exit[exit==0] = np.nan, np.nan
        entry, exit = entry[~np.isnan(entry)], exit[~np.isnan(exit)]
        return entry, exit

    def trade(self, universe, entry, exit, capital=10000, fee=0.001, position=0, quota=0.5, verbose=True):
        universe.loc[(entry==1).index, 'signal'] = 1
        universe.loc[(exit==1).index, 'signal'] = 0
        balance = capital
        for idx, row in universe.iterrows():
            if row["signal"] == 1:
                if balance > 0:
                    buy_price = row["close"]
                    buy_date = pd.to_datetime(idx).date()
                    amount = math.floor(balance * quota / buy_price)
                    balance = balance - amount * buy_price * (1+fee)
                    position = position + amount
                    if verbose:
                        print(f"Buy {amount * buy_price:.4f} at {buy_price:.4f} on {buy_date}")
                        print(f"Balance: {balance:.4f} Position: {position:.4f}")
                        print("-"*50)
            elif row["signal"] == 0:
                if position > 1:
                    sell_price = row["close"]
                    sell_date = pd.to_datetime(idx).date()
                    amount = math.floor(position * 0.5)
                    balance = balance + amount * sell_price * (1+fee)
                    position = position - amount
                    if verbose:
                        print(f"Sell {amount * sell_price:.4f} at {sell_price:.4f} on {sell_date}")
                        print(f"Balance: {balance:.4f} Position: {position:.4f}")
                        print("-"*50)
            universe.loc[idx, "value"] = balance + position * row["close"]
            universe.loc[idx, "balance"] = balance
            universe.loc[idx, "position"] = position
        return universe

    def plot(self):

        def align_xticks_for_df(df, baseline):
            return df.reindex(baseline.index).bfill()

        entry_date = self.universe[self.universe['signal']==1].index
        exit_date = self.universe[self.universe['signal']==0].index

        fig = plt.figure(figsize=(15, 8), constrained_layout=True)
        gs = gridspec.GridSpec(6, 3, figure=fig)
        ax1 = plt.subplot(gs[0:2, :])
        ax2 = plt.subplot(gs[2:4, :])
        ax3 = plt.subplot(gs[4, :])
        ax4 = plt.subplot(gs[5, :])

        ax1.plot(self.universe.close, label="Close Price")
        ax1.plot(self.universe['dc_high'], '#B97A95', label="DC-HIGH", alpha=1, linestyle="--")
        ax1.plot(self.universe['dc_low'], '#F6AE99', label="DC-LOW", alpha=1, linestyle="--")
        ax1.scatter(entry_date, self.universe.close[entry_date], label="buy", marker="^", alpha=.7, color="tab:green")
        ax1.scatter(exit_date, self.universe.close[exit_date], label="sell", marker="v", alpha=.7, color="tab:red")
        ax1.set_title("Entry and exit signals")
        ax1.legend(loc="upper left")
        ax1.grid()

        ax2.plot(align_xticks_for_df(self.universe['value'], self.universe))
        ax2.set_title("Portfolio value")
        ax2.grid()

        ax3.plot(align_xticks_for_df(self.universe['balance'], self.universe))
        ax3.set_title("Balance")
        ax3.grid()

        ax4.plot(align_xticks_for_df(self.universe['position'], self.universe))
        ax4.set_title("Position")
        ax4.grid()

        plt.show()
