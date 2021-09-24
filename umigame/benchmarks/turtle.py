import math
import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .base import BaseStrategy
from ..utils import column_name_lower, crossover, crossunder
from ..plotting import *


class TurtleStrategy(BaseStrategy):

    RISK_COEFFICIENT = 0.01

    def __init__(self, universe, window_up=20, window_down=10, window_atr=20):
        super(TurtleStrategy, self).__init__()
        self.universe = universe
        self.statements = None
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
        
    def run(self, capital=10000, fee=0.001, stop_loss=0.1, take_profit=0.5, position=0, verbose=False):
        entry, exit = self.generate_signals(self.universe)
        self.statements = self.trade(
            self.universe, 
            entry, 
            exit, 
            capital, 
            fee, 
            stop_loss, 
            take_profit, 
            position, 
            verbose
        )

    def generate_signals(self, universe):
        # Compute signal based on Donchian Channel
        entry = crossover(universe["close"], universe['dc_high']).astype(int)
        exit = crossunder(universe["close"], universe['dc_low']).astype(int)
        entry[entry==0], exit[exit==0] = np.nan, np.nan
        entry, exit = entry[~np.isnan(entry)], exit[~np.isnan(exit)]
        return entry, exit

    def trade(
            self, 
            universe, 
            entry, exit, 
            capital=10000, 
            fee=0.001, 
            stop_loss=0.1, 
            take_profit=0.5, 
            total_amount=0, 
            verbose=True
        ):
        universe.loc[(entry==1).index, 'signal'] = 1
        universe.loc[(exit==1).index, 'signal'] = 0
        universe["value"] = capital
        balance = capital
        buy_price, sell_price = None, None
        for idx, row in universe.iterrows():
            # Stop loss
            if (buy_price is not None) and (stop_loss is not None) and (row["close"] < buy_price*(1.0-stop_loss)):
                stop_loss_price = row["close"]
                stop_loss_date = pd.to_datetime(idx).date()
                if row["close"] < buy_price-2*row["atr"]:
                    balance += stop_loss_price * total_amount * (1-fee)
                    total_amount = 0
                else:
                    balance += stop_loss_price * math.floor(total_amount * 0.5) * (1-fee)
                    total_amount -= math.floor(total_amount * 0.5)
                if verbose:
                    print(f"Stop loss at price {stop_loss_price:.4f} on {stop_loss_date}")
                    print(f"Last buy at {buy_price:.4f}")
                    print(f"Balance: {balance:.4f} Total amount: {total_amount:.4f}")
                    print("-"*50)
            # Take profit
            elif (buy_price is not None) and (take_profit is not None) and (row["close"] >= buy_price*(1.0+take_profit)):
                take_profit_price = row["close"]
                take_profit_date = pd.to_datetime(idx).date()
                balance += take_profit_price * math.floor(total_amount * 0.5) * (1-fee)
                total_amount -= math.floor(total_amount * 0.5)
                if verbose:
                    print(f"Take profit at price {take_profit_price:.4f} on {take_profit_date}")
                    print(f"Last buy at {take_profit_date:.4f}")
                    print(f"Balance: {balance:.4f} Total amount: {total_amount:.4f}")
                    print("-"*50)

            # Buy signal
            if row["signal"] == 1:
                if balance > 0:
                    buy_price = row["close"]
                    buy_date = pd.to_datetime(idx).date()
                    unit = row["value"] * self.RISK_COEFFICIENT / row["atr"]
                    balance = balance - unit * buy_price * (1+fee)
                    total_amount = total_amount + unit
                    if verbose:
                        print(f"Buy {unit * buy_price:.4f} at {buy_price:.4f} on {buy_date}")
                        print(f"Balance: {balance:.4f} Total amount: {total_amount:.4f}")
                        print("-"*50)
            # Sell signal
            elif row["signal"] == 0:
                if total_amount > 1:
                    sell_price = row["close"]
                    sell_date = pd.to_datetime(idx).date()
                    sell_amount = total_amount * 1.0
                    balance = balance + sell_amount * sell_price * (1+fee)
                    total_amount = total_amount - sell_amount
                    if verbose:
                        print(f"Sell {sell_amount * sell_price:.4f} at {sell_price:.4f} on {sell_date}")
                        print(f"Balance: {balance:.4f} Total amount: {total_amount:.4f}")
                        print("-"*50)
            universe.loc[idx, "value"] = balance + total_amount * row["close"]
            universe.loc[idx, "balance"] = balance
            universe.loc[idx, "total_amount"] = total_amount
        
        statements = universe.copy()
        
        return statements

    def plot(self, whole_screen=False):

        def align_xticks_for_df(df, baseline):
            return df.reindex(baseline.index).bfill()

        entry_date = self.universe[self.universe['signal']==1].index
        exit_date = self.universe[self.universe['signal']==0].index

        fig = plt.figure(figsize=(15, 8), constrained_layout=True)
        gs = gridspec.GridSpec(7, 4, figure=fig)
        ax1 = plt.subplot(gs[0:2, :])
        ax2 = plt.subplot(gs[2:4, :])
        ax3 = plt.subplot(gs[4, :2])
        ax4 = plt.subplot(gs[5, :2])
        ax5 = plt.subplot(gs[6, :2])
        ax6 = plt.subplot(gs[4, 2:])
        ax7 = plt.subplot(gs[5, 2:])
        ax8 = plt.subplot(gs[6, 2:])

        ax1.plot(self.universe.close, label="Close Price")
        ax1.plot(self.universe['dc_high'], '#B97A95', label="DC-HIGH", alpha=1, linestyle="--")
        ax1.plot(self.universe['dc_low'], '#F6AE99', label="DC-LOW", alpha=1, linestyle="--")
        ax1.scatter(entry_date, self.universe.close[entry_date], label="buy", marker="^", alpha=.7, color="tab:green")
        ax1.scatter(exit_date, self.universe.close[exit_date], label="sell", marker="v", alpha=.7, color="tab:red")
        ax1.set_title("Entry and exit signals")
        ax1.legend(loc="upper left")
        ax1.grid()

        plot_drawdown_periods(self.universe['value'].pct_change(), top=5, ax=ax2)
        ax2.grid()

        plot_drawdown_underwater(self.universe['value'].pct_change(), ax=ax3)
        ax3.grid()

        ax4.plot(align_xticks_for_df(self.universe['balance'], self.universe))
        ax4.set_title("Balance")
        ax4.grid()

        ax5.plot(align_xticks_for_df(self.universe['total_amount'], self.universe))
        ax5.set_title("Total amount of holding the asset")
        ax5.grid()

        plot_annual_returns(self.universe['value'].pct_change(), ax=ax6)
        ax6.grid(axis="x")

        plot_monthly_returns_heatmap(self.universe['value'].pct_change(), ax=ax7)

        plot_monthly_returns_dist(self.universe['value'].pct_change(), ax=ax8)
        ax8.grid(axis="y")

        if whole_screen:
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
        plt.show()


class DonchianChannelStrategy(BaseStrategy):

    def __init__(self, universe, window_up=20, window_down=10):
        super(DonchianChannelStrategy, self).__init__()
        self.universe = universe
        self.statements = None
        self.initialise(window_up, window_down)

    def initialise(self, window_up, window_down):
        self.universe = column_name_lower(self.universe)
        # Compute returns
        self.universe['ret'] = self.universe["close"] / self.universe["close"].shift(1) - 1
        self.universe['ret'][0] = 0

        # Donchian Channel
        self.universe['dc_high'] = talib.MAX(self.universe["high"], timeperiod=window_up).shift(1)
        self.universe['dc_low'] = talib.MIN(self.universe["low"], timeperiod=window_down).shift(1)
        
    def run(self, capital=10000, fee=0.001, position=0, quota=0.5, stop_loss=0.1, take_profit=0.5, verbose=False):
        entry, exit = self.generate_signals(self.universe)
        self.statements = self.trade(
            self.universe, 
            entry, exit, 
            capital, 
            fee, 
            stop_loss, 
            take_profit, 
            position, 
            quota, 
            verbose
        )

    def generate_signals(self, universe):
        # Compute signal based on Donchian Channel
        entry = crossover(universe["close"], universe['dc_high']).astype(int)
        exit = crossunder(universe["close"], universe['dc_low']).astype(int)
        entry[entry==0], exit[exit==0] = np.nan, np.nan
        entry, exit = entry[~np.isnan(entry)], exit[~np.isnan(exit)]
        return entry, exit

    def trade(
            self, 
            universe, 
            entry, 
            exit, 
            capital=10000, 
            fee=0.001, 
            stop_loss=0.1, 
            take_profit=0.5, 
            total_amount=0, 
            quota=0.5, 
            verbose=True
        ):
        universe.loc[(entry==1).index, 'signal'] = 1
        universe.loc[(exit==1).index, 'signal'] = 0
        balance = capital
        buy_price, sell_price = None, None
        for idx, row in universe.iterrows():
            # Stop loss
            if (buy_price is not None) and (stop_loss is not None) and (row["close"] < buy_price*(1.0-stop_loss)):
                stop_loss_price = row["close"]
                stop_loss_date = pd.to_datetime(idx).date()
                balance += stop_loss_price * math.floor(total_amount * 0.5) * (1-fee)
                total_amount -= math.floor(total_amount * 0.5)
                if verbose:
                    print(f"Stop loss at price {stop_loss_price:.4f} on {stop_loss_date}")
                    print(f"Last buy at {buy_price:.4f}")
                    print(f"Balance: {balance:.4f} Total amount: {total_amount:.4f}")
                    print("-"*50)
            # Take profit
            elif (buy_price is not None) and (take_profit is not None) and (row["close"] >= buy_price*(1.0+take_profit)):
                take_profit_price = row["close"]
                take_profit_date = pd.to_datetime(idx).date()
                balance += take_profit_price * math.floor(total_amount * 0.5) * (1-fee)
                total_amount -= math.floor(total_amount * 0.5)
                if verbose:
                    print(f"Take profit at price {take_profit_price:.4f} on {take_profit_date}")
                    print(f"Last buy at {take_profit_date:.4f}")
                    print(f"Balance: {balance:.4f} Total amount: {total_amount:.4f}")
                    print("-"*50)

            if row["signal"] == 1:
                if balance > 0:
                    buy_price = row["close"]
                    buy_date = pd.to_datetime(idx).date()
                    buy_amount = balance * quota / buy_price
                    balance -= buy_amount * buy_price * (1+fee)
                    total_amount += buy_amount
                    if verbose:
                        print(f"Buy {buy_amount * buy_price:.4f} at {buy_price:.4f} on {buy_date}")
                        print(f"Balance: {balance:.4f} Total amount: {total_amount:.4f}")
                        print("-"*50)
            elif row["signal"] == 0:
                if total_amount > 1:
                    sell_price = row["close"]
                    sell_date = pd.to_datetime(idx).date()
                    sell_amount = total_amount * 0.5
                    balance += sell_amount * sell_price * (1-fee)
                    total_amount -= sell_amount
                    if verbose:
                        print(f"Sell {sell_amount * sell_price:.4f} at {sell_price:.4f} on {sell_date}")
                        print(f"Balance: {balance:.4f} Total amount: {total_amount:.4f}")
                        print("-"*50)
            universe.loc[idx, "value"] = balance + total_amount * row["close"]
            universe.loc[idx, "balance"] = balance
            universe.loc[idx, "total_amount"] = total_amount
        
        statements = universe.copy()

        return statements

    def plot(self, whole_screen=False):

        def align_xticks_for_df(df, baseline):
            return df.reindex(baseline.index).bfill()

        entry_date = self.universe[self.universe['signal']==1].index
        exit_date = self.universe[self.universe['signal']==0].index

        fig = plt.figure(figsize=(15, 8), constrained_layout=True)
        gs = gridspec.GridSpec(7, 4, figure=fig)
        ax1 = plt.subplot(gs[0:2, :])
        ax2 = plt.subplot(gs[2:4, :])
        ax3 = plt.subplot(gs[4, :2])
        ax4 = plt.subplot(gs[5, :2])
        ax5 = plt.subplot(gs[6, :2])
        ax6 = plt.subplot(gs[4, 2:])
        ax7 = plt.subplot(gs[5, 2:])
        ax8 = plt.subplot(gs[6, 2:])

        ax1.plot(self.universe.close, label="Close Price")
        ax1.plot(self.universe['dc_high'], '#B97A95', label="DC-HIGH", alpha=1, linestyle="--")
        ax1.plot(self.universe['dc_low'], '#F6AE99', label="DC-LOW", alpha=1, linestyle="--")
        ax1.scatter(entry_date, self.universe.close[entry_date], label="buy", marker="^", alpha=.7, color="tab:green")
        ax1.scatter(exit_date, self.universe.close[exit_date], label="sell", marker="v", alpha=.7, color="tab:red")
        ax1.set_title("Entry and exit signals")
        ax1.legend(loc="upper left")
        ax1.grid()

        plot_drawdown_periods(self.universe['value'].pct_change(), top=5, ax=ax2)
        ax2.grid()

        plot_drawdown_underwater(self.universe['value'].pct_change(), ax=ax3)
        ax3.grid()

        ax4.plot(align_xticks_for_df(self.universe['balance'], self.universe))
        ax4.set_title("Balance")
        ax4.grid()

        ax5.plot(align_xticks_for_df(self.universe['total_amount'], self.universe))
        ax5.set_title("Total amount of holding the asset")
        ax5.grid()

        plot_annual_returns(self.universe['value'].pct_change(), ax=ax6)
        ax6.grid(axis="x")

        plot_monthly_returns_heatmap(self.universe['value'].pct_change(), ax=ax7)

        plot_monthly_returns_dist(self.universe['value'].pct_change(), ax=ax8)
        ax8.grid(axis="y")

        if whole_screen:
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
        plt.show()

        plt.show()
