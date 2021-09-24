import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from abc import ABC, abstractmethod
from ..utils import column_name_lower
from ..metrics import *
from ..plotting import *


class BaseStrategy(ABC):

    def __init__(self):
        super(BaseStrategy, self).__init__()

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def trade(self):
        raise NotImplementedError

    def calculate_rets(self, universe):
        if not hasattr(self, "statements"):
            raise NotRunError("Strategy has not been run...")
        rets = universe["value"].pct_change(1)
        return rets

    def score(self, scoring="sharpe_ratio"):
        if not hasattr(self, "statements"):
            raise NotRunError("Strategy has not been run...")
        rets = self.calculate_rets(self.universe)
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

    def __init__(self, universe):
        super(BuyAndHoldStrategy, self).__init__()
        self.universe = universe
        self.statements = None
        self.universe = column_name_lower(self.universe)

    def run(self, capital=10000, fee=0.0, position=0, quota=1.0, verbose=False):
        self.statements = self.trade(self.universe, capital, fee, position, quota, verbose)

    def trade(self, universe, capital=10000, fee=0.001, total_amount=0, quota=0.5, verbose=True):
        universe["signal"] = np.nan
        universe.loc[universe.index[0], 'signal'] = 1
        universe.loc[universe.index[-1], 'signal'] = 0
        balance = capital

        for idx, row in universe.iterrows():
            if row["signal"] == 1:
                if balance > 0:
                    buy_price = row["close"]
                    buy_date = pd.to_datetime(idx).date()
                    buy_amount = math.floor(balance * quota / buy_price)
                    balance = balance - buy_amount * buy_price
                    total_amount = total_amount + buy_amount
                    if verbose:
                        print(f"Buy {buy_amount * buy_price:.4f} at {buy_price:.4f} on {buy_date}")
                        print(f"Balance: {balance:.4f} Total amount: {total_amount:.4f}")
                        print("-"*50)
            elif row["signal"] == 0:
                if total_amount > 1:
                    sell_price = row["close"]
                    sell_date = pd.to_datetime(idx).date()
                    sell_amount = total_amount
                    balance = balance + sell_amount * sell_price
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


def metric_from_name(name):
    """
    Return metrics from name.
    """
    metrics = (
        sharpe_ratio, 
        sortino_ratio, 
        calmars_ratio, 
        max_drawdown, 
        annualise_returns, 
        annualise_volatility
    )
    return {m.__name__: m for m in metrics}[name]


class NotRunError(ValueError):
    pass