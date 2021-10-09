import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .breakthrough import BreakthroughStrategy
from .base import BuyAndHoldStrategy
from ..utils import crossover, crossunder


class CCIStrategy(BreakthroughStrategy):

    def __init__(self, 
            ticker, 
            capital=10000, 
            fees=0.001, 
            cci_window=14, 
            start="2018-08-25", 
            end="2021-08-25", 
            show_progress=True
        ):
        self.ticker = ticker
        self.capital = capital
        self.fees = fees
        self.cci_window = cci_window
        self.start = start
        self.end = end
        self.show_progress = show_progress
        self.initialise_local()

    def generate_signals(self, universe):
        universe["cci"] = talib.CCI(universe["high"], universe["low"], universe["close"], timeperiod=self.cci_window)
        entry = crossover(universe["cci"], 100).astype(int)
        exit = crossunder(universe["cci"], -100).astype(int)
        entry[entry==0], exit[exit==0] = np.nan, np.nan
        entry, exit = entry[~np.isnan(entry)], exit[~np.isnan(exit)]
        return entry, exit

    def plot(self, whole_screen=False):

        def align_xticks_for_df(df, baseline):
            return df.reindex(baseline.index).bfill()
            
        entry_date = self.statements[self.statements['trade']=="buy"].index
        exit_date = self.statements[self.statements['trade']=="sell"].index

        fig = plt.figure(figsize=(15, 8), constrained_layout=True)
        gs = gridspec.GridSpec(6, 4, figure=fig)
        ax1 = plt.subplot(gs[0:2, :])
        ax2 = plt.subplot(gs[2:4, :])
        ax3 = plt.subplot(gs[4, :])
        ax4 = plt.subplot(gs[5, :])

        ax1.plot(self.universe.close, label="Close Price")
        ax1.plot(self.universe['cci'], label="fast", alpha=.5, linestyle="--")
        ax1.scatter(entry_date, self.universe.close[entry_date], label="buy", marker="^", alpha=.7, color="tab:green")
        ax1.scatter(exit_date, self.universe.close[exit_date], label="sell", marker="v", alpha=.7, color="tab:red")
        ax1.set_title("Entry and exit signals")
        ax1.legend(loc="upper left")
        ax1.grid()

        benchmark = BuyAndHoldStrategy(
            self.ticker, self.capital, self.fees, self.start, self.end, show_progress=False
        )
        benchmark.run()
        benchmark_returns = benchmark.universe["equity"]
        strategy_returns = self.universe["equity"]
        ax2.plot(benchmark_returns, label="benchmark")
        ax2.plot(strategy_returns, label="strategy")
        ax2.set_title("Cumulative returns")
        ax2.legend()
        ax2.grid()

        ax3.plot(align_xticks_for_df(self.universe['balance'], self.universe))
        ax3.set_title("Balance")
        ax3.grid()

        ax4.plot(align_xticks_for_df(self.universe['position'], self.universe))
        ax4.set_title("Position")
        ax4.grid()

        if whole_screen:
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())

        plt.show()