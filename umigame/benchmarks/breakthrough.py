import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import empyrical as ep
from copy import deepcopy
from abc import abstractmethod
from .base import BaseStrategy, BuyAndHoldStrategy
from ..utils import column_name_lower
from ..plotting import *


import os
import sys
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)
from umigame.datasets.crypto import fetch_crypto


class BreakthroughStrategy(BaseStrategy):

    def __init__(
            self, 
            ticker, 
            capital=10000, 
            fees=0.001, 
            fast_period=10, 
            slow_period=20, 
            start="2018-08-25", 
            end="2021-08-25"
        ):
        self.ticker = ticker
        self.capital = capital
        self.fees = fees
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.start = start
        self.end = end

    def initialise_local(self):
        self.balance = self.capital
        self.equity = self.capital
        self.position = 0
        self.position_history = []
        self.statements = None
        self.universe = None
        self.pnl = None

    @abstractmethod
    def generate_signals(self, universe):
        raise NotImplementedError

    def run(self):
        universe = self._get_price_data()
        buy_next_bar, close_next_bar = False, False

        for idx, row in universe.iterrows():
            if buy_next_bar:
                buy_price = row["open"]
                buy_date = pd.to_datetime(idx).date()
                size = self.balance * (0.5) / buy_price
                self.buy(buy_date, size, buy_price)
                buy_next_bar = False
                if self.show_progress:
                    print(f"Open position @ {buy_price:.4f} on {buy_date}")
            if close_next_bar:
                close_price = row["open"]
                close_date = pd.to_datetime(idx).date()
                size = self.position
                self.close(close_date, size, close_price, "sell")
                close_next_bar = False
                if self.show_progress:
                    print(f"Close position @ {close_price:.4f} on {close_date}")
            # Open the position
            if row["signal"] == 1:
                if self.position >= 0:
                    buy_next_bar = True
            # Close the position
            elif row["signal"] == -1:
                if self.position > 0:
                    close_next_bar = True

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

        self.pnl = self._calculate_pnl()
        self.stats = self._calculate_stats()

    def _prepare_price_data(self):
        universe = fetch_crypto([self.ticker], self.start, self.end)[self.ticker]
        universe = column_name_lower(universe)
        entry_, exit_ = self.generate_signals(universe)
        universe.loc[(entry_==1).index, 'signal'] = 1
        universe.loc[(exit_==1).index, 'signal'] = -1
        return universe

    def _get_price_data(self):
        universe = self._prepare_price_data()
        return universe

    def _calculate_pnl(self):
        success_history, failure_history = [], []
        cost = 0
        win_rate = 0
        for idx, row in self.statements.iterrows():
            if row["trade"] == "buy":
                cost += row["amount"]
            if row["trade"] != "buy":
                if row["amount"] > cost:
                    success_history.append([idx, row["amount"]-cost])
                else:
                    failure_history.append([idx, row["amount"]-cost])
                cost = 0
        if len(success_history) + len(failure_history) > 0:
            win_rate = len(success_history) / (len(failure_history) + len(success_history))
        print(f"Win rate: {win_rate:.4f}")
        pnl = pd.DataFrame(success_history + failure_history, columns=["date", "pnl"])
        return pnl.set_index("date").sort_index()

    def _calculate_stats(self):
        benchmark = BuyAndHoldStrategy(
            self.ticker, self.capital, self.fees, self.start, self.end, show_progress=False
        )
        benchmark.run()
        benchmark_returns = benchmark.universe["equity"].pct_change()
        strategy_returns = self.universe["equity"].pct_change()
        sharpe_ratio = [
            ep.sharpe_ratio(strategy_returns, annualization=252), 
            ep.sharpe_ratio(benchmark_returns, annualization=252)
        ]
        calmar_ratio = [
            ep.calmar_ratio(strategy_returns, annualization=252), 
            ep.calmar_ratio(benchmark_returns, annualization=252)
        ]
        omega_ratio = [
            ep.omega_ratio(strategy_returns), 
            ep.omega_ratio(benchmark_returns)
        ]
        sortino_ratio = [
            ep.sortino_ratio(strategy_returns, annualization=252), 
            ep.sortino_ratio(benchmark_returns, annualization=252)
        ]
        stability = [
            ep.stability_of_timeseries(strategy_returns), 
            ep.stability_of_timeseries(benchmark_returns)
        ]
        annual_return = [
            ep.annual_return(strategy_returns, annualization=252), 
            ep.annual_return(benchmark_returns, annualization=252)
        ]
        annual_volatility = [
            ep.annual_volatility(strategy_returns, annualization=252), 
            ep.annual_volatility(benchmark_returns, annualization=252)
        ]
        max_drawdown = [
            ep.max_drawdown(strategy_returns), 
            ep.max_drawdown(benchmark_returns)
        ]
        return pd.DataFrame([
            annual_return, annual_volatility, max_drawdown, 
            sharpe_ratio, calmar_ratio, omega_ratio, sortino_ratio, stability
        ], index=[
            "Annual Return", "Annual Volatility", "Max Drawdown", 
            "Sharpe Ratio", "Calmar Ratio", "Omega Ratio", "Sortino Ratio", "Stability"
        ], columns=["strategy", "benchmark"])

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
        ax1.plot(self.universe['fast'], '#B97A95', label="fast", alpha=.5, linestyle="--")
        ax1.plot(self.universe['slow'], '#F6AE99', label="slow", alpha=.5, linestyle="--")
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

    def plot_risk(self):
        fig = plt.figure(figsize=(15, 8), constrained_layout=True)
        gs = gridspec.GridSpec(3, 3, figure=fig)
        ax1 = plt.subplot(gs[0:2, :])
        ax2 = plt.subplot(gs[2, 0])
        ax3 = plt.subplot(gs[2, 1])
        ax4 = plt.subplot(gs[2, 2])

        plot_drawdown_underwater(self.universe['equity'].pct_change(), ax=ax1)
        ax1.grid()

        plot_annual_returns(self.universe['equity'].pct_change(), ax=ax2)
        ax2.grid(axis="x")

        plot_monthly_returns_heatmap(self.universe['equity'].pct_change(), ax=ax3)

        plot_monthly_returns_dist(self.universe['equity'].pct_change(), ax=ax4)
        ax4.grid(axis="y")

        plt.show()