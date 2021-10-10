import umigame
import matplotlib.pyplot as plt
from umigame.benchmarks.turtle import TurtleSystem
from umigame.benchmarks.base import BuyAndHoldStrategy
from umigame.benchmarks.fourier import FourierStrategy
from umigame.benchmarks.machine import MachineLearningStrategy
from umigame.benchmarks.cci import CCIStrategy
from copy import deepcopy


def main():
    test()


def test():
    ticker = "BTC-USD"
    benchmark = BuyAndHoldStrategy(ticker, capital=1000, start="2018-01-01", show_progress=False)
    benchmark.run()
    turtle = TurtleSystem(ticker, capital=1000, start="2018-01-01", show_progress=False)
    turtle.run()
    fourier = FourierStrategy(ticker, capital=1000, start="2018-01-01", show_progress=False)
    fourier.run()

    plt.figure(figsize=(15, 6))
    plt.plot(benchmark.universe["equity"], label="benchmark")
    plt.plot(turtle.universe["equity"], label="turtle")
    plt.plot(fourier.universe["equity"], label="fourier")
    plt.legend()
    plt.grid()
    plt.show()


def run_cci():
    cci_strategy = CCIStrategy(ticker="BTC-USD", start="2018-01-01", end="2021-09-28")
    cci_strategy.run()
    cci_strategy.plot()
    cci_strategy.plot_risk()
    print(cci_strategy.stats)


def run_machine():
    machine = MachineLearningStrategy(ticker="BTC-USD", start="2018-01-01", end="2021-09-28")
    machine.run()
    machine.plot()
    machine.plot_risk()
    print(machine.stats)


def run_turtle():
    turtle = TurtleSystem(ticker="BTC-USD", capital=1000, start="2018-01-01", end="2021-09-28", max_risk=0.01)
    turtle.run()
    turtle.plot()
    turtle.plot_risk()
    print(turtle.stats)


def run_fourier():
    fourier = FourierStrategy(ticker="BTC-USD", start="2018-01-01", end="2021-09-28")
    fourier.run()
    fourier.plot()
    fourier.plot_risk()
    print(fourier.stats)


if __name__ == "__main__":
    main()
