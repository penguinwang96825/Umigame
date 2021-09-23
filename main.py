import umigame
import matplotlib.pyplot as plt
from umigame.benchmarks.turtle import TurtleStrategy
from umigame.benchmarks.base import BuyAndHoldStrategy


def main():
    crypto = "BTC-USD"
    universe = umigame.datasets.fetch_crypto(tickers=[crypto])[crypto]
    # stock = "AAPL"
    # universe = umigame.datasets.fetch_usstock(tickers=[stock])[stock]
    universe = universe.dropna().loc["2021"]
    turtle = TurtleStrategy(universe, window_up=20, window_down=10)
    turtle.run(capital=100000, fee=0.001, quota=0.5, verbose=False)
    turtle_portfolio_value = turtle.trades["value"]
    print(turtle.score("calmars_ratio"))
    # turtle.plot()
    benchmark = BuyAndHoldStrategy(universe)
    benchmark.run(capital=100000, fee=0.001, quota=1.0, verbose=False)
    benchmark_portfolio_value = benchmark.trades["value"]
    print(benchmark.score("calmars_ratio"))
    # benchmark.plot()

    plt.figure(figsize=(15, 6))
    plt.plot(turtle_portfolio_value, color="tab:blue", label="Turtle")
    plt.plot(benchmark_portfolio_value, color="tab:orange", label="Buy&Hold")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()