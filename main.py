import umigame
import matplotlib.pyplot as plt
from umigame.benchmarks.turtle import TurtleStrategy, DonchianChannelStrategy
from umigame.benchmarks.base import BuyAndHoldStrategy
from umigame.benchmarks.fourier import fourier_denoise


def main():
    run()


def run_fourier():
    crypto = "BTC-USD"
    universe = umigame.datasets.fetch_crypto(tickers=[crypto])[crypto]
    universe = universe.dropna().loc["2020":]
    noise = universe["close"]
    clean = fourier_denoise(universe["close"], window=20)

    plt.figure(figsize=(15, 6))
    plt.plot(noise, label="close price")
    plt.plot(clean, label="close price (FFT)")
    plt.legend()
    plt.grid()
    plt.show()


def run():
    crypto = "BTC-USD"
    universe = umigame.datasets.fetch_crypto(tickers=[crypto])[crypto]
    # stock = "AAPL"
    # universe = umigame.datasets.fetch_usstock(tickers=[stock])[stock]
    universe = universe.dropna().loc["2017":]
    portfolio = {}

    turtle = TurtleStrategy(universe, window_up=20, window_down=10, window_atr=20)
    turtle.run(capital=100000, fee=0.001, stop_loss=0.1, take_profit=None, verbose=False)
    portfolio["turtle"] = turtle.statements["value"]
    # print(turtle.score("max_drawdown"))
    print(portfolio["turtle"])
    # turtle.plot()

    donchian = DonchianChannelStrategy(universe, window_up=20, window_down=10)
    donchian.run(capital=1000, fee=0.001, quota=0.5, stop_loss=0.1, take_profit=None, verbose=False)
    portfolio["donchian"] = donchian.statements["value"]
    # print(donchian.score("max_drawdown"))
    print(portfolio["donchian"])
    # donchian.plot()

    benchmark = BuyAndHoldStrategy(universe)
    benchmark.run(capital=100000, fee=0.001, verbose=False)
    portfolio["benchmark"] = benchmark.statements["value"]
    # print(benchmark.score("max_drawdown"))
    print(portfolio["benchmark"])
    # benchmark.plot()

    # print("A")
    # print(portfolio["turtle"])
    # print("-"*50)
    # print("B")
    # print(portfolio["donchian"])
    # print("-"*50)
    # print("C")
    # print(portfolio["benchmark"])
    # print("-"*50)
    print(portfolio)


def plot():
    plt.figure(figsize=(15, 6))
    for idx, year in enumerate(["2020", "2019", "2018", "2017"]):
        crypto = "BTC-USD"
        universe = umigame.datasets.fetch_crypto(tickers=[crypto])[crypto]
        # stock = "AAPL"
        # universe = umigame.datasets.fetch_usstock(tickers=[stock])[stock]
        universe = universe.dropna().loc[year]
        
        plt.subplot(2, 2, idx+1)
        plt.title(f"Year {year}")
        turtle = TurtleStrategy(universe, window_up=20, window_down=10, window_atr=20)
        turtle.run(capital=100000, fee=0.001, stop_loss=0.1, take_profit=None, verbose=False)
        turtle_portfolio_value = turtle.statements["value"]
        plt.plot(turtle_portfolio_value, color="tab:blue", label="Turtle")
        print(turtle.score("max_drawdown"))
        # turtle.plot()
        donchian = DonchianChannelStrategy(universe, window_up=20, window_down=10)
        donchian.run(capital=100000, fee=0.001, quota=0.5, stop_loss=0.1, take_profit=None, verbose=False)
        donchian_portfolio_value = donchian.statements["value"]
        plt.plot(donchian_portfolio_value, color="tab:green", label="Donchian")
        print(donchian.score("max_drawdown"))
        # donchian.plot()
        benchmark = BuyAndHoldStrategy(universe)
        benchmark.run(capital=100000, fee=0.001, verbose=False)
        benchmark_portfolio_value = benchmark.statements["value"]
        plt.plot(benchmark_portfolio_value, color="tab:orange", label="Buy&Hold")
        print(benchmark.score("max_drawdown"))
        # benchmark.plot()

        plt.legend(loc="upper left")
        plt.grid()
    
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()