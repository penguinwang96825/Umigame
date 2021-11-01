import numpy as np
import pandas as pd
from collections import defaultdict


class Trade(object):

    def __init__(self, initial_cash=100.0, positions=0, direction='long'):
        super(Trade, self).__init__()
        self.cash = initial_cash
        self.equity = initial_cash
        self.positions = positions
        self.direction = direction
        self._statement = defaultdict(list)
        self._entry_record = defaultdict(list)
        self._exit_record = defaultdict(list)

    def add_entry_order(self, order):
        self._entry_record['size'].append(order.size)
        self._entry_record['entry_date'].append(order.date)
        self._entry_record['entry_price'].append(order.price)
        self._entry_record['entry_fees'].append(order.fees)

    def add_exit_order(self, order):
        self._exit_record['size'].append(order.size)
        self._exit_record['exit_date'].append(order.date)
        self._exit_record['exit_price'].append(order.price)
        self._exit_record['exit_fees'].append(order.fees)

    def update_statement(self, date, price, entry, exit, sl, tp, size=0, fees=None):
        if size != 0:
            if size > 0:
                buy_amount = price * np.abs(size) + fees
                if self.cash > buy_amount:
                    self.cash -= buy_amount
                    self.positions += size
            elif (size < 0) and (self.positions > 0):
                sell_amount = price * np.abs(size) - fees
                self.cash += sell_amount
                self.positions += size
        self.equity = self.positions * price + self.cash

        self._statement['date'].append(date)
        self._statement['price'].append(price)
        self._statement['entry'].append(entry)
        self._statement['exit'].append(exit)
        self._statement['sl'].append(sl)
        self._statement['tp'].append(tp)
        self._statement['size'].append(size)
        self._statement['positions'].append(self.positions)
        self._statement['fees'].append(fees)
        self._statement['cash'].append(self.cash)
        self._statement['equity'].append(self.equity)

    @property
    def entry_record(self):
        return pd.DataFrame(self._entry_record)

    @property
    def exit_record(self):
        return pd.DataFrame(self._exit_record)

    @property
    def statement(self):
        return pd.DataFrame(self._statement)


class Order(object):

    def __init__(self, size, date, price, fees):
        super(Order, self).__init__()
        self.size = size
        self.date = date
        self.price = price
        self.fees = fees