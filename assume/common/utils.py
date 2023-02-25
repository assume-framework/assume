import inspect
from datetime import datetime
from functools import wraps
from math import isclose, log10

from dateutil import rrule

from .market_mechanisms import available_clearing_strategies
from .marketclasses import MarketOrderbook, MarketProduct, Orderbook


def initializer(func):
    """
    Automatically assigns the parameters.
    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):

        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


def is_mod_close(a, mod_b):
    """
    due to floating point, a mod b can be very close to 0 or very close to mod_b
    """
    abs_tol = 1e-10
    # abs_tol needed for comparison near zero
    return isclose(a % mod_b, 0, abs_tol=abs_tol) or isclose(
        a % mod_b, mod_b, abs_tol=abs_tol
    )


def round_digits(n, tick_size):
    """
    rounds n to the power of 10 of the needed tick_size
    Example_
    >>> round_digits(1.100001, 0.1)
    1.1
    >>> round_digits(400.1, 20)
    400
    """
    return round(n, 1-int(log10(tick_size)))


def get_available_products(market_products: list[MarketProduct], startdate: datetime):
    options = []
    for product in market_products:
        start = startdate + product.first_delivery_after_start
        if isinstance(product.duration, rrule.rrule):
            starts = list(product.duration.xafter(start, product.count + 1))
            for i in range(product.count):
                period_start = starts[i]
                period_end = starts[i + 1]
                options.append((period_start, period_end, product.only_hours))
        else:
            for i in range(product.count):
                period_start = start + product.duration * i
                period_end = start + product.duration * (i + 1)
                options.append((period_start, period_end, product.only_hours))
    return options
