class BaseStrategy:
    """A base class for a bidding strategy.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, *args, **kwargs):
        pass

    def calculate_bids(
        self,
        unit=None,
        market_config=None,
        operational_window: dict = None,
    ):
        raise NotImplementedError()
