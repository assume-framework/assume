class BaseStrategy:
    """A base class for a bidding strategy.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self):
        super().__init__()

    def calculate_bids(self, product, operational_window):
        raise NotImplementedError()
