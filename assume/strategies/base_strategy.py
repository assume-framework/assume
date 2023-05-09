class BaseStrategy:
    """A base class for a bidding strategy.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self):
        super().__init__()

    def calculate_bids(
        self,
        operational_window: dict,
        product_tuple: tuple,
    ):
        raise NotImplementedError()
