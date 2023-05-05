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
        unit=None,
        operational_window: dict = None,
    ):
        raise NotImplementedError()
