from .base_strategy import BaseStrategy


class RLStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    def load_strategy():
        """
        In case the strategy is learned with RL the policy (mapping of states to actions) it needs
        to be loaded from current model

        Return: ?
        """

        raise NotImplementedError

    def calculate_bids(self, operational_window):

        """
        load current RL model to map state to action

        Return: volume, price
        """

        # load model for market
        # sample action based on state

        raise NotImplementedError
