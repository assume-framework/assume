from assume.strategies.base_strategy import BaseStrategy
from assume.units.storage_unit import StorageUnit

class NaiveStorageStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    def calculate_bids(
        self,
        unit: StorageUnit = None,
        operational_window: dict = None,
    ):
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price

        """
        #bids depent of current SOC 
        if unit.current_SOC < (unit.max_SOC + unit.min_SOC) / 2:
            #place bid to discharge
            price = operational_window["max_power_discharge"]["marginal_cost"]
            volume = operational_window["max_power_discharge"]["power_discharge"]
        else: 
            #place bid to charge
            price = operational_window["max_power_charge"]["marginal_cost"]
            volume = operational_window["max_power_charge"]["power_charge"]

        bids = [{"price": price, "volume": volume}]
        return bids