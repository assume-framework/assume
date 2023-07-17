import numpy as np
import pandas as pd

from assume.common.market_objects import MarketConfig
from assume.strategies.base_strategy import BaseStrategy, OperationalWindow
from assume.units.storage import Storage


class flexableEOMStorage(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = pd.Timedelta(kwargs.get("eom_foresight", "12h"))

    def calculate_bids(
        self,
        unit: Storage,
        operational_window: OperationalWindow,
        market_config: MarketConfig,
    ):
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price
        Strategy analogue to flexABLE

        """
        # =============================================================================
        # Storage Unit is either charging, discharging, or off
        # =============================================================================

        start = operational_window["window"][0]
        end = operational_window["window"][1]
   
        average_price = calculate_price_average(
            unit=unit, 
            current_time=start,
            foresight=self.foresight
        )

        bid_quantity = 0

        if (
            unit.price_forecast[start] >= average_price / unit.efficiency_discharge
        ) and (operational_window["ops"]["max_power_discharge"]["volume"] > 0):
            # place bid to discharge
            bid_quantity = operational_window["ops"]["max_power_discharge"]["volume"]

        elif (
            unit.price_forecast[start] <= average_price * unit.efficiency_charge
        ) and (operational_window["ops"]["max_power_charge"]["volume"] < 0):
            # place bid to charge
            bid_quantity = operational_window["ops"]["max_power_charge"]["volume"]

        if bid_quantity != 0:
            return [{"price": average_price, "volume": bid_quantity}]
        else:
            return []




class flexablePosCRMStorage(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains crm_foresight argument
        self.foresight = pd.Timedelta(kwargs.get("crm_foresight", "4h"))

        self.current_time = None

    def calculate_bids(
        self,
        unit: Storage,
        operational_window: OperationalWindow,
        market_config: MarketConfig,
    ):
        self.current_time = operational_window["window"][0]
        
        bid_quantity = operational_window["ops"]["pos_reserve"]["volume"]
        if bid_quantity == 0:
            return []
        
        marginal_cost = operational_window["ops"]["pos_reserve"]["cost"]

        specific_revenue = get_specific_revenue(
            unit=unit,
            marginal_cost=marginal_cost,
            current_time=self.current_time,
            foresight=self.foresight,
        )
        if specific_revenue >= 0:
            capacity_price = specific_revenue
        else:
            capacity_price = abs(specific_revenue) * unit.min_power_discharge / bid_quantity
        
        energy_price = capacity_price / unit.current_SOC

        if market_config.product_type == "capacity_pos":
            bids = [
                {"price": capacity_price, "volume": bid_quantity},
            ]
        elif market_config.product_type == "energy_pos":
            bids = [
                {"price": energy_price, "volume": bid_quantity},
            ]
        else:
            raise ValueError(
                f"Product {market_config.product_type} is not supported by this strategy."
            )

        return bids
    


class flexableNegCRMStorage(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.foresight = pd.Timedelta(kwargs.get("crm_foresight", "4h"))

        self.current_time = None

    def calculate_bids(
        self,
        unit: Storage,
        operational_window: OperationalWindow,
        market_config: MarketConfig,
    ):
        #in flexable no prices calculated for CRM_neg
        bid_quantity = operational_window["ops"]["neg_reserve"]["volume"]
        if bid_quantity == 0:
            return []
        #if bid_quantity >= min_bid_volume  

        if market_config.product_type == "capacity_neg":
            bids = [
                {"price": 0, "volume": bid_quantity},
            ]
        elif market_config.product_type == "energy_neg":
            bids = [
                {"price": 0, "volume": bid_quantity},
            ]
        else:
            raise ValueError(
                f"Product {market_config.product_type} is not supported by this strategy."
            )

        return bids


def calculate_price_average(unit, current_time, foresight):
    average_price = np.mean(
        unit.price_forecast[
            current_time - foresight : current_time + foresight
        ]
    )

    return average_price

def get_specific_revenue(
    unit,
    marginal_cost,
    current_time,
    foresight,
):
    t = current_time
    price_forecast = []

    if t + foresight > unit.price_forecast.index[-1]:
        price_forecast = unit.price_forecast.loc[t:]
    else:
        price_forecast = unit.price_forecast.loc[t : t + foresight]

    possible_revenue = 0
    theoretic_SOC = unit.current_SOC
    for market_price in price_forecast:
        theoretic_power_discharge = min(max(theoretic_SOC-unit.min_SOC, 0), 
                                                                 unit.max_power_discharge)
        possible_revenue += (market_price - marginal_cost) * theoretic_power_discharge
        theoretic_SOC -= theoretic_power_discharge

    if unit.current_SOC - theoretic_SOC != 0:
        possible_revenue = possible_revenue / (unit.current_SOC - theoretic_SOC)
    

    return possible_revenue

    """
    def calculatingBidPricesSTO_CRM(self, t):
        fl = int(4 / self.world.dt)
        SOC_theoretical = self.soc[t]
        revenue_theoretical = []

        for tick in range(t, t + fl):
            BidSTO_EOM = self.calculate_bids_eom(tick       SOC_theoretical)

            if len(BidSTO_EOM) == 0:
                continue

            BidSTO_EOM = BidSTO_EOM[0]
            if BidSTO_EOM.bidType == 'Demand':
                SOC_theoretical += BidSTO_EOM.amount * self.efficiency_ch * self.world.dt
                revenue_theoretical.append(- self.world.pfc[t] * BidSTO_EOM.amount * self.world.dt)

            elif BidSTO_EOM.bidType == 'Supply':
                SOC_theoretical -= BidSTO_EOM.amount / self.efficiency_dis * self.world.dt
                revenue_theoretical.append(self.world.pfc[t] * BidSTO_EOM.amount * self.world.dt)

        capacityPrice = abs(sum(revenue_theoretical))
        energyPrice = -self.energy_cost[self.world.currstep] / self.soc[t]

        return capacityPrice, energyPrice
    """