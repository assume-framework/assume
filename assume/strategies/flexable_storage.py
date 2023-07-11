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
        start = operational_window["window"][0]
        end = operational_window["window"][1]
        
        bid_quantity = operational_window["ops"]["pos_reserve"]["volume"]
        if bid_quantity == 0:
            return []
        
        marginal_cost = operational_window["ops"]["pos_reserve"]["cost"]

        duration = (end - start).total_seconds() / 3600
        
        SOC_theoretical = unit.current_SOC
        revenue_theoretical = []

        for tick in pd.date_range(start, end-pd.Timedelta('1h'), freq='h'):
            
            average_price = calculate_price_average(unit, tick, self.foresight)
            operational_window_theoretical = unit.calculate_pos_reserve_operational_window(
                start=tick,
                end=pd.Timestamp(tick+pd.Timedelta(duration, 'h'))
            )
            bid_quantity_theoretical = operational_window_theoretical["ops"]["pos_reserve"]["volume"]
   
            if (
                unit.price_forecast[tick] >= average_price / unit.efficiency_discharge
            ) and (bid_quantity_theoretical > 0):
                # place bid to discharge
                SOC_theoretical -= bid_quantity_theoretical / unit.efficiency_discharge
                revenue_theoretical.append(unit.price_forecast[tick] * bid_quantity_theoretical)

        capacity_price = abs(sum(revenue_theoretical))
        #TODO compare to flexable: energyPrice = -self.dictEnergyCost[self.world.currstep] / self.dictSOC[t]
        '''
        with 
        self.dictEnergyCost[self.world.currstep + 1] = (self.dictEnergyCost[self.world.currstep] 
                                                                + self.dictCapacity[self.world.currstep] 
                                                                * self.world.PFC[self.world.currstep] * self.world.dt) 
        and dictCapacity aggregted bid.confirmedAmount
        whats the difference to capacity_price?
        '''
        energy_price = -marginal_cost / unit.current_SOC

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
        #if bid_quantity >= min_bid_neg_CRM   

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