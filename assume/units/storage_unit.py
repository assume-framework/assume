import pandas as pd

from assume.strategies import BaseStrategy
from assume.units.base_unit import BaseUnit

class StorageUnit(BaseUnit):
    """A class for a storage unit.

    Attributes
    ----------
    id : str
        The ID of the storage unit.
    technology : str
        The technology of the storage unit.
    node : str
        The node of the storage unit.
    max_power_charge : float
        The maximum power input of the storage unit in MW (negative value).
    min_power_charge : float
        The minimum power input of the storage unit in MW (negative value).
    max_power_discharge : float
        The maximum power output of the storage unit in MW.
    min_power_discharge : float
        The minimum power output of the storage unit in MW.
    max_SOC : float
        The maximum state of charge of the storage unit in MWh (equivalent to capacity).
    min_SOC : float
        The minimum state of charge of the storage unit in MWh.
    efficiency_charge : float
        The efficiency of the storage unit while charging.
    efficiency_discharge : float
        The efficiency of the storage unit while discharging.
    variable_cost_charge : float
        Variable costs to charge the storage unit in €/MW.
    variable_costs_discharge : float
        Variable costs to discharge the storage unit in €/MW.
    emission_factor : float
        The emission factor of the storage unit.
    ramp_up_charge : float, optional
        The ramp up rate of charging the storage unit in MW/15 minutes (negative value).
    ramp_down_charge : float, optional
        The ramp down rate of charging the storage unit in MW/15 minutes (negative value).
    ramp_up_discharge : float, optional
        The ramp up rate of discharging the storage unit in MW/15 minutes.
    ramp_down_discharge : float, optional
        The ramp down rate of discharging the storage unit in MW/15 minutes.
    fixed_cost : float, optional
        The fixed cost of the storage unit in €/MW. (related to capacity?)
    hot_start_cost : float, optional
        The hot start cost of the storage unit in €/MW.
    warm_start_cost : float, optional
        The warm start cost of the storage unit in €/MW.
    cold_start_cost : float, optional
        The cold start cost of the storage unit in €/MW.
    downtime_hot_start : float, optional
        Definition of downtime before hot start in h.
    downtime_warm_start : float
        Definition of downtime before warm start in h.
    min_operating_time : float, optional
        The minimum operating time of the storage unit in hours.
    min_down_time : float, optional
        The minimum down time of the storage unit in hours.
    min_down_time : float, optional
        The minimum down time of the storage unit in hours.
    availability : dict, optional
        The availability of the storage unit in MW for each time step.
    is_active: bool
        Defines whether or not the unit bids itself or is portfolio optimized.
    bidding_startegy: str
        In case the unit is active it has to be defined which bidding strategy should be used
    **kwargs
        Additional keyword arguments.

    Methods
    -------
    reset()
        Reset the storage unit.
    calculate_operational_window()
        Calculate the operation window for the next time step.
    calc_marginal_cost(power_output, partial_load_eff)
        Calculate the marginal cost of the storage unit.
    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        max_power_charge: float or pd.Series,
        max_power_discharge: float or pd.Series,
        max_SOC: float,
        min_power_charge: float or pd.Series = 0.0,  
        min_power_discharge: float or pd.Series = 0.0,
        min_SOC: float  = 0.0,
        efficiency_charge: float = 1,
        efficiency_discharge: float = 1,
        variable_cost_charge: float or pd.Series = 0.0,
        variable_cost_discharge: float or pd.Series = 0.0,
        price_forecast: pd.Series = None,
        emission_factor: float = 0.0,
        ramp_up_charge: float = 0.0,
        ramp_down_charge: float = 0.0,
        ramp_up_discharge: float = -1,
        ramp_down_discharge: float = -1,
        fixed_cost: float = 0,
        hot_start_cost: float = 0,
        warm_start_cost: float = 0,
        cold_start_cost: float = 0,
        min_operating_time: float = 0,
        min_down_time: float = 0,
        downtime_hot_start: int = 8,  # hours
        downtime_warm_start: int = 48,  # hours
        index: pd.DatetimeIndex = None,
        location: tuple[float, float] = None,
        node: str = None,
        **kwargs
    ):
        super().__init__(
            id=id,
            technology=technology,
            node=node,
            bidding_strategies=bidding_strategies,
            index=index,
            unit_operator=unit_operator,
        )

        self.max_power_charge = max_power_charge
        self.min_power_charge = min_power_charge
        self.max_power_discharge = max_power_discharge
        self.min_power_discharge = min_power_discharge
        self.min_SOC = min_SOC
        self.max_SOC = max_SOC
        self.efficiency_charge = efficiency_charge if efficiency_charge > 0 else 1
        self.efficiency_discharge = efficiency_discharge if efficiency_discharge > 0 else 1
        self.variable_cost_charge = variable_cost_charge
        self.variable_cost_discharge = variable_cost_discharge
        self.price_forecast = (
            price_forecast if price_forecast is not None else pd.Series(0, index=index)
        )
        self.emission_factor = emission_factor

        self.ramp_up_charge = ramp_up_charge
        self.ramp_down_charge = ramp_down_charge
        self.ramp_up_discharge = ramp_up_discharge
        self.ramp_down_discharge = ramp_down_discharge
        self.min_operating_time = min_operating_time
        self.min_down_time = min_down_time
        self.downtime_hot_start = downtime_hot_start
        self.warm_start_cost = downtime_warm_start

        self.fixed_cost = fixed_cost
        # Do the start_up costs make sense for storages? Same for charging?
        self.hot_start_cost = hot_start_cost * max_power_discharge
        self.warm_start_cost = warm_start_cost * max_power_discharge
        self.cold_start_cost = cold_start_cost * max_power_discharge

        self.location = location


    def reset(self):
        """Reset the unit to its initial state."""

        #current_status = 0 means the unit is not dispatched
        self.current_status = 1
        self.current_down_time = self.min_down_time
        

        #total_power > 0 discharging, total_power < 0 charging
        self.total_power = pd.Series(0.0, index=self.index)

        #always starting with discharging?
        self.total_power.iat[0] = self.min_power_discharge

        #starting half way charged
        self.current_SOC = self.max_SOC * 0.5

        self.pos_capacity_reserve = pd.Series(0.0, index=self.index)
        self.neg_capacity_reserve = pd.Series(0.0, index=self.index)

        self.mean_market_success = 0
        self.market_success_list = [0]


    def calculate_operational_window(
        self,
        product_type: str,
        product_tuple: tuple,
    ) -> dict:
        """Calculate the operation window for the next time step.

        Returns
        -------
        operational_window : dict
            Dictionary containing the operational window for the next time step.
        """
        start, end, only_hours = product_tuple
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        # TODO remove current_time from flexable_strategy, so that the product config is always used
        self.current_time = start

        if self.current_status == 0 and self.current_down_time < self.min_down_time:
            return None

        #current_power = self.total_power_exchange.at[self.current_time]

        current_power_discharge = (
            self.total_power.at[self.current_time]
            if self.total_power.at[self.current_time] > 0
            else 0)
        
        current_power_charge = (
            self.total_power.at[self.current_time]
            if self.total_power.at[self.current_time] < 0
            else 0)

        min_power_discharge = (
            self.min_power_discharge[self.current_time]
            if type(self.min_power_discharge) is pd.Series
            else self.min_power_discharge
        )
        min_power_charge = (
            self.min_power_charge[self.current_time]
            if type(self.min_power_charge) is pd.Series
            else self.min_power_charge
        )

        max_power_discharge = (
            self.max_power_discharge[self.current_time]
            if type(self.max_power_discharge) is pd.Series
            else self.max_power_discharge
        )
        max_power_charge = (
            self.max_power_charge[self.current_time]
            if type(self.max_power_charge) is pd.Series
            else self.max_power_charge
        )

        #was charging before
        if self.min_down_time > 0 and self.total_power < 0:
            min_power_discharge = 0
            max_power_discharge = 0
        else:
            if self.ramp_down_discharge != -1:
                min_power_discharge = max(current_power_discharge - self.ramp_down_discharge, 
                                        min_power_discharge)
            
            if self.ramp_up_discharge != -1:
                max_power_discharge = min(self.ramp_up_discharge + current_power_discharge, 
                                        max_power_discharge)

        #was discharging before
        if self.min_down_time > 0 and self.total_power > 0:
            min_power_charge = 0
            max_power_charge = 0
        else:
            if self.ramp_down_charge < 0:
                min_power_charge = min(current_power_charge - self.ramp_down_charge, 
                                    min_power_charge)
                
            if self.ramp_up_charge < 0:
                max_power_charge = max(current_power_charge + self.ramp_up_charge, 
                                    max_power_charge)
        
        
        min_SOC = (self.min_SOC[self.current_time]
                   if type(self.min_SOC) is pd.Series
                   else self.min_SOC)
        max_SOC = (self.max_SOC[self.current_time]
                   if type(self.max_SOC) is pd.Series
                   else self.max_SOC)    
        
        #restrict according to min_SOC
        max_power_discharge = min(max_power_discharge, 
                                  max(0,(self.current_SOC - min_SOC 
                                         - self.pos_capacity_reserve[self.current_time])
                                         *self.efficiency_discharge))
        
        #restrict charging according to max_SOC
        max_power_charge = max(max_power_charge, min(0, self.current_SOC - max_SOC - self.neg_capacity_reserve[self.current_time]))
        
        #what form does the operational window have?
        operational_window = {
            "window": {"start": start, "end": end},
            "current_power_discharge": {
                "power_discharge": current_power_discharge,
                "marginal_cost": self.calc_marginal_cost(
                    current_time=self.current_time,
                    discharge=True,
                ),
            },
            "current_power_charge": {
                "power_charge": current_power_charge,
                "marginal_cost": self.calc_marginal_cost(
                    current_time=self.current_time,
                    discharge=False,
                ),
            },
            "min_power_discharge": {
                "power_discharge": min_power_discharge,
                "marginal_cost": self.calc_marginal_cost(
                    current_time=self.current_time,
                    discharge=True,
                ),
            },
            "max_power_discharge": {
                "power_discharge": max_power_discharge,
                "marginal_cost": self.calc_marginal_cost(
                    current_time=self.current_time,
                    discharge=True,
                ),
            },
            "min_power_charge": {
                "power_charge": min_power_charge,
                "marginal_cost": self.calc_marginal_cost(
                    current_time=self.current_time,
                    discharge=False,
                )
            },
            "max_power_charge": {
                "power_charge": max_power_charge,
                "marginal_cost": self.calc_marginal_cost(
                    current_time=self.current_time,
                    discharge=False,
                )
            },
        }

        return operational_window
    
    def calculate_bids(
        self,
        product_type,
        product_tuple,
    ):
        return super().calculate_bids(
            product_type=product_type,
            product_tuple=product_tuple,
        )
    
    def get_dispatch_plan(self, dispatch_plan, time_period):
        if (dispatch_plan["total_power"] > self.min_power_discharge):
            self.market_success_list[-1] += 1
            self.current_status = 1 #discharging
            self.current_down_time = 0
            self.total_power.loc[time_period] = dispatch_plan["total_power"]
            self.current_SOC = self.current_SOC - dispatch_plan["total_power"]

        elif dispatch_plan["total_power"] < -self.min_power_charge:
            self.market_success_list[-1] += 1
            self.current_status = 1 #charging
            self.current_down_time = 0
            self.total_power.loc[time_period] = dispatch_plan["total_power"]
            self.current_SOC = self.current_SOC - dispatch_plan["total_power"]

        elif dispatch_plan["total_power"] < self.min_power_discharge:
            self.current_status = 0
            self.current_down_time += 1
            self.total_power.loc[time_period] = 0

            if self.market_success_list[-1] != 0:
                self.mean_market_success = sum(self.market_success_list) / len(
                    self.market_success_list
                )
                self.market_success_list.append(0)
        
    
    def calc_marginal_cost(
        self,
        current_time: pd.Timestamp,
        discharge: bool = True,
    ) -> float:
        if discharge == True:
            variable_cost = (
                self.variable_cost_discharge.at[current_time]
                if type(self.variable_cost_discharge) is pd.Series
                else self.variable_cost_discharge
            )
            efficiency = self.efficiency_discharge
    
        else:
            variable_cost = (
                self.variable_cost_charge.at[current_time]
                if type(self.variable_cost_charge) is pd.Series
                else self.variable_cost_charge
            )
            efficiency = self.efficiency_charge

        marginal_cost = (
            variable_cost / efficiency
            + self.fixed_cost
        )

        return marginal_cost

        