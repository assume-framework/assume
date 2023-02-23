from common.utils import initializer
from base_unit import BaseUnit, OperationalWindow


class PowerPlant(BaseUnit):
    @initializer
    def __init__(self,
                 id: str,
                 technology: str,
                 node: str,
                 max_power: float,
                 min_power: float,
                 efficiency: float,
                 fuel_type: str,
                 fuel_price: list,
                 co2_price: list,
                 emission_factor: float,
                 ramp_up: float = -1,
                 ramp_down: float = -1,
                 fixed_cost: float = 0,
                 hot_start_cost: float = 0,
                 warm_start_cost: float = 0,
                 cold_start_cost: float = 0,
                 min_operating_time: float = -1,
                 min_down_time: float = -1,
                 heat_extraction: bool = False,
                 max_heat_extraction: float = 0,
                 availability: dict = None,
                 **kwargs):
                 
        super().__init__()

        self.ramp_up = ramp_up if ramp_up > 0 else max_power
        self.ramp_down = ramp_down if ramp_down > 0 else max_power
        self.min_operating_time = max(min_operating_time, 0)
        self.min_down_time = max(min_down_time, 0)

        self.hot_start_cost *= self.max_power
        self.warm_start_cost *= self.max_power
        self.cold_start_cost *= self.max_power


    def reset(self):
        self.current_time_step = 0
        self.current_status = 1
        self.current_down_time = self.min_down_time

        self.total_power_output = [0.5*self.max_power]
        self.total_heat_output = [0.]
        self.pos_capacity_reserve = [0.]
        self.neg_capacity_reserve = [0.]

        self.partial_load_eff = []


    def calculate_operational_window(self) -> OperationalWindow:
        """Calculate the operation window for the next time step."""
        
        t = self.current_time_step

        current_power = self.total_power_output[t-1]

        if self.availability is None:
            min_power = max(self.total_power_output[t-1] - self.ramp_down, 
                            min_power)
            max_power = min(self.total_power_output[t-1] + self.ramp_up, 
                            max_power)
        
        elif self.availability[t] >= min_power:
            min_power = max(self.total_power_output[t-1] - self.ramp_down, 
                            min_power)
            max_power = min(self.total_power_output[t-1] + self.ramp_up, 
                            self.availability[t])
       
        else:
            min_power = 0.
            max_power = 0.
        
        operational_window = {'current_power': {'power': current_power,
                                                'marginal_cost': self.calc_marginal_cost(power_output=current_power,
                                                                                         partial_load_eff=True)},
                              'min_power': {'power': min_power,
                                            'marginal_cost': self.calc_marginal_cost(power_output=min_power,
                                                                                     partial_load_eff=True)},
                              'max_power': {'power': max_power,
                                            'marginal_cost': self.calc_marginal_cost(power_output=max_power,
                                                                                     partial_load_eff=True)}
                              }

        return operational_window

    
    def calc_marginal_cost(self,
                           power_output: float,
                           partial_load_eff: bool = False):

        t = self.current_time_step

        fuel_price = self.fuel_price[t]
        co2_price = self.co2_price[t]

        # Partial load efficiency dependent marginal costs
        if not partial_load_eff:
            marginal_cost = fuel_price/self.efficiency \
                + co2_price * self.emission_factor/self.efficiency \
                    + self.fixed_cost
            
            return marginal_cost


        capacity_ratio = power_output / self.max_power

        if self.fuel_type in ['lignite', 'hard coal']:
            eta_loss = 0.095859 * (capacity_ratio ** 4) \
                - 0.356010 * (capacity_ratio ** 3) \
                    + 0.532948 * (capacity_ratio ** 2) \
                        - 0.447059 * capacity_ratio \
                            + 0.174262

        elif self.fuel_type == 'combined cycle gas turbine':
            eta_loss = 0.178749 * (capacity_ratio ** 4) \
                - 0.653192 * (capacity_ratio ** 3) \
                    + 0.964704 * (capacity_ratio ** 2) \
                        - 0.805845 * capacity_ratio \
                            + 0.315584

        elif self.fuel_type == 'open cycle gas turbine':
            eta_loss = 0.485049 * (capacity_ratio ** 4) \
                - 1.540723 * (capacity_ratio ** 3) \
                    + 1.899607 * (capacity_ratio ** 2) \
                        - 1.251502 * capacity_ratio \
                            + 0.407569

        else:
            eta_loss = 0

        efficiency = self.efficiency - eta_loss

        marginal_cost = fuel_price/efficiency \
            + co2_price * self.emission_factor/efficiency \
                + self.variableCosts

        return marginal_cost 