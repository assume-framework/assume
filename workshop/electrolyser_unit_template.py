from datetime import datetime

import pandas as pd

from assume.common.base import SupportsMinMax


class Electrolyser(SupportsMinMax):
    def __init__(
        self,
        id: str,
        technology: str,
        index: pd.DatetimeIndex,
        # Co-code: Include must to have attributes
        # ... ... ...
        max_power: float,
        min_power: float,
        # Co-code: Include key attributes for our Electrolyser class
        # ... ... ...
        **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            index=index,
            **kwargs,
        )
        self.max_power = max_power
        self.min_power = min_power
        # Initialise key atributes
        # ... ... ...
        # Co-code: We'll be coding this part together to set up the initial attributes for our Electrolyser class.

        self.conversion_factors = self.get_conversion_factors()

    def execute_current_dispatch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        end_excl = end - self.index.freq

        # Calculate mean power for this time period
        avg_power = abs(self.outputs["energy"].loc[start:end_excl]).mean()

        # Decide which efficiency point to use
        if avg_power < self.min_power:
            self.outputs["energy"].loc[start:end_excl] = 0
            self.outputs["hydrogen"].loc[start:end_excl] = 0
        else:
            if avg_power <= 0.35 * self.max_power:
                dynamic_conversion_factor = self.conversion_factors[0]
            else:
                dynamic_conversion_factor = self.conversion_factors[1]

            self.outputs["energy"].loc[start:end_excl] = avg_power
            self.outputs["hydrogen"].loc[start:end_excl] = (
                avg_power / dynamic_conversion_factor
            )

        return self.outputs["energy"].loc[start:end_excl]

    def calculate_min_max_power(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        hydrogen_demand=0,
    ):
        # check if hydrogen_demand is within min and max hydrogen production
        # and adjust accordingly

        # Co-code: Check if the hydrogen demand is below the minimum production level
        # Co-code:Check if the hydrogen demand is above the maximum production level
        # Co-code:If the hydrogen demand is within the allowable range, proceed as usual

        # Co-code:get dynamic conversion factor below
        dynamic_conversion_factor = None
        
        # Calculate the gross power needed for the useful hydrogen
        # Co-code:... ... ...
        return power, hydrogen_production

    # Adjust efficiency based on power
    def get_dynamic_conversion_factor(self, hydrogen_demand=None):
        # Adjust efficiency based on power
        if hydrogen_demand <= 0.35 * self.max_hydrogen:
            return self.conversion_factors[0]
        else:
            return self.conversion_factors[1]

    def get_conversion_factors(self):
        # Calculate the conversion factor for the two efficiency points
        conversion_point_1 = (0.3 * self.max_power) / (
            0.35 * self.max_hydrogen
        )  # MWh / Tonne
        conversion_point_2 = self.max_power / self.max_hydrogen  # MWh / Tonne

        return [conversion_point_1, conversion_point_2]

    def as_dict(self) -> dict:
        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "max_power": self.max_power,
                "min_power": self.min_power,
                "min_hydrogen": self.min_hydrogen,
                "max_hydrogen": self.max_hydrogen,
                "unit_type": "electrolyzer",
            }
        )
        return unit_dict
