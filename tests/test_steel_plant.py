# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd

from assume.common.forecasts import NaiveForecast
from assume.units.dst_components import Electrolyser
from assume.units.steel_plant import SteelPlant


class TestSteelPlant(unittest.TestCase):
    def setUp(self):
        # Define test data
        index = pd.date_range(start="2024-01-01", periods=4, freq="H")
        availability = 1
        fuel_price = [10, 11, 12, 13]  # Example fuel price data
        co2_price = [10, 20, 30, 30]  # Example CO2 price data

        # Initialize the forecaster with similar parameters as NaiveForecast
        self.forecaster = NaiveForecast(
            index=index,
            availability=availability,
            fuel_price=fuel_price,
            co2_price=co2_price,
        )
        components = {
            "electrolyser": {
                "efficiency": 0.9,
                "rated_power": 100,
                "min_power": 50,
                "ramp_up": 10,
                "ramp_down": 10,
                "min_operating_time": 1,
                "min_down_time": 1,
                "fuel_type": "electricity",
            },
            "h2storage": {
                "max_capacity": 1000,
                "min_capacity": 100,
                "initial_soc": 0.5,
                "storage_loss_rate": 0.01,
                "charge_loss_rate": 0.02,
                "discharge_loss_rate": 0.02,
            },
            "dri_plant": {
                "specific_hydrogen_consumption": 0.5,
                "specific_natural_gas_consumption": 0.6,
                "specific_electricity_consumption": 0.7,
                "specific_iron_ore_consumption": 0.8,
                "rated_power": 100,
                "min_power": 50,
                "fuel_type": "hydrogen",
                "ramp_up": 10,
                "ramp_down": 10,
                "min_operating_time": 1,
                "min_down_time": 1,
            },
            "dri_storage": {
                "max_capacity": 1000,  # Example value for max capacity
                "min_capacity": 100,  # Example value for min capacity
                "initial_soc": 500,  # Example value for initial state of charge
                "storage_loss_rate": 0.1,  # Example value for storage loss rate
                "charge_loss_rate": 0.05,  # Example value for charge loss rate
                "discharge_loss_rate": 0.05,  # Example value for discharge loss rate
            },
            "eaf": {
                "rated_power": 100,  # Example value for rated power
                "min_power": 50,  # Example value for min power
                "specific_electricity_consumption": 0.7,  # Example value for specific electricity consumption
                "specific_dri_demand": 0.8,  # Example value for specific DRI demand
                "specific_lime_demand": 0.9,  # Example value for specific lime demand
                "ramp_up": 10,  # Example value for ramp up rate
                "ramp_down": 10,  # Example value for ramp down rate
                "min_operating_time": 1,  # Example value for min operating time
                "min_down_time": 1,  # Example value for min down time
            },
        }

        self.steel_plant = SteelPlant(
            id="test_plant",
            unit_operator="Hamburg DE",
            bidding_strategies={},
            index=index,
            components=components,
            objective="min_variable_cost",
            demand=1000,
            forecaster=self.forecaster,
            location=(0.0, 0.0),
        )
        self.steel_plant.forecaster = self.forecaster

    def test_initialization(self):
        self.assertEqual(self.steel_plant.id, "test_plant")
        self.assertEqual(self.steel_plant.unit_operator, "Hamburg DE")
        self.assertEqual(self.steel_plant.objective, "min_variable_cost")
        self.assertEqual(len(self.steel_plant.components), 5)
        self.assertEqual(self.steel_plant.steel_demand, 1000)
        self.assertEqual(self.steel_plant.location, (0.0, 0.0))

    def test_define_sets(self):
        self.assertEqual(len(self.steel_plant.model.time_steps), 4)

    def test_define_parameters(self):
        self.assertEqual(len(self.steel_plant.model.electricity_price), 4)
        self.assertEqual(len(self.steel_plant.model.natural_gas_price), 4)
        self.assertEqual(self.steel_plant.model.steel_demand.value, 1000)

    def test_define_variables(self):
        self.assertEqual(len(self.steel_plant.model.total_power_input), 4)
        self.assertEqual(len(self.steel_plant.model.variable_cost), 4)

    def test_define_constraints(self):
        # Ensure that the total_power_input_constraint is defined and not None
        self.assertIsNotNone(self.steel_plant.model.total_power_input_constraint)

        # Assert that lower bound is 0
        for t in self.steel_plant.model.time_steps:
            self.assertEqual(
                self.steel_plant.model.total_power_input_constraint[t].lower, 0
            )

        # Assert that upper bound is None (indicating no upper bound)
        for t in self.steel_plant.model.time_steps:
            self.assertIsNone(
                self.steel_plant.model.total_power_input_constraint[t].upper
            )

    def test_define_objective(self):
        self.assertTrue(hasattr(self.steel_plant.model, "obj_rule"))

    def test_calculate_marginal_cost(self):
        # Define input parameters
        start_time = datetime(2024, 1, 1, 0, 0)
        steel_output = 500  # Example steel output

        # Call the method
        marginal_cost = self.steel_plant.calculate_marginal_cost(
            start_time, steel_output
        )

        # Assert the output
        self.assertIsInstance(marginal_cost, float)

    # def test_calculate_marginal_cost(self):
    #     start_time = datetime(2024, 1, 1, 0, 0)
    #     marginal_cost = self.steel_plant.calculate_marginal_cost(start_time, 500)
    #     self.assertIsInstance(marginal_cost, float)


if __name__ == "__main__":
    unittest.main()
