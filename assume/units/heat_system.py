# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pyomo.environ as pyo

from assume.common.base import SupportsMinMax
from assume.units.dsm_load_shift import DSMFlex


class HeatSystem(DSMFlex, SupportsMinMax):

    required_technologies = ["heat_pump"]

    optional_technologies = [
        "thermal_storage",
        "boiler",
    ]

    def __init__(
        self,
        id,
        unit_operator,
        bidding_strategies,
        forecaster,
        components=None,
        technology="heat_system",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        cost_tolerance=10,
        node="node0",
        location=(0.0, 0.0),
        enforce_terminal_soc=True,
        **kwargs,
    ):
        if components is None:
            components = {}

        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            components=components,
            bidding_strategies=bidding_strategies,
            forecaster=forecaster,
            node=node,
            location=location,
            **kwargs,
        )

        # Required technologies
        for required_technology in self.required_technologies:
            if not any(
                component.startswith(required_technology)
                for component in self.components
            ):
                raise ValueError(
                    f"HeatSystem '{id}' requires at least one "
                    f"'{required_technology}' component."
                )

        # Heat demand

        self.heat_demand = forecaster.heat_demand

        # Component groups

        self.heat_pumps = [
            k
            for k in self.components
            if k.startswith("heat_pump")
        ]

        self.boilers = [
            k
            for k in self.components
            if k.startswith("boiler")
        ]

        self.thermal_storages = [
            k
            for k in self.components
            if k.startswith("thermal_storage")
        ]

        # Optimisation settings

        self.objective = objective
        self.flexibility_measure = flexibility_measure
        self.cost_tolerance = cost_tolerance
        self.enforce_terminal_soc = enforce_terminal_soc

    def define_parameters(self):

        # Electricity price
        self.model.electricity_price = pyo.Param(
            self.model.time_steps,
            initialize={
                t: value
                for t, value in enumerate(
                    self._values_for_model(
                        self.forecaster.electricity_price
                    )
                )
            },
        )
        # Heat demand

        self.model.heat_demand = pyo.Param(
            self.model.time_steps,
            initialize={
                t: value
                for t, value in enumerate(
                    self._values_for_model(
                        self.heat_demand
                    )
                )
            },
        )
        # Fuel prices required by boilers

        boiler_fuel_types = {
            self.components[boiler].get(
                "fuel_type",
                "electricity",
            )
            for boiler in self.boilers
        }

        if "natural_gas" in boiler_fuel_types:
            self.model.natural_gas_price = pyo.Param(
                self.model.time_steps,
                initialize={
                    t: value
                    for t, value in enumerate(
                        self._values_for_model(
                            self.forecaster.get_price(
                                "natural gas"
                            )
                        )
                    )
                },
            )

        if "hydrogen_gas" in boiler_fuel_types:
            self.model.hydrogen_gas_price = pyo.Param(
                self.model.time_steps,
                initialize={
                    t: value
                    for t, value in enumerate(
                        self._values_for_model(
                            self.forecaster.get_price(
                                "hydrogen"
                            )
                        )
                    )
                },
            )

    def define_variables(self):

        # Total electrical consumption of the HeatSystem
        self.model.total_power_input = pyo.Var(
            self.model.time_steps,
            within=pyo.NonNegativeReals,
        )

        # Can become negative when electricity prices are negative
        self.model.variable_cost = pyo.Var(
            self.model.time_steps,
            within=pyo.Reals,
        )

    def initialize_process_sequence(self):
        # Local heat balance

        @self.model.Constraint(self.model.time_steps)
        def heat_balance(m, t):

            heat_supply = 0
            storage_charge = 0

            for hp in self.heat_pumps:
                heat_supply += (
                    m.dsm_blocks[hp].heat_out[t]
                )

            for boiler in self.boilers:
                heat_supply += (
                    m.dsm_blocks[boiler].heat_out[t]
                )

            for storage in self.thermal_storages:
                heat_supply += (
                    m.dsm_blocks[storage].discharge[t]
                )

                storage_charge += (
                    m.dsm_blocks[storage].charge[t]
                )

            return (
                heat_supply
                == m.heat_demand[t]
                + storage_charge
            )

    def define_constraints(self):
        # Total electrical consumption

        @self.model.Constraint(self.model.time_steps)
        def total_power_input_constraint(m, t):
            return (
                m.total_power_input[t]
                == self._component_power_expr(m, t)
            )
        
        # Total operating cost
        @self.model.Constraint(self.model.time_steps)
        def variable_cost_constraint(m, t):
            return (
                m.variable_cost[t]
                == pyo.quicksum(
                    m.dsm_blocks[block].operating_cost[t]
                    for block in m.dsm_blocks
                )
            )

        # Terminal TES SOC
        if (
            self.enforce_terminal_soc
            and self.horizon_mode == "full_horizon"
        ):

            last_time_step = self.model.time_steps.last()

            @self.model.Constraint(
                self.thermal_storages
            )
            def terminal_soc_constraint(m, storage):
                return (
                    m.dsm_blocks[storage].soc[
                        last_time_step
                    ]
                    == m.dsm_blocks[storage].initial_soc
                )