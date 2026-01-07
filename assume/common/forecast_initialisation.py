# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections.abc import Callable

import pandas as pd

# for locational marginal price forecast
import pypsa
from assume.common.grid_utils import read_pypsa_grid
from assume.common.utils import create_incidence_matrix
from assume.common.market_objects import MarketConfig
import numpy as np



def _ensure_not_none(
    df: pd.DataFrame | None, index: pd.DatetimeIndex | pd.Series, check_index=False
) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(index=index)
    if check_index and index.freq != df.index.inferred_freq:
        raise ValueError("Forecast frequency does not match index frequency.")
    return df


class ForecastInitialisation:
    """
    This class represents a forecaster that provides timeseries for forecasts derived from existing files.

    It initializes with the provided index and configuration data, including power plants, demand units,
    and market configurations. The forecaster also supports optional inputs like DSM (demand-side management) units,
    buses, and lines for more advanced forecasting scenarios.

    Methods are included to retrieve forecasts for specific columns, availability of units,
    and prices of fuel types, returning the corresponding timeseries as pandas Series.

    Note:
    - Some built-in forecasts are calculated at the beginning of the simulation, such as price forecast and residual load forecast.
    - Price forecast is calculated for energy-only markets using a merit order approach.
    - Residual load forecast is calculated by subtracting the total available power from variable renewable energy power plants from the overall demand forecast. Only power plants containing 'wind' or 'solar' in their technology column are considered VRE power plants.

    Args:
        index (FastIndex | pd.Series | pd.DatetimeIndex): The index of the forecasts.
        powerplants_units (pd.DataFrame): A DataFrame containing information about power plants.
        demand_units (pd.DataFrame): A DataFrame with demand unit data.
        market_configs (dict[str, dict]): Configuration details for the markets.
        buses (pd.DataFrame | None, optional): A DataFrame of buses information. Defaults to None.
        lines (pd.DataFrame | None, optional): A DataFrame of line information. Defaults to None.
        *args (object): Additional positional arguments.
        **kwargs (object): Additional keyword arguments.

    """

    def __init__(
        self,
        index: pd.Series | pd.DatetimeIndex,
        powerplants_units: pd.DataFrame,
        demand_units: pd.DataFrame,
        market_configs: dict[str, dict],
        demand: pd.DataFrame = None,
        availability: pd.DataFrame = None,
        exchanges: pd.DataFrame = None,
        forecasts: pd.DataFrame = None,
        fuel_prices: pd.DataFrame = None,
        exchange_units: pd.DataFrame = None,
        buses: pd.DataFrame = None,
        lines: pd.DataFrame = None,
        storage_units: pd.DataFrame = None,
    ):
        self.index = index
        self._logger = logging.getLogger(__name__)
        self.powerplants_units = powerplants_units
        self.demand_units = demand_units
        self.market_configs = market_configs
        self.exchange_units = exchange_units
        self.buses = buses
        self.lines = lines
        self.demand = _ensure_not_none(demand, index, check_index=True)
        self.exchanges = _ensure_not_none(exchanges, index, check_index=True)
        self.storage_units = _ensure_not_none(storage_units, index)

        fuel_prices = _ensure_not_none(fuel_prices, index)
        if len(fuel_prices) <= 1:  # single value provided, extend to full index
            fuel_prices.index = index[:1]
            fuel_prices = fuel_prices.reindex(index, method="ffill")
        self.fuel_prices = fuel_prices
        self._forecasts = _ensure_not_none(forecasts, index, check_index=True)
        self._availability = _ensure_not_none(availability, index, check_index=True)

    def forecasts(self, id: str) -> pd.Series:
        if self._forecasts is not None and id in self._forecasts.columns:
            return self._forecasts[id]
        return pd.Series(0.0, self.index, name=id)

    def availability(self, id: str) -> pd.Series:
        if self._availability is not None and id in self._availability.columns:
            return self._availability[id]
        return pd.Series(1.0, self.index, name=f"availability_{id}")

    def calculate_market_forecasts(
        self,
    ) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
        """Calculate market-specific price and residual load forecasts."""
        price_forecasts: dict[str, pd.Series] = {}
        residual_loads: dict[str, pd.Series] = {}

        for market_id, config in self.market_configs.items():
            if config["product_type"] != "energy":
                self._logger.warning(
                    f"Price forecast could not be calculated for {market_id}. It can only be calculated for energy-only markets for now."
                )
                continue
            price_forecasts[market_id] = self._forecasts.get(f"price_{market_id}")
            if price_forecasts[market_id] is None:
                # calculate if not present
                if config['market_mechanism'] in ["redispatch", "nodal_clearing"]:
                    forecast = self.calculate_locational_marginal_price_forecast(market_id)
                    price_forecasts['LMPs'] = forecast
                else:
                    forecast = self.calculate_market_price_forecast(market_id)
                    price_forecasts[market_id] = forecast

            residual_loads[market_id] = self._forecasts.get(
                f"residual_load_{market_id}"
            )
            if residual_loads[market_id] is None:
                # calculate if not present
                load_forecast = self.calculate_residual_load_forecast(market_id)
                residual_loads[market_id] = load_forecast

        return price_forecasts, residual_loads

    def calc_node_forecasts(self):
        if self.buses is None or self.lines is None:
            return None, None
        if (
            "node" not in self.demand_units.columns
            or not self.demand_units["node"].isin(self.buses.index).all()
        ):
            self._logger.warning(
                "Node-specific congestion signals and renewable utilisation forecasts could not be calculated. "
                "Either 'node' column is missing in demand_units or nodes are not available in buses."
            )
            return None, None

        nodes = {node for node in self.demand_units["node"].unique()}
        all_nodes = nodes | {"all"}
        return (
            self._calc_unpresent(nodes, self.calc_congestion_forecast),
            self._calc_unpresent(all_nodes, self.calc_renewable_utilisation),
        )

    def _calc_unpresent(
        self, columns: set, calculation_func: Callable[[], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Checks if all specified columns are present in the forecasts, if not calculate them using the provided function.
        """
        if columns.issubset(set(self._forecasts.columns)):
            return self._forecasts[columns]
        df = calculation_func()
        for col in df.columns:
            if col in self._forecasts.columns:
                df[col] = self._forecasts[col]
        return df

    def get_registered_market_participants(self, market_id: str) -> pd.DataFrame:
        """
        Retrieves information about market participants to make accurate price forecasts.

        Currently, the functionality for using different markets and specified registration for
        the price forecast is not implemented, so it returns the power plants as a DataFrame.

        Args:
            market_id (str): The market ID.

        Returns:
            pd.DataFrame: The registered market participants.

        """
        self._logger.warning(
            "Functionality of using the different markets and specified registration for the price forecast is not implemented yet"
        )
        return self.powerplants_units

    def _calc_power(self, mask: pd.Series = None) -> pd.DataFrame:
        """
        Calculates the output power of each power plant based on its availability and maximum power.

        Args:
            mask (pd.Series, optional): An optional boolean mask to filter power plants
        """
        powerplants = self.powerplants_units
        if mask is not None:
            powerplants = powerplants[mask]
        av_list = [self.availability(id) for id in powerplants.index]
        availabilities = pd.DataFrame(av_list).T
        availabilities.columns = powerplants.index
        return powerplants.max_power * availabilities

    def calculate_residual_load_forecast(self, market_id) -> pd.Series:
        """
        This method calculates the residual demand forecast by subtracting the total available power from renewable energy (VRE) power plants from the overall demand forecast for each time step.

        Returns:
            pd.Series: The residual demand forecast.

        Note:
            1. Selects VRE power plants from the powerplants_units DataFrame based on the technology column (wind or solar).
            2. Creates a DataFrame, vre_feed_in_df, with columns representing VRE power plants and initializes it with zeros.
            3. Calculates the power feed-in for each VRE power plant based on its availability and maximum power.
            4. Calculates the residual demand by subtracting the total VRE power feed-in from the overall demand forecast.
            5. If exchange units are available, imports and exports are subtracted and added to the demand forecast, respectively.

        """
        demand_units = self.demand_units[
            self.demand_units[f"bidding_{market_id}"].notnull()
        ]
        sum_demand = self.demand[demand_units.index].sum(axis=1)

        # get exchanges if exchange_units are available
        if self.exchange_units is not None:
            exchange_units = self.exchange_units[
                self.exchange_units[f"bidding_{market_id}"].notnull()
            ]
            # get sum of imports as name of exchange_unit_import
            import_units = [f"{unit}_import" for unit in exchange_units.index]
            sum_imports = self.exchanges[import_units].sum(axis=1)
            # get sum of exports as name of exchange_unit_export
            export_units = [f"{unit}_export" for unit in exchange_units.index]
            sum_exports = self.exchanges[export_units].sum(axis=1)
            # add imports and exports to the sum_demand
            sum_demand += sum_imports - sum_exports

        mask = self.powerplants_units["technology"].str.contains(
            r"\b(?:wind|solar)", case=False, na=False
        )
        vre_feed_in_df = self._calc_power(mask).sum(axis=1)
        if vre_feed_in_df.empty:
            vre_feed_in_df = 0
        res_demand_df = sum_demand - vre_feed_in_df

        return res_demand_df

    def calculate_market_price_forecast(self, market_id) -> pd.Series:
        """
        Computes the merit order price forecast for the entire time horizon.

        This method estimates electricity prices by considering renewable energy infeed, residual demand,
        and marginal costs of power plants. It follows a merit-order approach to determine the price at
        each time step.

        Args:
            market_id (str): The market identifier for which the price forecast is calculated.

        Returns:
            pd.Series: A time-indexed series representing the merit order price forecast.

        Methodology:
            1. Filters power plant units that participate in the specified market.
            2. Calculates the marginal costs for each unit based on fuel costs, efficiencies, emissions, and fixed costs.
            3. Retrieves forecasted unit availabilities and computes available power for each time step.
            4. Aggregates demand forecasts, including imports and exports if applicable.
            5. Iterates over each time step:
                - Sorts power plants by marginal cost.
                - Computes cumulative available power.
                - Sets the price based on the marginal cost of the unit that meets demand.
                - Assigns a default price of 1000 if supply is insufficient.

        Note:
            - Extending the price forecast to additional markets beyond the DAM is planned.
            - Future enhancements may include storage integration in the price forecast.

        """

        # 1. Filter power plant units with a bidding strategy for the given market_id
        powerplants_units = self.powerplants_units[
            self.powerplants_units[f"bidding_{market_id}"].notnull()
        ]

        # 2. Calculate marginal costs for each unit and time step.
        #    The resulting DataFrame has rows = time steps and columns = units.
        marginal_costs = powerplants_units.apply(self.calculate_marginal_cost, axis=1).T

        # 3. Compute available power for each unit at each time step.
        #    Since max_power is a float, this multiplication broadcasts over each column.
        power = self._calc_power()

        # 4. Process the demand.
        #    Filter demand units with a bidding strategy and sum their forecasts for each time step.
        demand_units = self.demand_units[
            self.demand_units[f"bidding_{market_id}"].notnull()
        ]
        sum_demand = self.demand[demand_units.index].sum(axis=1)

        # get exchanges if exchange_units are available
        if self.exchange_units is not None:
            exchange_units = self.exchange_units[
                self.exchange_units[f"bidding_{market_id}"].notnull()
            ]
            # get sum of imports as name of exchange_unit_import
            import_units = [f"{unit}_import" for unit in exchange_units.index]
            sum_imports = self.exchanges[import_units].sum(axis=1)
            # get sum of exports as name of exchange_unit_export
            export_units = [f"{unit}_export" for unit in exchange_units.index]
            sum_exports = self.exchanges[export_units].sum(axis=1)
            # add imports and exports to the sum_demand
            sum_demand += sum_imports - sum_exports

        # 5. Initialize the price forecast series.
        price_forecast = pd.Series(index=self.index, data=0.0)

        # 6. Loop over each time step
        for t in self.index:
            # Get marginal costs and available power for time t (both are Series indexed by unit)
            mc_t = marginal_costs.loc[t]
            power_t = power.loc[t]
            demand_t = sum_demand.loc[t]

            # Sort units by their marginal cost in ascending order for time t.
            sorted_units = mc_t.sort_values().index
            sorted_mc = mc_t.loc[sorted_units]
            sorted_power = power_t.loc[sorted_units]

            # Compute the cumulative sum of available power in the sorted order.
            cumsum_power = sorted_power.cumsum()

            # Find the first unit where the cumulative available power meets or exceeds demand.
            matching_units = cumsum_power[cumsum_power >= demand_t]
            if matching_units.empty:
                # If available capacity is insufficient, set the price to 1000.
                price = 1000.0
            else:
                # The marginal cost of the first unit that meets demand becomes the price.
                price = sorted_mc.loc[matching_units.index[0]]

            price_forecast.loc[t] = price

        return price_forecast
    
    def calculate_locational_marginal_price_forecast(self, market_id: str) -> pd.DataFrame:
        """
        Calculates locational marginal price (LMP) forecasts for each bus and timestep in the network.
        It follows an optimal power flow approach taking into account renewable availability, demand, storages and marginal costs of power plants at each location.

        """
        self.network = pypsa.Network()

        snapshots = self.index
        self.network.set_snapshots(snapshots)
        self.incidence_matrix = None

        if market_id != "redispatch":
            # does not make sense for zonal redispatch markets
            self.zones_id = self.market_configs[market_id]['param_dict'].get("zones_identifier")
            self.node_to_zone = None

            # Generate the incidence matrix and set the nodes based on zones or individual buses
            if self.zones_id:
                # Zonal Case
                self.incidence_matrix = create_incidence_matrix(
                    self.lines, self.buses, zones_id=self.zones_id
                )
                self.nodes = self.buses[self.zones_id].unique()
                self.node_to_zone = self.buses[self.zones_id].to_dict()
            else:
                # Nodal Case
                self.incidence_matrix = create_incidence_matrix(self.lines, self.buses)
                self.nodes = self.buses.index.values
        # if buses and lines dont contain carrier, set it to AC to silence PyPSA warning
        if "carrier" not in self.lines.columns:
            self.lines["carrier"] = "AC"
        if "carrier" not in self.buses.columns:
            self.buses["carrier"] = "AC"
        self.network.add("Bus", self.buses.index, **self.buses)
        self.network.add("Line", self.lines.index, **self.lines)
        
        # add all units to the PyPSA network
         # TODO check if this is already checked before and forecast is market specific
        powerplants_units = self.powerplants_units[
            self.powerplants_units[f"bidding_{market_id}"].notnull()
        ]
        # step 2: calculate marginal costs for each unit and time step.
        marginal_costs = powerplants_units.apply(self.calculate_marginal_cost, axis=1).T
        # step 3: compute available power for each unit at each time step.
        power = self._calc_power()
        # generators have p_min_pu - p_max_pu 0 to 1
        self.network.add(
            "Generator",
            powerplants_units.index,
            bus=powerplants_units["node"],
            p_nom=powerplants_units["max_power"],
            p_min_pu=0,
            p_max_pu=power.div(powerplants_units["max_power"], axis=1),
            marginal_cost=marginal_costs,
        )

        demand_units = self.demand_units[
            self.demand_units[f"bidding_{market_id}"].notnull()
        ]
        sum_demand = self.demand[demand_units.index]
        # demand units have p_min_pu - p_max_pu -1 to 0
        self.network.add(
            "Generator",
            demand_units.index,
            bus=demand_units["node"],
            p_nom=demand_units["max_power"],
            p_min_pu=-1*sum_demand.div(demand_units["max_power"], axis=1),
            p_max_pu=0,
            # temporarily set to max bid price
            marginal_cost=self.market_configs[market_id]['maximum_bid_price'],
        )
        # storage units
        # we take the max of discharging and charging power as p_nom for PyPSA.
        if not self.storage_units.empty:
            storage_units = self.storage_units[
                self.storage_units[f"bidding_{market_id}"].notnull()
            ]
            p_nom = np.maximum(
                storage_units["max_power_discharge"].values,
                storage_units["max_power_charge"].values,
            )
            max_hours = storage_units["capacity"].values / p_nom
            self.network.add(
                "StorageUnit",
                storage_units.index,
                bus=storage_units["node"],
                p_nom=p_nom,
                max_hours=max_hours,
                # check
                marginal_cost=storage_units.get("marginal_cost", 0.0),
            )

        if self.exchange_units is not None:
            exchange_units = self.exchange_units[
                self.exchange_units[f"bidding_{market_id}"].notnull()
            ]
            # get sum of imports as name of exchange_unit_import
            import_units = [f"{unit}_import" for unit in exchange_units.index]
            sum_imports = self.exchanges[import_units].sum(axis=1)
            # get sum of exports as name of exchange_unit_export
            export_units = [f"{unit}_export" for unit in exchange_units.index]
            sum_exports = self.exchanges[export_units].sum(axis=1)
            # add imports and exports to the sum_demand
            sum_demand += sum_imports - sum_exports

            self.network.add(
                "Generator",
                import_units,
                bus=exchange_units["node"],
                p_nom=exchange_units["max_import_power"],
                p_min_pu=0,
                p_max_pu=1,
                # TODO add p as timeseries?
            )
            self.network.add(
                "Generator",
                export_units,
                bus=exchange_units["node"],
                p_nom=exchange_units["max_export_power"],
                p_min_pu=-1,
                p_max_pu=0,
                # TODO add p as timeseries?
            )

        # TODO sicher self hier?
        self.solver = self.market_configs[market_id]['param_dict'].get("solver", "highs")
        if self.solver == "gurobi":
            self.solver_options = {"LogToConsole": 0, "OutputFlag": 0}
        elif self.solver == "highs":
            self.solver_options = {"output_flag": False, "log_to_console": False}
        else:
            self.solver_options = {}

        # run linear optimal powerflow
        self.network.optimize.fix_optimal_capacities()
        status, termination_condition = self.network.optimize(
            solver=self.solver,
            solver_options=self.solver_options,
            progress=False,
        )

        if status != "ok":
            self._logger.error(f"Solver exited with {termination_condition}")
            raise Exception("Solver in nodal clearing forecast did not converge")
        
        # extract lmps
         # make sure the order of the columns is same as in the buses csv
        lmp_forecast = self.network.buses_t.marginal_price.copy()[self.network.buses.index]
        self.zones_id = self.market_configs[market_id]["param_dict"].get("zones_identifier")
        if self.zones_id:
            # map zonal prices to nodes
            lmp_forecast_nodes = pd.DataFrame(index=lmp_forecast.index, columns=self.buses.index)
            for node in self.buses.index:
                zone = self.node_to_zone[node]
                lmp_forecast_nodes[node] = lmp_forecast[zone]
            lmp_forecast = lmp_forecast_nodes

        return lmp_forecast


    def calculate_marginal_cost(self, pp_series: pd.Series) -> pd.Series:
        """
        Calculates time series of marginal costs for a power plant.

        This method calculates the marginal cost of a power plant by taking into account the following factors: \n
        - Fuel costs based on the fuel type and efficiency of the power plant. \n
        - Emissions costs considering the CO2 price and emission factor of the power plant. \n
        - Fixed costs, if specified for the power plant.

        Args:
            pp_series (pandas.Series): Series containing power plant data.

        Returns:
            pandas.Series: The marginal cost of the power plant.

        Note:
            1. Determines the fuel price based on the fuel type of the power plant.
            2. Calculates the fuel cost by dividing the fuel price by the efficiency of the power plant.
            3. Calculates the emissions cost based on the CO2 price and emission factor, adjusted by the efficiency of the power plant.
            4. Considers any fixed costs specified for the power plant.
            5. Aggregates the fuel cost, emissions cost, and fixed cost to obtain the marginal cost of the power plant.
        """

        if pp_series.fuel_type in self.fuel_prices.keys():
            fuel_price = self.fuel_prices[pp_series.fuel_type]
        else:
            fuel_price = pd.Series(0.0, index=self.index)

        emission_factor = pp_series["emission_factor"]
        co2_price = self.fuel_prices["co2"]

        fuel_cost = fuel_price / pp_series["efficiency"]
        emissions_cost = co2_price * emission_factor / pp_series["efficiency"]
        additional_cost = (
            pp_series["additional_cost"] if "additional_cost" in pp_series else 0.0
        )

        marginal_cost = fuel_cost + emissions_cost + additional_cost

        return marginal_cost

    def calc_congestion_forecast(self) -> pd.DataFrame:
        """
        Calculates a collective node-specific congestion signal by aggregating the congestion severity of all
        transmission lines connected to each node, taking into account powerplant load based on availability factors.

        Returns:
            pd.DataFrame: A DataFrame with columns for each node, where each column represents the collective
                          congestion signal time series for that node.
        """

        # Step 1: Calculate load for each powerplant based on availability factor and max power
        availability_factor_df = self._calc_power()

        # Step 2: Calculate net load for each node (demand - generation)
        net_load_by_node = {}

        for node in self.demand_units["node"].unique():
            # Calculate total demand for this node
            node_demand_units = self.demand_units[
                self.demand_units["node"] == node
            ].index
            node_demand = self.demand[node_demand_units].sum(axis=1)

            # Calculate total generation for this node by summing powerplant loads
            node_generation_units = self.powerplants_units[
                self.powerplants_units["node"] == node
            ].index
            node_generation = availability_factor_df[node_generation_units].sum(axis=1)

            # Calculate net load (demand - generation)
            net_load_by_node[node] = node_demand - node_generation

        # Step 3: Calculate line-specific congestion severity
        line_congestion_severity = pd.DataFrame(index=self.index)

        for line_id, line_data in self.lines.iterrows():
            node1, node2 = line_data["bus0"], line_data["bus1"]
            s_max_pu = (
                    self.lines.at[line_id, "s_max_pu"]
                    if "s_max_pu" in self.lines.columns
                    and not pd.isna(self.lines.at[line_id, "s_max_pu"])
                    else 1.0
                )
            line_capacity = line_data["s_nom"] * s_max_pu

            # Calculate net load for the line as the sum of net loads from both connected nodes
            line_net_load = net_load_by_node[node1] + net_load_by_node[node2]
            congestion_severity = line_net_load / line_capacity

            # Store the line-specific congestion severity in DataFrame
            line_congestion_severity[f"{line_id}_congestion_severity"] = (
                congestion_severity
            )

        # Step 4: Calculate node-specific congestion signal by aggregating connected lines
        node_congestion_signal = pd.DataFrame(index=self.index)

        for node in self.demand_units["node"].unique():
            # Find all lines connected to this node
            connected_lines = self.lines[
                (self.lines["bus0"] == node) | (self.lines["bus1"] == node)
            ].index

            # Collect all relevant line congestion severities
            relevant_lines = [
                f"{line_id}_congestion_severity" for line_id in connected_lines
            ]

            # Ensure only existing columns are used to avoid KeyError
            relevant_lines = [
                line
                for line in relevant_lines
                if line in line_congestion_severity.columns
            ]

            # Aggregate congestion severities for this node (use max or mean)
            if relevant_lines:
                node_congestion_signal[f"{node}_congestion_severity"] = (
                    line_congestion_severity[relevant_lines].max(axis=1)
                )

        return node_congestion_signal

    def calc_renewable_utilisation(self) -> pd.DataFrame:
        """
        Calculates the renewable utilisation forecast by summing the available renewable generation
        for each node and an overall 'all_nodes' summary.

        Returns:
            pd.DataFrame: A DataFrame with columns for each node, where each column represents the renewable
                        utilisation signal time series for that node and a column for total utilisation across all nodes.
        """
        # Initialize a DataFrame to store renewable utilisation for each node
        renewable_utilisation = pd.DataFrame(index=self.index)

        # calculate power for renewable plants
        renewable_mask = self.powerplants_units["fuel_type"] == "renewable"
        powers = self._calc_power(renewable_mask)
        # Calculate utilisation based on availability and max power for each node
        for node in self.demand_units["node"].unique():
            node_mask = self.powerplants_units["node"] == node
            # get the renewable units for this node
            node_units = self.powerplants_units[node_mask & renewable_mask].index
            utilization = powers[node_units].sum(axis=1)
            renewable_utilisation[f"{node}_renewable_utilisation"] = utilization

        # Calculate the total renewable utilisation across all nodes
        all_nodes_sum = renewable_utilisation.sum(axis=1)
        renewable_utilisation["all_nodes_renewable_utilisation"] = all_nodes_sum

        return renewable_utilisation

    def save_forecasts(self, path: str):
        """
        Saves the forecasts to a csv file located at the specified path.

        Args:
            path (str): The path to save the forecasts to.

        Raises:
            ValueError: If no forecasts are provided, an error message is logged.
        """

        merged_forecasts = pd.DataFrame(self._forecasts)
        merged_forecasts.index = pd.date_range(
            start=self.index[0], end=self.index[-1], freq=self.index.freq
        )
        merged_forecasts.to_csv(f"{path}/forecasts_df.csv", index=True)
