# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import numpy as np
import pandas as pd

from assume.common.fast_pandas import FastIndex, FastSeries


class Forecaster:
    """
    Forecaster represents a base class for forecasters based on existing files,
    random noise, or actual forecast methods. It initializes with the provided index. It includes methods
    to retrieve forecasts for specific columns, availability of units, and prices of fuel types, returning
    the corresponding timeseries as pandas Series.

    Attributes:
        index (pandas.Series): The index of the forecasts.

    Args:
        index (pandas.Series): The index of the forecasts.

    Example:
        >>> forecaster = Forecaster(index=pd.Series([1, 2, 3]))
        >>> forecast = forecaster['temperature']
        >>> print(forecast)

    """

    def __init__(self, index: FastIndex):
        self.index = index

    def __getitem__(self, column: str) -> FastSeries:
        """
        Returns the forecast for a given column.

        Args:
            column (str): The column of the forecast.

        Returns:
            FastSeries: The forecast.

        This method returns the forecast for a given column as a pandas Series based on the provided index.
        """
        return FastSeries(value=0.0, index=self.index)

    def get_availability(self, unit: str) -> FastSeries:
        """
        Returns the availability of a given unit as a pandas Series based on the provided index.

        Args:
            unit (str): The unit.

        Returns:
            FastSeries: The availability of the unit.

        Example:
        >>> forecaster = Forecaster(index=pd.Series([1, 2, 3]))
        >>> availability = forecaster.get_availability('unit_1')
        >>> print(availability)
        """

        return self[f"availability_{unit}"]

    def get_price(self, fuel_type: str) -> FastSeries:
        """
        Returns the price for a given fuel type as a pandas Series or zeros if the type does
        not exist.

        Args:
            fuel_type (str): The fuel type.

        Returns:
            FastSeries: The price of the fuel.

        Example:
            >>> forecaster = Forecaster(index=pd.Series([1, 2, 3]))
            >>> price = forecaster.get_price('lignite')
            >>> print(price)
        """

        return self[f"fuel_price_{fuel_type}"]


class CsvForecaster(Forecaster):
    """
    This class represents a forecaster that provides timeseries for forecasts derived from existing files.

    It initializes with the provided index and configuration data, including power plants, demand units,
    and market configurations. The forecaster also supports optional inputs like DSM (demand-side management) units,
    buses, and lines for more advanced forecasting scenarios.

    Methods are included to retrieve forecasts for specific columns, availability of units,
    and prices of fuel types, returning the corresponding timeseries as pandas Series.

    Notes:
    - Some built-in forecasts are calculated at the beginning of the simulation, such as price forecast and residual load forecast.
    - Price forecast is calculated for energy-only markets using a merit order approach.
    - Residual load forecast is calculated by subtracting the total available power from variable renewable energy power plants from the overall demand forecast. Only power plants containing 'wind' or 'solar' in their technology column are considered VRE power plants.

    Args:
        index (pd.Series): The index of the forecasts.
        powerplants_units (pd.DataFrame): A DataFrame containing information about power plants.
        demand_units (pd.DataFrame): A DataFrame with demand unit data.
        market_configs (dict[str, dict]): Configuration details for the markets.
        buses (pd.DataFrame | None, optional): A DataFrame of buses information. Defaults to None.
        lines (pd.DataFrame | None, optional): A DataFrame of line information. Defaults to None.
        save_path (str, optional): Path where the forecasts should be saved. Defaults to an empty string.
        *args (object): Additional positional arguments.
        **kwargs (object): Additional keyword arguments.

    """

    def __init__(
        self,
        index: pd.Series,
        powerplants_units: pd.DataFrame,
        demand_units: pd.DataFrame,
        market_configs: dict[str, dict],
        exchange_units: pd.DataFrame | None = None,
        buses: pd.DataFrame | None = None,
        lines: pd.DataFrame | None = None,
        save_path: str = "",
        *args,
        **kwargs,
    ):
        super().__init__(index, *args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.powerplants_units = powerplants_units
        self.demand_units = demand_units
        self.market_configs = market_configs
        self.exchange_units = exchange_units
        self.buses = buses
        self.lines = lines

        self.forecasts = pd.DataFrame(index=index)
        self.save_path = save_path

    def __getitem__(self, column: str) -> FastSeries:
        """
        Returns the forecast for a given column.

        If the column does not exist in the forecasts, a Series of zeros is returned. If the column contains "availability", a Series of ones is returned.

        Args:
            column (str): The column of the forecast.

        Returns:
            FastSeries: The forecast for the given column.

        """

        if column not in self.forecasts.keys():
            if "availability" in column:
                self.forecasts[column] = FastSeries(
                    value=1.0, index=self.index, name=column
                )
            else:
                self.forecasts[column] = FastSeries(
                    value=0.0, index=self.index, name=column
                )

        return self.forecasts[column]

    def set_forecast(self, data: pd.DataFrame | pd.Series | None, prefix=""):
        """
        Sets the forecast for a given column.
        If data is a DataFrame, it sets the forecast for each column in the DataFrame. If data is
        a Series, it sets the forecast for the given column. If data is None, no action is taken.

        Args:
            data (pd.DataFrame | pd.Series | None): The forecast data.
            prefix (str): The prefix of the column.

        Example:
            >>> forecaster = CsvForecaster(index=pd.Series([1, 2, 3]))
            >>> forecaster.set_forecast(pd.Series([22, 25, 17], name='temperature'), prefix='location_1_')
            >>> print(forecaster['location_1_temperature'])
        """

        if data is None:
            return
        elif isinstance(data, pd.DataFrame):
            if prefix:
                # set prefix for columns to set
                columns = [prefix + column for column in data.columns]
                data.columns = columns
            if len(data.index) == 1:
                # if we have a single value which should be set for the whole series
                for column in data.columns:
                    self.forecasts[column] = data[column].item()
            else:
                # Add new columns to the existing DataFrame, overwriting any existing columns with the same names
                new_columns = set(data.columns) - set(self.forecasts.columns)
                self.forecasts = pd.concat(
                    [self.forecasts, data[list(new_columns)]], axis=1
                )
        else:
            self.forecasts[prefix + data.name] = data

    def calc_forecast_if_needed(self):
        """
        Calculates the forecasts if they are not already calculated.

        This method calculates price forecast and residual load forecast for available markets,
        and other necessary forecasts if they don't already exist.
        """
        self.add_missing_availability_columns()
        self.calculate_market_forecasts()

        # the following forecasts are only calculated if buses and lines are available
        # and self.demand_units have a node column
        if self.buses is not None and self.lines is not None:
            # check if the demand_units have a node column and
            # if the nodes are available in the buses
            if (
                "node" in self.demand_units.columns
                and self.demand_units["node"].isin(self.buses.index).all()
            ):
                self.add_node_congestion_signals()
                self.add_utilisation_forecasts()
            else:
                self.logger.warning(
                    "Node-specific congestion signals and renewable utilisation forecasts could not be calculated. "
                    "Either 'node' column is missing in demand_units or nodes are not available in buses."
                )

    def add_missing_availability_columns(self):
        """Add missing availability columns to the forecasts."""
        missing_cols = [
            f"availability_{pp}"
            for pp in self.powerplants_units.index
            if f"availability_{pp}" not in self.forecasts.columns
        ]

        if missing_cols:
            # Create a DataFrame with the missing columns initialized to 1
            missing_data = pd.DataFrame(
                1, index=self.forecasts.index, columns=missing_cols
            )
            # Append the missing columns to the forecasts
            self.forecasts = pd.concat([self.forecasts, missing_data], axis=1).copy()

    def calculate_market_forecasts(self):
        """Calculate market-specific price and residual load forecasts."""
        for market_id, config in self.market_configs.items():
            if config["product_type"] != "energy":
                self.logger.warning(
                    f"Price forecast could not be calculated for {market_id}. It can only be calculated for energy-only markets for now."
                )
                continue

            if f"price_{market_id}" not in self.forecasts.columns:
                self.forecasts[f"price_{market_id}"] = (
                    self.calculate_market_price_forecast(market_id=market_id)
                )

            if f"residual_load_{market_id}" not in self.forecasts.columns:
                self.forecasts[f"residual_load_{market_id}"] = (
                    self.calculate_residual_load_forecast(market_id=market_id)
                )

    def add_node_congestion_signals(self):
        """Add node-specific congestion signals to the forecasts."""
        node_congestion_signal_df = self.calculate_node_specific_congestion_forecast()
        for col in node_congestion_signal_df.columns:
            if col not in self.forecasts.columns:
                self.forecasts[col] = node_congestion_signal_df[col]

    def add_utilisation_forecasts(self):
        """Add renewable utilisation forecasts if missing."""
        utilisation_columns = [
            f"{node}_renewable_utilisation"
            for node in self.demand_units["node"].unique()
        ]
        utilisation_columns.append("all_nodes_renewable_utilisation")

        if not all(col in self.forecasts.columns for col in utilisation_columns):
            renewable_utilisation_forecast = (
                self.calculate_renewable_utilisation_forecast()
            )
            for col in renewable_utilisation_forecast.columns:
                if col not in self.forecasts.columns:
                    self.forecasts[col] = renewable_utilisation_forecast[col]

    def get_registered_market_participants(self, market_id):
        """
        Retrieves information about market participants to make accurate price forecasts.

        Currently, the functionality for using different markets and specified registration for
        the price forecast is not implemented, so it returns the power plants as a DataFrame.

        Args:
            market_id (str): The market ID.

        Returns:
            pd.DataFrame: The registered market participants.

        """

        self.logger.warning(
            "Functionality of using the different markets and specified registration for the price forecast is not implemented yet"
        )
        return self.powerplants_units

    def calculate_residual_load_forecast(self, market_id) -> pd.Series:
        """
        This method calculates the residual demand forecast by subtracting the total available power from renewable energy (VRE) power plants from the overall demand forecast for each time step.

        Returns:
            pd.Series: The residual demand forecast.

        Notes:
            1. Selects VRE power plants from the powerplants_units DataFrame based on the technology column (wind or solar).
            2. Creates a DataFrame, vre_feed_in_df, with columns representing VRE power plants and initializes it with zeros.
            3. Calculates the power feed-in for each VRE power plant based on its availability and maximum power.
            4. Calculates the residual demand by subtracting the total VRE power feed-in from the overall demand forecast.
            5. If exchange units are available, imports and exports are subtracted and added to the demand forecast, respectively.

        """

        vre_powerplants_units = self.powerplants_units[
            self.powerplants_units["technology"].str.contains(
                r"\b(?:wind|solar)\b", case=False, na=False
            )
        ].copy()

        vre_feed_in_df = pd.DataFrame(
            index=self.index, columns=vre_powerplants_units.index, data=0.0
        )

        for pp, max_power in vre_powerplants_units["max_power"].items():
            vre_feed_in_df[pp] = self.forecasts[f"availability_{pp}"] * max_power

        demand_units = self.demand_units[
            self.demand_units[f"bidding_{market_id}"].notnull()
        ]
        sum_demand = self.forecasts[demand_units.index].sum(axis=1)

        # get exchanges if exchange_units are available
        if self.exchange_units is not None:
            exchange_units = self.exchange_units[
                self.exchange_units[f"bidding_{market_id}"].notnull()
            ]
            # get sum of imports as name of exchange_unit_import
            import_units = [f"{unit}_import" for unit in exchange_units.index]
            sum_imports = self.forecasts[import_units].sum(axis=1)
            # get sum of exports as name of exchange_unit_export
            export_units = [f"{unit}_export" for unit in exchange_units.index]
            sum_exports = self.forecasts[export_units].sum(axis=1)
            # add imports and exports to the sum_demand
            sum_demand += sum_imports - sum_exports

        res_demand_df = sum_demand - vre_feed_in_df.sum(axis=1)

        return res_demand_df

    def calculate_market_price_forecast(self, market_id):
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

        Notes:
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

        # 3. Get forecast availabilities and reformat the column names to match unit identifiers.
        col_availabilities = self.forecasts.columns[
            self.forecasts.columns.str.startswith("availability")
        ]
        availabilities = self.forecasts[col_availabilities].copy()
        availabilities.columns = col_availabilities.str.replace("availability_", "")

        # 4. Compute available power for each unit at each time step.
        #    Since max_power is a float, this multiplication broadcasts over each column.
        power = self.powerplants_units.max_power * availabilities

        # 5. Process the demand.
        #    Filter demand units with a bidding strategy and sum their forecasts for each time step.
        demand_units = self.demand_units[
            self.demand_units[f"bidding_{market_id}"].notnull()
        ]
        sum_demand = self.forecasts[demand_units.index].sum(axis=1)

        # get exchanges if exchange_units are available
        if self.exchange_units is not None:
            exchange_units = self.exchange_units[
                self.exchange_units[f"bidding_{market_id}"].notnull()
            ]
            # get sum of imports as name of exchange_unit_import
            import_units = [f"{unit}_import" for unit in exchange_units.index]
            sum_imports = self.forecasts[import_units].sum(axis=1)
            # get sum of exports as name of exchange_unit_export
            export_units = [f"{unit}_export" for unit in exchange_units.index]
            sum_exports = self.forecasts[export_units].sum(axis=1)
            # add imports and exports to the sum_demand
            sum_demand += sum_imports - sum_exports

        # 6. Initialize the price forecast series.
        price_forecast = pd.Series(index=self.index, data=0.0)

        # 7. Loop over each time step
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

        Notes:
            1. Determines the fuel price based on the fuel type of the power plant.
            2. Calculates the fuel cost by dividing the fuel price by the efficiency of the power plant.
            3. Calculates the emissions cost based on the CO2 price and emission factor, adjusted by the efficiency of the power plant.
            4. Considers any fixed costs specified for the power plant.
            5. Aggregates the fuel cost, emissions cost, and fixed cost to obtain the marginal cost of the power plant.
        """

        fp_column = f"fuel_price_{pp_series.fuel_type}"
        if fp_column in self.forecasts.columns:
            fuel_price = self.forecasts[fp_column]
        else:
            fuel_price = pd.Series(0.0, index=self.index)

        emission_factor = pp_series["emission_factor"]
        co2_price = self.forecasts["fuel_price_co2"]

        fuel_cost = fuel_price / pp_series["efficiency"]
        emissions_cost = co2_price * emission_factor / pp_series["efficiency"]
        additional_cost = (
            pp_series["additional_cost"] if "additional_cost" in pp_series else 0.0
        )

        marginal_cost = fuel_cost + emissions_cost + additional_cost

        return marginal_cost

    def calculate_node_specific_congestion_forecast(self) -> pd.DataFrame:
        """
        Calculates a collective node-specific congestion signal by aggregating the congestion severity of all
        transmission lines connected to each node, taking into account powerplant load based on availability factors.

        Returns:
            pd.DataFrame: A DataFrame with columns for each node, where each column represents the collective
                          congestion signal time series for that node.
        """
        # Step 1: Calculate powerplant load using availability factors
        availability_factor_df = pd.DataFrame(
            index=self.index, columns=self.powerplants_units.index, data=0.0
        )

        # Calculate load for each powerplant based on availability factor and max power
        for pp, max_power in self.powerplants_units["max_power"].items():
            availability_factor_df[pp] = (
                self.forecasts[f"availability_{pp}"] * max_power
            )

        # Step 2: Calculate net load for each node (demand - generation)
        net_load_by_node = {}

        for node in self.demand_units["node"].unique():
            # Calculate total demand for this node
            node_demand_units = self.demand_units[
                self.demand_units["node"] == node
            ].index
            node_demand = self.forecasts[node_demand_units].sum(axis=1)

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
            line_capacity = line_data["s_nom"]

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

    def calculate_renewable_utilisation_forecast(self) -> pd.DataFrame:
        """
        Calculates the renewable utilisation forecast by summing the available renewable generation
        for each node and an overall 'all_nodes' summary.

        Returns:
            pd.DataFrame: A DataFrame with columns for each node, where each column represents the renewable
                        utilisation signal time series for that node and a column for total utilisation across all nodes.
        """
        # Initialize a DataFrame to store renewable utilisation for each node
        renewable_utilisation = pd.DataFrame(index=self.index)

        # Identify renewable power plants by filtering `powerplants_units` DataFrame
        renewable_plants = self.powerplants_units[
            self.powerplants_units["fuel_type"] == "renewable"
        ]

        # Calculate utilisation based on availability and max power for each renewable plant
        for node in self.demand_units["node"].unique():
            node_renewable_sum = pd.Series(0, index=self.index)

            # Filter renewable plants in this specific node
            node_renewable_plants = renewable_plants[renewable_plants["node"] == node]

            for pp in node_renewable_plants.index:
                max_power = node_renewable_plants.loc[pp, "max_power"]
                availability_col = f"availability_{pp}"

                # Calculate renewable power based on availability and max capacity
                if availability_col in self.forecasts.columns:
                    node_renewable_sum += self.forecasts[availability_col] * max_power

            # Store the node-specific renewable utilisation
            renewable_utilisation[f"{node}_renewable_utilisation"] = node_renewable_sum

        # Calculate the total renewable utilisation across all nodes
        all_nodes_sum = renewable_utilisation.sum(axis=1)
        renewable_utilisation["all_nodes_renewable_utilisation"] = all_nodes_sum

        return renewable_utilisation

    def save_forecasts(self, path=None):
        """
        Saves the forecasts to a csv file located at the specified path.

        Args:
            path (str): The path to save the forecasts to.

        Raises:
            ValueError: If no forecasts are provided, an error message is logged.
        """

        path = path or self.save_path

        merged_forecasts = pd.DataFrame(self.forecasts)
        merged_forecasts.index = pd.date_range(
            start=self.index[0], end=self.index[-1], freq=self.index.freq
        )
        merged_forecasts.to_csv(f"{path}/forecasts_df.csv", index=True)

    def convert_forecasts_to_fast_series(self):
        """
        Converts all forecasts in self.forecasts (DataFrame) into FastSeries and saves them
        in a dictionary. It also converts the self.index to a FastIndex.
        """
        # Convert index to FastIndex
        inferred_freq = pd.infer_freq(self.index)
        if inferred_freq is None:
            raise ValueError("Frequency could not be inferred from the index.")

        self.index = FastIndex(
            start=self.index[0], end=self.index[-1], freq=inferred_freq
        )

        # Initialize an empty dictionary to store FastSeries
        fast_forecasts = {}

        # Convert each column in the forecasts DataFrame to a FastSeries
        for column_name in self.forecasts.columns:
            # Convert each column in self.forecasts to FastSeries
            forecast_series = self.forecasts[column_name]
            fast_forecasts[column_name] = FastSeries.from_pandas_series(forecast_series)

        # Replace the DataFrame with the dictionary of FastSeries
        self.forecasts = fast_forecasts


class RandomCsvForecaster(CsvForecaster):
    """
    This class represents a forecaster that generates forecasts using random noise. It inherits
    from the `CsvForecaster` class and initializes with the provided index, power plants, and
    standard deviation of the noise.

    Attributes:
        index (pandas.Series): The index of the forecasts.
        powerplants_units (pandas.DataFrame): The power plants.
        sigma (float): The standard deviation of the noise.

    Args:
        index (pandas.Series): The index of the forecasts.
        powerplants_units (pandas.DataFrame): The power plants.
        sigma (float): The standard deviation of the noise.

    Example:
        >>> forecaster = RandomCsvForecaster(index=pd.Series([1, 2, 3]))
        >>> forecaster.set_forecast(pd.Series([22, 25, 17], name='temperature'), prefix='location_1_')
        >>> print(forecaster['location_1_temperature'])

    """

    def __init__(
        self,
        index: pd.Series,
        powerplants_units: pd.DataFrame,
        demand_units: pd.DataFrame,
        market_configs: dict = {},
        sigma: float = 0.02,
        *args,
        **kwargs,
    ):
        super().__init__(
            index, powerplants_units, demand_units, market_configs, *args, **kwargs
        )

        self.index = FastIndex(start=index[0], end=index[-1], freq=pd.infer_freq(index))
        self.sigma = sigma

    def __getitem__(self, column: str) -> FastSeries:
        """
        Retrieves forecasted values modified by random noise.

        This method returns the forecast for a given column as a pandas Series modified
        by random noise based on the provided standard deviation of the noise and the existing
        forecasts. If the column does not exist in the forecasts, a Series of zeros is returned.

        Args:
            column (str): The column of the forecast.

        Returns:
            FastSeries: The forecast modified by random noise.

        """

        if column not in self.forecasts.columns:
            return FastSeries(value=0.0, index=self.index)
        noise = np.random.normal(0, self.sigma, len(self.index))
        forecast_data = self.forecasts[column].values * noise
        return FastSeries(index=self.index, value=forecast_data)


class NaiveForecast(Forecaster):
    """
    This class represents a forecaster that generates forecasts using naive methods.

    It inherits from the `Forecaster` class and initializes with the provided index and optional parameters
    for availability, fuel price, CO2 price, demand, and price forecast.
    If the optional parameters are constant values, they are converted to pandas Series with the
    provided index. If the optional parameters are lists, they are converted to pandas Series with
    the provided index and the corresponding values.

    Attributes:
        index (pandas.Series): The index of the forecasts.
        availability (float | list, optional): The availability of the power plants.
        fuel_price (float | list, optional): The fuel price.
        co2_price (float | list, optional): The CO2 price.
        demand (float | list, optional): The demand.
        price_forecast (float | list, optional): The price forecast.

    Args:
        index (pandas.Series): The index of the forecasts.
        availability (float | list, optional): The availability of the power plants.
        fuel_price (float | list, optional): The fuel price.
        co2_price (float | list, optional): The CO2 price.
        demand (float | list, optional): The demand.
        price_forecast (float | list, optional): The price forecast.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Example:
        >>> forecaster = NaiveForecast(demand=100, co2_price=10, fuel_price=10, availability=1, price_forecast=50)
        >>> print(forecaster['demand'])

        >>> forecaster = NaiveForecast(index=pd.Series([1, 2, 3]), demand=[100, 200, 300], co2_price=[10, 20, 30])
        >>> print(forecaster["demand"][2])
    """

    def __init__(
        self,
        index: pd.Series,
        availability: float | list = 1,
        fuel_price: float | list = 10,
        co2_price: float | list = 10,
        demand: float | list = 100,
        price_forecast: float | list = 50,
        *args,
        **kwargs,
    ):
        super().__init__(index)
        self.index = FastIndex(start=index[0], end=index[-1], freq=pd.infer_freq(index))

        # Convert attributes to FastSeries if they are not already Series
        self.fuel_price = FastSeries(
            index=self.index, value=fuel_price, name="fuel_price"
        )
        self.availability = FastSeries(
            index=self.index, value=availability, name="availability"
        )
        self.co2_price = FastSeries(index=self.index, value=co2_price, name="co2_price")
        self.demand = FastSeries(index=self.index, value=demand, name="demand")
        self.price_forecast = FastSeries(
            index=self.index, value=price_forecast, name="price_forecast"
        )

        self.data_dict = {}

        for key, value in kwargs.items():
            self.data_dict[key] = FastSeries(index=self.index, value=value, name=key)

    def __getitem__(self, column: str) -> FastSeries:
        """
        Retrieves forecasted values.

        This method retrieves the forecasted values for a specific column based on the
        provided parameters such as availability, fuel price, CO2 price, demand, and price
        forecast. If the column matches one of the predefined parameters, the corresponding
        value is returned as a pandas Series. If the column does not match, a Series of zeros is returned.

        Args:
            column (str): The column for which forecasted values are requested.

        Returns:
            FastSeries: The forecasted values for the specified column.

        """

        if "availability" in column:
            return self.availability
        elif column == "fuel_price_co2":
            return self.co2_price
        elif "fuel_price" in column:
            return self.fuel_price
        elif "demand" in column:
            return self.demand
        elif column == "price_EOM":
            return self.price_forecast
        elif column in self.data_dict.keys():
            return self.data_dict[column]
        else:
            return FastSeries(value=0.0, index=self.index)
