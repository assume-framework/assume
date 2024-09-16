.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: AGPL-3.0-or-later

===============================
Demand-Side Agents & Management
===============================

This section describes the demand-side units and management strategies available in the toolbox. The module allows for modeling demand-side units like industrial facilities (e.g., Steel Plants) and residential buildings with detailed optimization of electricity consumption and load-shifting capabilities.

Overview
========
The Demand-Side Units (DSU) represent entities that manage energy consumption and contribute to the electricity market. These entities are optimized based on specific objectives, such as minimizing costs or maximizing operational flexibility. The two main types of demand-side agents modeled in this module are **Industrial Agents** (e.g., Steel Plant) and **Residential Agents** (e.g., Buildings).

--------------------------------------------
Modeling Demand-Side Units (DSUs)
--------------------------------------------

The Demand-Side Units (DSUs) are modeled using a component-based approach, where each DSU is a combination of multiple technologies or processes. Each DSU can be either an industrial facility or a residential building. The DSU model is designed to automatically create the process flow by connecting the appropriate components based on user inputs.

For example:
- **Industrial DSU**: The Steel Plant agent connects components like electrolysers, hydrogen storage, DRI plants, and electric arc furnaces to form a complete production process. This is automatically done by providing input components, and the system handles the rest.
- **Residential DSU**: The Building agent connects components like heat pumps, thermal storage, and EV chargers to optimize the energy consumption of a residential building.

### DSU Components:
Each component (e.g., Electrolyser, Heat Pump, DRI Plant) is modeled in detail with specific constraints and parameters, such as:
- Power input/output limits
- Ramp rates
- Efficiency
- Operating costs
- Load flexibility

### Bidding and Optimization:
Once the DSU is defined, it can participate in the electricity market by optimizing its operation to either minimize costs or maximize flexibility. The bidding strategy depends on the agent type and market conditions.

---

--------------------------------------------
1. Industrial Agent: Steel Plant
--------------------------------------------

The **Industrial Agent** represents facilities like steel plants, where key components such as electrolysers, DRI plants, hydrogen storage, and electric arc furnaces are modeled. These components create a process flow that optimizes electricity consumption based on the agent’s objective.

Attributes:
-----------
- **Technology**: Represents components like Electrolyser, DRI Plant, Hydrogen Storage, and Electric Arc Furnace.
- **Node**: Connection point in the energy system.
- **Bidding Strategy**: Defines how the steel plant bids in electricity markets. Example: `NaiveDASteelplantStrategy`.
- **Objective**: Optimization target, such as minimizing operational costs.
- **Flexibility Measure**: Load-shifting to optimize flexibility.

Components:
-----------
- **Electrolyser**: Hydrogen production via electrolysis with constraints on power limits, ramp rates, operating time, and costs.
- **DRI Plant**: Models direct reduced iron production with fuel and power consumption constraints.
- **Electric Arc Furnace (EAF)**: Steel production from DRI with power consumption, ramp rates, and CO2 emissions constraints.

Example Workflow:
-----------------
#. Instantiate the SteelPlant agent with required components (Electrolyser, DRI Plant, etc.).
#. Define objectives, such as minimizing electricity costs.
#. Run the optimization to generate bids for the electricity market.

---

--------------------------------------------
2. Residential Agent: Building
--------------------------------------------

The **Residential Agent** models buildings containing technologies like heat pumps, electric vehicles (EVs), and thermal storage. These components allow the building to participate in the electricity market both as a consumer and prosumer.

Attributes:
-----------
- **Technology**: Components such as heat pumps, boilers, thermal storage, and EV charging.
- **Node**: Connection to the electrical grid.
- **Bidding Strategy**: Example: `NaiveDABuildingStrategy`.
- **Objective**: Optimization target, such as minimizing energy expenses.
- **Flexibility Measure**: Load-shifting to adjust energy usage based on market conditions.

Components:
-----------
- **Heat Pump**: Provides heating/cooling and interacts with thermal storage.
- **EV Charging**: Manages EV battery charging based on availability periods and electricity prices.
- **Thermal Storage**: Buffers thermal energy for flexible heating/cooling operations.

Example Workflow:
-----------------
#. Create a Building agent with components such as a heat pump and thermal storage.
#. Define objectives like minimizing electricity usage.
#. Execute optimization to determine the best dispatch plan based on market prices.

---

--------------------------------------------
3. Strategies for Bidding
--------------------------------------------

Several naive bidding strategies are implemented for managing how agents participate in electricity markets. These strategies define how energy bids are created for different agent types.

- **NaiveDASteelplantStrategy**: Optimizes steel plant operations and generates bids based on marginal costs.
- **NaiveDABuildingStrategy**: Manages the bidding process for residential agents, calculating bids based on optimal energy use and load-shifting.
- **Redispatch Strategies**: Adjust operations based on redispatch market signals, focusing on reducing operational costs or maximizing flexibility.

---

--------------------------------------------
4. DSM Flexibility and Load Shifting
--------------------------------------------

The `dsm_load_shift.py` module integrates load-shifting capabilities into agents. This allows for:
- Defining cost tolerance for flexibility.
- Shifting loads between time steps based on operational flexibility.
- Recalculating operations based on market offers and accepted bids.

Attributes:
-----------
- **Cost Tolerance**: Defines how much additional cost can be tolerated for load-shifting.
- **Load Shift**: Adjusts the total power input based on flexibility constraints and available storage or generation resources.

Example:
--------
For a steel plant, the load-shifting mechanism can balance power input between the electrolyser, DRI plant, and EAF, adjusting production to minimize costs while meeting production targets.
