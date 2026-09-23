<!-- SPDX-FileCopyrightText: ASSUME Developers -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Updated 2019 example

This is a copy of `example_03`, with researched fuel-price inputs and a default
`base_case_2019` covering all 8,760 hourly deliveries in 2019. The original
example is unchanged. Price research was performed on 2026-09-21.

Run from the repository root:

```sh
assume -s example_2019_update -c base_case_2019 -csv examples/outputs
```

## Which original example contains the required data?

| Example | Data year | Generation entries | Storage units | Exchange units | Configuration |
| --- | --- | ---: | ---: | ---: | --- |
| `example_03` | 2019 | 262 | 25 | 9 | Base case, day-ahead, reserve-market and DSM variants |
| `example_03a` | 2019 | 148 | 0 | 9 | Aggregated generation fleet for reinforcement learning |
| `example_03b` | 2021 | 116 | 0 | 11 | Different year, learning scenario |
| `example_03c` | 2019 | 262 | 25 | 9 | Storage learning; configured for March–May, not the full year |

`example_03` contains the requested combination without requiring learning.
All its generation, storage, exchange and demand unit definitions are retained
byte-for-byte. The generation entries comprise 67 hard-coal, 43 lignite,
115 gas, 25 oil, seven nuclear and five aggregated renewable entries. The nine
exchanges have 18 import/export time series. The base case uses the EOM demand
unit; two further demand entries belong to reserve markets.

“Complete” here means the entire supplied example fleet, not every individual
German generating installation. Wind, solar, hydro and biomass are aggregates.
Conventional units do not have individual historical outage profiles in this
example. Exchange volumes remain exogenous historical profiles.

## Fuel-price audit and changes

Fuel columns are interpreted as EUR/MWh of thermal fuel input; `co2` is EUR/tCO2.
ASSUME divides fuel prices by plant efficiency when calculating marginal cost.
Neither fuel prices nor carbon prices are electricity wholesale prices.

Annual means below exclude simulation boundary padding:

| Column | Original mean | Updated mean | Treatment |
| --- | ---: | ---: | --- |
| natural gas | 18.922649 | 15.517321 | Observed monthly NCG gas prices, converted from gross to net calorific-value basis |
| hard coal | 7.434393 | 9.703968 | Published 2019 import-coal annual average, held constant |
| oil | 25.668321 | 33.165459 | Published 2019 heavy-fuel-oil annual average, held constant as an oil proxy |
| co2 | 24.829370 | 24.829370 | Original series retained |
| lignite | 1.800000 | 1.800000 | Original modelling assumption retained; not verified as an observed market price |
| uranium | 0.900000 | 0.900000 | Original modelling assumption retained; not verified as a complete nuclear fuel-cycle cost |
| biomass | 20.700000 | 20.700000 | Original assumption retained; unused by the base fleet's biomass entry |

These replacements establish explicit historical benchmarks. They do **not**
establish that every original price was erroneous: the old file does not identify
its quotation, procurement contract, calorific-value basis or source. In
particular, coal spot prices differ from the import-price benchmark adopted here.

### Gas: monthly historical observations

Source: Fraunhofer ISE [Energy-Charts monthly 2019 prices](https://www.energy-charts.info/charts/price_average/data/de/month_euro_mwh_2019.json),
series `Gas (NCG, THE)` (NCG is the relevant 2019 market).
The twelve extracted values are in `price_sources/gas_monthly_2019.csv`.
The downloaded JSON's SHA-256 and source URL are in `price_sources/provenance.json`.

The calendar-hour-weighted source mean is 14.002830 EUR/MWh. As a cross-check,
the [Bundesnetzagentur/Bundeskartellamt Monitoring Report 2020, printed p. 405](https://data.bundesnetzagentur.de/Bundesnetzagentur/SharedDocs/Downloads/EN/Areas/ElectricityGas/CollectionCompanySpecificData/Monitoring/monitoringreport2020.pdf)
reports a 2019 NCG EGSI trading-day average of 14.18 EUR/MWh. Different averaging
conventions need not produce identical annual values.

This scenario adopts net/lower calorific value for fuel input. Using the German
energy-balance conversion factor from [AGEB's 2019 calorific-value table](https://ag-energiebilanzen.de/wp-content/uploads/2022/04/Heizwerte_2005-2021.pdf),
`MWh_LHV = 0.9024 × MWh_HHV`, so `price_LHV = price_HHV / 0.9024`.
This is an explicit scenario convention; the original unit file does not document
the calorific-value basis of each individual efficiency.

Each monthly observation is held constant over its calendar month. There is no
invented daily or quarter-hourly variation. Monthly realised means are ex-post
simulation inputs, not prices that would have been known before every auction.

### Coal and oil: annual benchmark proxies

Source: [VDKi Annual Report 2020, printed p. 18](https://english.kohlenimporteure.de/files/user_upload/jahresberichte_en/VDKi_Annual_Report_2020.pdf).
Its 2019 figures are 79 EUR/tce for imported steam coal and 270 EUR/tce for
heavy fuel oil. Values are recorded in `price_sources/annual_benchmarks_2019.csv`.
Conversion: `1 tce = 29.3076 GJ = 8.141 MWh`; divide EUR/tce by 8.141.
A tonne of coal equivalent is an energy unit, not a physical tonne of fuel.
The figures are used as published in that report, rather than represented as
later revisions or as a recovered daily price history.

The copy uses constant annual prices because a matching daily or monthly series
was not established. Gas uses a wholesale benchmark; coal uses an import
benchmark and oil a power-sector heavy-fuel-oil benchmark. These are not a
uniform delivered-to-plant procurement dataset. The original `oil` category does
not distinguish oil grades: heavy fuel oil is a documented proxy, not a verified
fuel choice or purchase price for each of the 25 oil units. Light-oil turbines
would require a separate fuel category and price series.

### Retained inputs and limitations

The original carbon-price mean is close to the 24.86 EUR/tCO2 2019 benchmark in
the empirical study [What caused 2019's drop in German carbon emissions?](https://pmc.ncbi.nlm.nih.gov/articles/PMC7716793/).
This supports retaining it, but does not independently verify every timestamp.
Carbon is applied separately in the plant marginal-cost equation.

Lignite costs depend on the mine/plant combination; a uranium commodity quote
cannot directly replace a reactor fuel cost per thermal MWh. No matching 2019
plant-level replacement was established, so these constants remain explicitly
unverified model assumptions. The biomass aggregate has `fuel_type=renewable`
and zero additional cost; changing the `biomass` price column alone would not
affect its bids. Its dispatch representation has not been changed.

## Full-year delivery configuration

The original base case starts on January 1 with first delivery 24 hours later
and ends on December 31 at midnight. It therefore does not schedule all 8,760
2019 delivery hours despite having full-year input data.

The updated default starts on **2018-12-31 at 11:00** for the first auction,
closes each daily auction one hour later, and delivers 24 hourly products from
midnight on the following day (`first_delivery: 13h`). The simulation ends on
**2020-01-01 at 00:00**. This produces exactly January 1 00:00 through December 31
23:00 delivery starts. Times use the original timezone-naive, fixed-hour
convention; no new daylight-saving-time conversion is applied.

Time-series files include 52 quarter-hour initialization rows before 2019 and
four terminal rows after it, repeating their nearest available endpoint.
These 56 rows are synthetic boundary padding, **not historical observations**.
There are no market deliveries outside 2019 in the default case. Filter analyses
to 2019. Every 2019 row of demand, availability, exchange and supplied forecast
data remains unchanged. EOM forecasts are recalculated from the new fuel inputs
(`use_forecasts_df: false`).

The other three inherited study cases retain their original schedules and are
not all full-year cases. They share the updated price file. Use `base_case_2019`
for the full-year configuration described here.

## Day-ahead and redispatch sequence

The default case now clears the German/Luxembourg day-ahead market as one bidding
zone at 12:00, followed by a redispatch auction at 13:00 for the same 24 delivery
hours. Redispatch uses pay-as-bid activation at marginal cost, a 10,000 EUR/MWh
backup penalty, and records line flows. Conventional plants provide upward and
downward flexibility. Storage and border exchanges retain their cleared
day-ahead schedules as fixed nodal injections; this prevents omitted schedules
from appearing as artificial backup redispatch.

`buses.csv` and `lines.csv` form a four-node reduced network (north, east, west,
south). Demand is split 16%, 16%, 38%, and 30%, respectively, while preserving
the national demand total at every timestamp. Plant assignment uses documented
name/operator rules in `build_redispatch_inputs.py`; border exchanges are placed
at the corresponding edge node. The five corridor ratings are explicit scenario
assumptions designed for a tractable redispatch example. They are **not a
reconstruction or validation of the physical 2019 German transmission grid**.
Redispatch volumes and costs therefore require calibration against a validated
network before they can be interpreted historically.

The day-ahead price remains a single-zone market price and is unaffected by the
chosen network until redispatch. Its historical realism still depends on plant
availability, bidding assumptions, renewable support behavior, and coupled
neighboring-market prices. The source data provide aggregate renewable profiles
and do not provide unit-level conventional outages, so this case is suitable for
model development and sensitivity analysis rather than price backtesting without
further calibration.

## Hybrid forward-price signal

`base_case_2019` enables the adaptive merit-order forecast for storage
operators. It starts with the physical merit-order price, learns the residual
against cleared EOM prices, and supplies the corrected forward EOM price to the
storage energy and redispatch strategies before they bid. The nonlinear model
trains after 504 cleared hourly products; earlier forecasts retain the
merit-order point estimate and use empirical uncertainty. Other unit types can
be selected through `adaptive_merit_order.unit_types` in `config.yaml`.

## Reproduction and validation

```sh
.venv/bin/python examples/inputs/example_2019_update/rebuild_prices.py
.venv/bin/python examples/inputs/example_2019_update/build_redispatch_inputs.py
```

The offline script regenerates the price file from the recorded benchmarks and
the unchanged `example_03` retained columns. Original-file hashes are recorded in
the provenance file. No new data download is required.

Checks completed:

- All original source-file hashes recorded for price reconstruction still match.
- All five time-series files have continuous, finite quarter-hour data, including
  all 35,040 timestamps in 2019 and the documented boundary padding.
- Full default-scenario initialization loads the complete original fleet, with
  national demand divided among four network nodes.
- Enumerating the actual market schedule yields exactly 8,760 delivery products,
  with no gaps or duplicate hours, ending at 2020-01-01 00:00.
- A two-day beginning-of-year smoke run completes both the day-ahead and
  redispatch auctions with the full fleet, storage, and exchanges.
- A separate final-day smoke run clears all 24 December 31 products, balances
  accepted supply and demand, and completes the storage terminal-state update.
- The price-rebuild script passes Ruff checks.

The full-year simulation has not been executed. These checks validate input
integrity, loading and boundary execution, not agreement of simulated generation
or electricity prices with historical outcomes.
