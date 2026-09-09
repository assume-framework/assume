# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the cement plant DSM unit.

Covers the kiln line (preheater → calciner → kiln), the firing modes, waste heat
recovery, the optional components (electrolyser, electric thermal storage, cement mill),
the flexibility measures and rolling-horizon operation.
"""

import copy

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.common.fast_pandas import FastSeries
from assume.common.forecaster import CementForecaster
from assume.strategies.naive_strategies import (
    DsmEnergyNaiveRedispatchStrategy,
    DsmEnergyOptimizationStrategy,
)
from assume.units.cement_plant import CementPlant

# Base configuration of the kiln line used throughout the tests. Thermal demands are
# given per tonne of throughput, auxiliary electricity per tonne as well.
RAW_MEAL_RATIO = 1.55
PREHEATER_HEAT = 0.3  # MWh_th per t raw meal
CALCINER_HEAT = 0.7  # MWh_th per t clinker
KILN_HEAT = 1.7  # MWh_th per t clinker
AUX_ELECTRICITY = 0.0015  # MWh_el per t throughput
WASTE_HEAT = 0.22  # MWh_th per t clinker
WASTE_HEAT_UTILISATION = 0.9

# Auxiliary electricity per tonne of clinker across the three stages
AUX_PER_TONNE_CLINKER = AUX_ELECTRICITY * (2 + RAW_MEAL_RATIO)


@pytest.fixture
def cement_components():
    return {
        "preheater": {
            "max_heat_out": 10,
            "specific_heat_demand": PREHEATER_HEAT,
            "specific_electricity_aux": AUX_ELECTRICITY,
            "fuel_type": "fossil",
            "eta_fossil": 0.9,
            "fossil_ng_share": 1,
            "ramp_up": 10,
            "ramp_down": 10,
        },
        "calciner": {
            "max_heat_out": 5,
            "specific_heat_demand": CALCINER_HEAT,
            "specific_electricity_aux": AUX_ELECTRICITY,
            "fuel_type": "fossil",
            "eta_fossil": 0.9,
            "fossil_ng_share": 0.7,
            "calcination_emission_factor": 0.525,
            "ramp_up": 5,
            "ramp_down": 5,
        },
        "kiln": {
            "max_heat_out": 20,
            "specific_heat_demand": KILN_HEAT,
            "specific_electricity_aux": AUX_ELECTRICITY,
            "fuel_type": "fossil",
            "eta_fossil": 0.9,
            "fossil_ng_share": 1,
            "ramp_up": 20,
            "ramp_down": 20,
        },
    }


def make_forecaster(n=24, prices=None, **kwargs):
    index = pd.date_range("2023-01-01", periods=n, freq="h")
    return CementForecaster(
        index,
        electricity_price=prices if prices is not None else [50] * n,
        fuel_prices={
            "natural_gas": [30] * n,
            "coal": [15] * n,
            "hydrogen": [80] * n,
            "co2": [90] * n,
        },
        renewable_utilisation_signal=[0.1 * i for i in range(n)],
        **kwargs,
    )


def make_plant(
    components,
    forecaster,
    flexibility_measure="cost_based_load_shift",
    demand=100,
    setup=True,
    plant_id=None,
    **kwargs,
):
    plant = CementPlant(
        id=plant_id or f"test_cement_plant_{flexibility_measure}",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure=flexibility_measure,
        bidding_strategies={
            "EOM": DsmEnergyOptimizationStrategy(),
            "RD": DsmEnergyNaiveRedispatchStrategy(),
        },
        node="south",
        components=components,
        forecaster=forecaster,
        demand=demand,
        technology="cement_plant",
        cost_tolerance=10,
        congestion_threshold=0.8,
        peak_load_cap=95,
        **kwargs,
    )
    if setup:
        plant.setup_model()
    return plant


def solved_instance(plant):
    """Solve the cost-optimal model and return the solved instance."""
    instance = plant.model.create_instance()
    instance = plant.switch_to_opt(instance)
    plant.solver.solve(instance, tee=False)
    return instance


def interior_runs(status_values):
    """Maximal runs of equal consecutive values, excluding runs touching either edge.

    Used to check minimum up/down time: a run touching index 0 may have started before
    the horizon with no ``start_up`` event to anchor it, and a run touching the last
    index may simply be cut off by the end of the horizon rather than genuinely
    violating a minimum length. Both are legitimate exclusions, not enforcement gaps -
    excluding them keeps the check meaningful for the interior of the horizon, where a
    run is unambiguously bounded by real transitions on both sides.
    """
    values = list(status_values)
    n = len(values)
    runs = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[j + 1] == values[i]:
            j += 1
        if i > 0 and j < n - 1:
            runs.append((values[i], i, j))
        i = j + 1
    return runs


# ---------------------------------------------------------------------------
# 1. Model structure
# ---------------------------------------------------------------------------


def test_initialize_components(cement_components):
    plant = make_plant(cement_components, make_forecaster())
    for block in ("preheater", "calciner", "kiln"):
        assert block in plant.model.dsm_blocks.keys()


def test_unknown_component_is_rejected(cement_components):
    components = copy.deepcopy(cement_components)
    components["eaf"] = {"max_power": 10}
    with pytest.raises(ValueError, match="not a valid component"):
        make_plant(components, make_forecaster(), setup=False)


def test_electrolyser_requires_a_burner():
    with pytest.raises(ValueError, match="needs a calciner or a kiln"):
        make_plant(
            {"electrolyser": {"max_power": 10, "efficiency": 0.7}},
            make_forecaster(),
            demand=0,
            setup=False,
        )


def test_thermal_storage_requires_a_calciner(cement_components):
    components = copy.deepcopy(cement_components)
    components.pop("calciner")
    components["thermal_storage"] = {
        "capacity": 20,
        "max_power_charge": 10,
        "max_power_discharge": 20,
        "ramp_up": 10,
        "ramp_down": 10,
        "initial_soc": 0,
        "storage_type": "short-term_with_generator",
    }
    with pytest.raises(ValueError, match="calciner is required"):
        make_plant(components, make_forecaster(), setup=False)


def test_demand_without_any_output_stage_still_raises():
    """A demand with no kiln line and no mill at all has nothing to produce it."""
    with pytest.raises(ValueError, match="requires a calciner"):
        make_plant(
            {
                "preheater": {
                    "max_heat_out": 10,
                    "specific_heat_demand": PREHEATER_HEAT,
                    "fuel_type": "fossil",
                }
            },
            make_forecaster(),
            demand=100,
            setup=False,
        )


def test_unsupported_fuel_type_is_rejected(cement_components):
    components = copy.deepcopy(cement_components)
    components["kiln"]["fuel_type"] = "biomass"
    with pytest.raises(ValueError, match="Unsupported fuel_type"):
        make_plant(components, make_forecaster())


# ---------------------------------------------------------------------------
# 2. Process chain and cost-optimal operation
# ---------------------------------------------------------------------------


def test_process_chain_mass_and_energy_balances(cement_components):
    """Raw meal, clinker, waste heat and fuel demands are consistent per time step."""
    demand = 20
    plant = make_plant(
        cement_components, make_forecaster(n=8), demand=demand, plant_id="test_chain"
    )
    instance = solved_instance(plant)

    total_clinker = 0.0
    for t in instance.time_steps:
        preheater = instance.dsm_blocks["preheater"]
        calciner = instance.dsm_blocks["calciner"]
        kiln = instance.dsm_blocks["kiln"]

        clinker = pyo.value(kiln.clinker_out[t])
        total_clinker += clinker

        # Kiln finishes exactly what the calciner calcined
        assert clinker == pytest.approx(pyo.value(calciner.clinker_out[t]), abs=1e-6)
        # Raw meal follows the clinker rate through the raw meal ratio
        assert pyo.value(preheater.raw_meal_out[t]) == pytest.approx(
            clinker * RAW_MEAL_RATIO, abs=1e-6
        )
        # Heat demands of the stages
        assert pyo.value(kiln.heat_out[t]) == pytest.approx(
            clinker * KILN_HEAT, abs=1e-6
        )
        assert pyo.value(calciner.effective_heat_in[t]) == pytest.approx(
            clinker * CALCINER_HEAT, abs=1e-6
        )
        # Waste heat recovered from the kiln lowers the preheater's own firing
        recovered = clinker * WASTE_HEAT * WASTE_HEAT_UTILISATION
        assert pyo.value(preheater.external_heat_in[t]) == pytest.approx(
            recovered, abs=1e-6
        )
        assert pyo.value(preheater.heat_out[t]) == pytest.approx(
            clinker * RAW_MEAL_RATIO * PREHEATER_HEAT, abs=1e-6
        )
        # Fossil fuel covers the heat the stage generates itself
        assert pyo.value(preheater.fossil_in[t]) * 0.9 == pytest.approx(
            pyo.value(preheater.heat_out[t]) - recovered, abs=1e-6
        )
        # Fossil mix split of the calciner
        fossil = pyo.value(calciner.fossil_in[t])
        assert pyo.value(calciner.natural_gas_in[t]) == pytest.approx(
            0.7 * fossil, abs=1e-6
        )
        assert pyo.value(calciner.coal_in[t]) == pytest.approx(0.3 * fossil, abs=1e-6)

    assert total_clinker == pytest.approx(demand, rel=1e-4)


def test_preheater_feeds_kiln_directly_without_a_calciner(cement_components):
    """A single-stage line (preheater + kiln, no calciner) still links raw meal to clinker.

    Without a dedicated precalciner the kiln itself performs the calcination reaction,
    so the preheater must feed the kiln's raw meal demand directly rather than being
    left as an unconstrained (and therefore idle) stage.
    """
    demand = 20
    components = {
        "preheater": copy.deepcopy(cement_components["preheater"]),
        "kiln": copy.deepcopy(cement_components["kiln"]),
    }
    plant = make_plant(
        components, make_forecaster(n=6), demand=demand, plant_id="test_two_stage"
    )
    instance = solved_instance(plant)

    total_clinker = 0.0
    for t in instance.time_steps:
        preheater = instance.dsm_blocks["preheater"]
        kiln = instance.dsm_blocks["kiln"]
        clinker = pyo.value(kiln.clinker_out[t])
        total_clinker += clinker

        assert pyo.value(preheater.raw_meal_out[t]) == pytest.approx(
            clinker * RAW_MEAL_RATIO, abs=1e-6
        )
        # The kiln is present here too, so waste heat recovery still applies.
        recovered = clinker * WASTE_HEAT * WASTE_HEAT_UTILISATION
        assert pyo.value(preheater.external_heat_in[t]) == pytest.approx(
            recovered, abs=1e-6
        )

    assert total_clinker == pytest.approx(demand, rel=1e-4)


def test_fossil_line_load_is_auxiliary_electricity_only(cement_components):
    """With fossil firing the plant load is the auxiliary electricity of the stages."""
    demand = 20
    plant = make_plant(
        cement_components, make_forecaster(n=8), demand=demand, plant_id="test_aux_load"
    )
    plant.determine_optimal_operation_without_flex()

    assert isinstance(plant.opt_power_requirement, FastSeries)
    assert sum(plant.opt_power_requirement.data) == pytest.approx(
        demand * AUX_PER_TONNE_CLINKER, rel=1e-4
    )


def test_process_emissions_are_priced(cement_components):
    """Calcination CO2 is charged on top of the energy emissions."""
    demand = 20
    plant = make_plant(
        cement_components, make_forecaster(n=4), demand=demand, plant_id="test_co2"
    )
    instance = solved_instance(plant)

    process_co2 = sum(
        pyo.value(instance.dsm_blocks["calciner"].co2_process[t])
        for t in instance.time_steps
    )
    assert process_co2 == pytest.approx(demand * 0.525, rel=1e-4)


def test_electric_calciner_shifts_load_to_cheap_hours(cement_components):
    """An electrically heated calciner concentrates its load in the cheap hours."""
    n = 8
    components = copy.deepcopy(cement_components)
    components["calciner"]["fuel_type"] = "electricity"
    components["calciner"]["eta_electric"] = 0.95
    prices = [200.0] * 4 + [10.0] * 4
    plant = make_plant(
        components,
        make_forecaster(n=n, prices=prices),
        demand=20,
        plant_id="test_electric_calciner",
    )
    plant.determine_optimal_operation_without_flex()

    opt_power = plant.opt_power_requirement
    assert sum(opt_power.data[4:]) > sum(opt_power.data[:4])

    instance = solved_instance(plant)
    for t in instance.time_steps:
        calciner = instance.dsm_blocks["calciner"]
        # The electric heating path covers the whole calciner heat demand
        assert pyo.value(calciner.power_in[t]) * 0.95 == pytest.approx(
            pyo.value(calciner.heat_out[t]), abs=1e-6
        )
        assert pyo.value(calciner.fossil_in[t]) == pytest.approx(0.0, abs=1e-6)


def test_hydrogen_fired_kiln_with_electrolyser(cement_components):
    """Hydrogen from the electrolyser covers the burner demand it is routed to."""
    n = 4
    components = copy.deepcopy(cement_components)
    components["kiln"]["fuel_type"] = "hydrogen"
    components["electrolyser"] = {
        "max_power": 200,
        "min_power": 0,
        "efficiency": 0.7,
        "ramp_up": 200,
        "ramp_down": 200,
    }
    plant = make_plant(
        components, make_forecaster(n=n), demand=10, plant_id="test_h2_kiln"
    )
    instance = solved_instance(plant)

    for t in instance.time_steps:
        kiln = instance.dsm_blocks["kiln"]
        electrolyser = instance.dsm_blocks["electrolyser"]
        assert pyo.value(kiln.hydrogen_in[t]) == pytest.approx(
            pyo.value(instance.h2_to_calciner[t] * 0 + instance.h2_to_kiln[t]), abs=1e-6
        )
        # All hydrogen produced is either routed to a burner or explicitly unused
        assert pyo.value(electrolyser.hydrogen_out[t]) == pytest.approx(
            pyo.value(
                instance.h2_to_calciner[t]
                + instance.h2_to_kiln[t]
                + instance.h2_unutilised[t]
            ),
            abs=1e-6,
        )
        # Hydrogen fires the kiln, so it draws no fossil fuel
        assert pyo.value(kiln.fossil_in[t]) == pytest.approx(0.0, abs=1e-6)


def test_electric_thermal_storage_displaces_burner_heat(cement_components):
    """The E-TES charges in cheap hours and covers calciner heat in expensive ones."""
    n = 6
    components = copy.deepcopy(cement_components)
    components["thermal_storage"] = {
        "capacity": 20,
        "max_power_charge": 10,
        "max_power_discharge": 20,
        "ramp_up": 10,
        "ramp_down": 10,
        "initial_soc": 0,
        "storage_type": "short-term_with_generator",
        "eta_electric": 0.97,
    }
    # Cheap power in the first half, very expensive gas throughout
    prices = [1.0, 1.0, 1.0, 300.0, 300.0, 300.0]
    forecaster = CementForecaster(
        pd.date_range("2023-01-01", periods=n, freq="h"),
        electricity_price=prices,
        fuel_prices={
            "natural_gas": [500] * n,
            "coal": [500] * n,
            "hydrogen": [500] * n,
            "co2": [10] * n,
        },
    )
    plant = make_plant(
        components, forecaster, demand=6, plant_id="test_etes", setup=True
    )
    instance = solved_instance(plant)

    storage = instance.dsm_blocks["thermal_storage"]
    charge_power = [pyo.value(storage.power_in[t]) for t in instance.time_steps]
    discharge = [pyo.value(storage.discharge[t]) for t in instance.time_steps]

    # The heater ran, and it ran in the cheap hours
    assert sum(charge_power) > 0
    assert sum(charge_power[:3]) > sum(charge_power[3:])
    # Discharged heat reached the calciner
    for t in instance.time_steps:
        assert pyo.value(
            instance.dsm_blocks["calciner"].effective_heat_in[t]
        ) == pytest.approx(
            pyo.value(instance.dsm_blocks["calciner"].heat_out[t]) + discharge[t],
            abs=1e-6,
        )
    # Charging is the electric heat the power draw produced
    for t in instance.time_steps:
        assert pyo.value(storage.charge[t]) == pytest.approx(
            charge_power[t] * 0.97, abs=1e-6
        )


def test_cement_mill_grinds_the_clinker(cement_components):
    """The cement mill load follows the clinker the line produces."""
    n = 4
    demand = 20
    components = copy.deepcopy(cement_components)
    components["cement_mill"] = {
        "max_power": 25,
        "min_power": 0,
        "efficiency": 1.0,
        "specific_electricity_consumption": 0.04,
        "ramp_up": 25,
        "ramp_down": 25,
    }
    plant = make_plant(
        components, make_forecaster(n=n), demand=demand, plant_id="test_cement_mill"
    )
    instance = solved_instance(plant)

    for t in instance.time_steps:
        assert pyo.value(
            instance.dsm_blocks["cement_mill"].material_input[t]
        ) == pytest.approx(pyo.value(instance.clinker_rate[t]), abs=1e-6)

    total_power = sum(
        pyo.value(instance.total_power_input[t]) for t in instance.time_steps
    )
    expected = demand * (AUX_PER_TONNE_CLINKER + 0.04)
    assert total_power == pytest.approx(expected, rel=1e-4)


def test_per_timestep_demand_constraint_is_enforced(cement_components):
    """Hourly minimum clinker production from the forecaster is respected."""
    n = 4
    per_hour_min = [2.0, 4.0, 6.0, 1.0]
    forecaster = make_forecaster(
        n=n, prices=[500.0, 500.0, 10.0, 10.0], clinker_demand=per_hour_min
    )
    plant = make_plant(
        cement_components, forecaster, demand=0, plant_id="test_per_timestep"
    )

    # The global equality constraint must give way to the per-hour minimums
    assert not hasattr(plant.model, "clinker_output_association_constraint")
    assert hasattr(plant.model, "clinker_demand_per_timestep_constraint")

    instance = solved_instance(plant)
    for t, minimum in enumerate(per_hour_min):
        assert pyo.value(instance.clinker_rate[t]) >= minimum - 1e-6


def test_availability_profile_blocks_a_stage(cement_components):
    """A zero availability of the kiln forbids clinker burning in those hours."""
    n = 8
    availability = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    forecaster = make_forecaster(n=n, availability_profiles={"kiln": availability})
    plant = make_plant(
        cement_components, forecaster, demand=20, plant_id="test_availability"
    )
    instance = solved_instance(plant)

    for t, available in enumerate(availability):
        if available == 0:
            assert pyo.value(instance.dsm_blocks["kiln"].heat_out[t]) == pytest.approx(
                0.0, abs=1e-6
            )


def test_minimum_heat_output_adds_unit_commitment(cement_components):
    """A minimum thermal output turns a stage into a unit-commitment block."""
    components = copy.deepcopy(cement_components)
    components["kiln"]["min_heat_out"] = 4
    plant = make_plant(
        components, make_forecaster(n=6), demand=20, plant_id="test_min_heat"
    )
    assert hasattr(plant.model.dsm_blocks["kiln"], "operational_status")

    instance = solved_instance(plant)
    for t in instance.time_steps:
        heat = pyo.value(instance.dsm_blocks["kiln"].heat_out[t])
        assert heat <= 1e-6 or heat >= 4 - 1e-6


# ---------------------------------------------------------------------------
# 2b. Standalone grinding mill operation
# ---------------------------------------------------------------------------


def test_raw_material_mill_is_a_valid_component(cement_components):
    """raw_material_mill is accepted as a component, not rejected as unknown."""
    components = copy.deepcopy(cement_components)
    components["raw_material_mill"] = {
        "max_power": 15,
        "min_power": 0,
        "efficiency": 1.0,
        "specific_electricity_consumption": 0.02,
        "ramp_up": 15,
        "ramp_down": 15,
    }
    plant = make_plant(
        components, make_forecaster(n=4), demand=20, plant_id="test_raw_mill_valid"
    )
    assert "raw_material_mill" in plant.model.dsm_blocks.keys()


def test_standalone_cement_mill_meets_its_own_demand():
    """A cement mill with no kiln line grinds directly to meet the declared demand."""
    n = 4
    demand = 20
    components = {
        "cement_mill": {
            "max_power": 25,
            "min_power": 0,
            "efficiency": 1.0,
            "specific_electricity_consumption": 0.04,
            "ramp_up": 25,
            "ramp_down": 25,
        }
    }
    plant = make_plant(
        components,
        make_forecaster(n=n),
        demand=demand,
        plant_id="test_standalone_cement_mill",
    )
    instance = solved_instance(plant)

    total_output = sum(
        pyo.value(instance.dsm_blocks["cement_mill"].material_output[t])
        for t in instance.time_steps
    )
    assert total_output == pytest.approx(demand, rel=1e-4)
    for t in instance.time_steps:
        assert pyo.value(instance.clinker_rate[t]) == pytest.approx(
            pyo.value(instance.dsm_blocks["cement_mill"].material_output[t]), abs=1e-6
        )


def test_standalone_raw_material_mill_meets_its_own_demand():
    """A raw material mill with no kiln line grinds directly to meet the declared demand."""
    n = 4
    demand = 30
    components = {
        "raw_material_mill": {
            "max_power": 15,
            "min_power": 0,
            "efficiency": 1.0,
            "specific_electricity_consumption": 0.02,
            "ramp_up": 15,
            "ramp_down": 15,
        }
    }
    plant = make_plant(
        components,
        make_forecaster(n=n),
        demand=demand,
        plant_id="test_standalone_raw_mill",
    )
    instance = solved_instance(plant)

    total_output = sum(
        pyo.value(instance.dsm_blocks["raw_material_mill"].material_output[t])
        for t in instance.time_steps
    )
    assert total_output == pytest.approx(demand, rel=1e-4)


def test_kiln_line_demand_ignores_a_present_but_unconnected_mill(cement_components):
    """Demand still targets clinker when a kiln line exists, even with a mill present.

    The mill is not wired into the process chain, so it stays idle - nothing requires
    or rewards running it.
    """
    n = 4
    demand = 20
    components = copy.deepcopy(cement_components)
    components["raw_material_mill"] = {
        "max_power": 15,
        "min_power": 0,
        "efficiency": 1.0,
        "specific_electricity_consumption": 0.02,
        "ramp_up": 15,
        "ramp_down": 15,
    }
    plant = make_plant(
        components, make_forecaster(n=n), demand=demand, plant_id="test_kiln_plus_mill"
    )
    instance = solved_instance(plant)

    for t in instance.time_steps:
        assert pyo.value(instance.clinker_rate[t]) == pytest.approx(
            pyo.value(instance.dsm_blocks["kiln"].clinker_out[t]), abs=1e-6
        )
        assert pyo.value(
            instance.dsm_blocks["raw_material_mill"].material_output[t]
        ) == pytest.approx(0.0, abs=1e-6)

    total_clinker = sum(
        pyo.value(instance.clinker_rate[t]) for t in instance.time_steps
    )
    assert total_clinker == pytest.approx(demand, rel=1e-4)


def test_cement_mill_takes_priority_over_raw_material_mill_when_standalone():
    """With both mills and no kiln line, demand targets the cement mill."""
    n = 4
    demand = 20
    components = {
        "cement_mill": {
            "max_power": 25,
            "min_power": 0,
            "efficiency": 1.0,
            "specific_electricity_consumption": 0.04,
            "ramp_up": 25,
            "ramp_down": 25,
        },
        "raw_material_mill": {
            "max_power": 15,
            "min_power": 0,
            "efficiency": 1.0,
            "specific_electricity_consumption": 0.02,
            "ramp_up": 15,
            "ramp_down": 15,
        },
    }
    plant = make_plant(
        components, make_forecaster(n=n), demand=demand, plant_id="test_mill_priority"
    )
    assert plant._standalone_output_stage == "cement_mill"
    instance = solved_instance(plant)

    total_cement = sum(
        pyo.value(instance.dsm_blocks["cement_mill"].material_output[t])
        for t in instance.time_steps
    )
    assert total_cement == pytest.approx(demand, rel=1e-4)
    for t in instance.time_steps:
        assert pyo.value(
            instance.dsm_blocks["raw_material_mill"].material_output[t]
        ) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2c. Exact / rigorous verification
#
# These tests avoid loose aggregate checks in favour of two stricter techniques:
#   (a) forcing a *unique* optimum by construction (e.g. demand exactly equal to the
#       maximum output achievable under a binding ramp limit), so the solved schedule
#       can be compared to a hand-computed value at every time step, not just in total;
#   (b) reconstructing a constraint's right-hand side from the solved values of its own
#       inputs (efficiencies, prices, ratios) and checking it against the solved
#       left-hand side exactly, which catches wiring bugs (wrong sign, wrong ratio,
#       double-counted term) that a loose "some flexibility happened" check would miss.
# ---------------------------------------------------------------------------


def test_ramp_up_exactly_determines_a_forced_startup_trajectory():
    """A demand equal to the ramp-limited maximum pins down a unique optimal schedule.

    Standalone kiln, electric, eta_electric=1 and specific_heat_demand=1 so clinker_out
    equals heat_out exactly. With ramp_up=5 and max_heat_out=20, the largest total
    achievable over 6 hours starting from zero is 5+10+15+20+20+20=90 - and that total
    is achievable in exactly one way. Setting demand=90 therefore leaves the solver no
    freedom at all: the hourly schedule must match that trajectory exactly, regardless
    of price (there is only one feasible point at the demand-equality boundary).
    """
    n = 6
    components = {
        "kiln": {
            "max_heat_out": 20,
            "specific_heat_demand": 1.0,
            "fuel_type": "electricity",
            "eta_electric": 1.0,
            "ramp_up": 5,
            "ramp_down": 20,
            "initial_operational_status": 1,
        }
    }
    plant = make_plant(
        components,
        make_forecaster(n=n, prices=[80, 10, 80, 10, 80, 10]),
        demand=90,
        plant_id="test_ramp_up_exact",
    )
    instance = solved_instance(plant)

    expected = [5, 10, 15, 20, 20, 20]
    actual = [pyo.value(instance.dsm_blocks["kiln"].heat_out[t]) for t in range(n)]
    assert actual == pytest.approx(expected, abs=1e-5)
    clinker = [pyo.value(instance.clinker_rate[t]) for t in range(n)]
    assert clinker == pytest.approx(expected, abs=1e-5)


def test_ramp_down_exactly_determines_a_forced_shutdown_trajectory():
    """A steeply descending per-hour floor pins down a unique optimal schedule too.

    Same standalone electric kiln, with ramp_down=5 the binding limit for decreases and
    ramp_up=20 (== max) never limiting increases (nor the first step - ramp_down does
    not apply there either, since there is no previous value to decrease from). The
    first three hours sit flat at the plateau their floor demands; from hour 3 the
    cheapest feasible choice is to shed exactly ramp_down per hour until the floor is
    reached, giving the unique trajectory [20, 20, 20, 15, 10, 5].
    """
    n = 6
    components = {
        "kiln": {
            "max_heat_out": 20,
            "specific_heat_demand": 1.0,
            "fuel_type": "electricity",
            "eta_electric": 1.0,
            "ramp_up": 20,
            "ramp_down": 5,
            "initial_operational_status": 1,
        }
    }
    per_hour_min = [20.0, 20.0, 20.0, 15.0, 10.0, 5.0]
    forecaster = make_forecaster(n=n, prices=[50] * n, clinker_demand=per_hour_min)
    plant = make_plant(
        components, forecaster, demand=0, plant_id="test_ramp_down_exact"
    )
    instance = solved_instance(plant)

    expected = [20, 20, 20, 15, 10, 5]
    actual = [pyo.value(instance.dsm_blocks["kiln"].heat_out[t]) for t in range(n)]
    assert actual == pytest.approx(expected, abs=1e-5)


def test_min_operating_and_down_steps_prevent_rapid_switching_on_a_thermal_stage():
    """Alternating prices create constant pressure to switch; the run-length floor must hold.

    A kiln with min_operating_steps=3 and min_down_steps=2, under a price that flips
    cheap/expensive every hour, would - if the constraint did nothing - want to turn on
    and off every other hour to chase the cheap ones. A per-horizon demand forces it to
    run somewhere. The first hour is deliberately made expensive so the optimal on-block
    does not start at t=0, where the shared min-up-time formulation has a known
    horizon-boundary exemption (a start-up at the very first step predates the earliest
    constraint window) - this test is about the steady-state mechanism, not that edge.
    """
    n = 12
    components = {
        "kiln": {
            "max_heat_out": 10,
            "min_heat_out": 3,
            "specific_heat_demand": 1.0,
            "fuel_type": "electricity",
            "eta_electric": 1.0,
            "ramp_up": 10,
            "ramp_down": 10,
            "min_operating_steps": 3,
            "min_down_steps": 2,
            "initial_operational_status": 0,
        }
    }
    prices = [200, 10, 200, 10, 200, 10, 200, 10, 200, 10, 200, 10]
    plant = make_plant(
        components,
        make_forecaster(n=n, prices=prices),
        demand=30,
        plant_id="test_min_up_down_kiln",
    )
    assert hasattr(plant.model.dsm_blocks["kiln"], "operational_status")
    instance = solved_instance(plant)

    status = [
        round(pyo.value(instance.dsm_blocks["kiln"].operational_status[t]))
        for t in range(n)
    ]
    for value, start, end in interior_runs(status):
        run_length = end - start + 1
        if value == 1:
            assert run_length >= 3, (
                f"on-run {start}-{end} shorter than min_operating_steps"
            )
        else:
            assert run_length >= 2, f"off-run {start}-{end} shorter than min_down_steps"

    total_clinker = sum(pyo.value(instance.clinker_rate[t]) for t in range(n))
    assert total_clinker == pytest.approx(30, rel=1e-4)


def test_min_operating_and_down_steps_prevent_rapid_switching_on_a_standalone_mill():
    """Same run-length check, but for GrindingMill's own unit-commitment path.

    min_power > 0 is what turns a GrindingMill into a unit-commitment block; this
    exercises that path directly (no cement plant test did before), independent of the
    thermal-stage formulation exercised above.
    """
    n = 12
    components = {
        "cement_mill": {
            "max_power": 25,
            "min_power": 5,
            "efficiency": 1.0,
            "specific_electricity_consumption": 1.0,
            "ramp_up": 25,
            "ramp_down": 25,
            "min_operating_steps": 3,
            "min_down_steps": 2,
            "initial_operational_status": 0,
        }
    }
    prices = [200, 10, 200, 10, 200, 10, 200, 10, 200, 10, 200, 10]
    plant = make_plant(
        components,
        make_forecaster(n=n, prices=prices),
        demand=60,
        plant_id="test_min_up_down_mill",
    )
    assert hasattr(plant.model.dsm_blocks["cement_mill"], "operational_status")
    instance = solved_instance(plant)

    status = [
        round(pyo.value(instance.dsm_blocks["cement_mill"].operational_status[t]))
        for t in range(n)
    ]
    for value, start, end in interior_runs(status):
        run_length = end - start + 1
        if value == 1:
            assert run_length >= 3, (
                f"on-run {start}-{end} shorter than min_operating_steps"
            )
        else:
            assert run_length >= 2, f"off-run {start}-{end} shorter than min_down_steps"


def test_grinding_mill_efficiency_below_one_creates_a_real_mass_loss():
    """With efficiency < 1, material_input must exceed material_output at every step.

    Every existing mill test uses efficiency=1.0 (no mass loss), which never exercises
    the ``material_input == material_output / efficiency`` relation for efficiency < 1.
    """
    n = 4
    demand = 20
    components = {
        "cement_mill": {
            "max_power": 25,
            "min_power": 0,
            "efficiency": 0.8,
            "specific_electricity_consumption": 0.04,
            "ramp_up": 25,
            "ramp_down": 25,
        }
    }
    plant = make_plant(
        components, make_forecaster(n=n), demand=demand, plant_id="test_mill_efficiency"
    )
    instance = solved_instance(plant)

    mill = instance.dsm_blocks["cement_mill"]
    for t in instance.time_steps:
        output = pyo.value(mill.material_output[t])
        assert pyo.value(mill.material_input[t]) == pytest.approx(
            output / 0.8, rel=1e-6
        )
        if output > 1e-6:
            assert pyo.value(mill.material_input[t]) > output + 1e-6

    total_output = sum(pyo.value(mill.material_output[t]) for t in instance.time_steps)
    assert total_output == pytest.approx(demand, rel=1e-4)


def test_dual_hydrogen_burners_share_one_electrolyser_exactly():
    """Calciner and kiln both hydrogen-fired, fed by a single electrolyser.

    Verifies the full split - electrolyser production, the two burner draws, and the
    unused slack - balances exactly, and that each burner's own hydrogen_in exactly
    matches the routed amount, not just in aggregate.
    """
    n = 4
    components = {
        "calciner": {
            "max_heat_out": 5,
            "specific_heat_demand": 0.7,
            "fuel_type": "hydrogen",
            "eta_fossil": 0.9,
            "calcination_emission_factor": 0.525,
            "ramp_up": 5,
            "ramp_down": 5,
        },
        "kiln": {
            "max_heat_out": 20,
            "specific_heat_demand": 1.7,
            "fuel_type": "hydrogen",
            "eta_fossil": 0.9,
            "ramp_up": 20,
            "ramp_down": 20,
        },
        "electrolyser": {
            "max_power": 30,
            "min_power": 0,
            "efficiency": 0.7,
            "ramp_up": 30,
            "ramp_down": 30,
        },
    }
    plant = make_plant(
        components, make_forecaster(n=n), demand=10, plant_id="test_dual_h2"
    )
    instance = solved_instance(plant)

    calciner = instance.dsm_blocks["calciner"]
    kiln = instance.dsm_blocks["kiln"]
    electrolyser = instance.dsm_blocks["electrolyser"]

    for t in instance.time_steps:
        h2_to_calciner = pyo.value(instance.h2_to_calciner[t])
        h2_to_kiln = pyo.value(instance.h2_to_kiln[t])
        h2_unutilised = pyo.value(instance.h2_unutilised[t])

        assert pyo.value(calciner.hydrogen_in[t]) == pytest.approx(
            h2_to_calciner, abs=1e-6
        )
        assert pyo.value(kiln.hydrogen_in[t]) == pytest.approx(h2_to_kiln, abs=1e-6)
        assert pyo.value(electrolyser.hydrogen_out[t]) == pytest.approx(
            h2_to_calciner + h2_to_kiln + h2_unutilised, abs=1e-6
        )
        assert pyo.value(electrolyser.hydrogen_out[t]) == pytest.approx(
            pyo.value(electrolyser.power_in[t]) * 0.7, abs=1e-6
        )
        # Neither burner draws any fossil fuel while hydrogen-fired
        assert pyo.value(calciner.fossil_in[t]) == pytest.approx(0.0, abs=1e-6)
        assert pyo.value(kiln.fossil_in[t]) == pytest.approx(0.0, abs=1e-6)


def test_thermal_storage_soc_balance_matches_the_exact_formula():
    """Reconstruct soc[t] from the solved charge/discharge/efficiency values and compare.

    Uses non-default efficiencies (efficiency_charge=0.9, efficiency_discharge=0.95,
    storage_loss_rate=0.02) and an electric heater (eta_electric=0.97) that every prior
    storage test left at their lossless defaults, so this is the first test to actually
    exercise the loss terms in ``soc_balance_rule`` and ``electric_heater_charge``.
    """
    n = 6
    components = {
        "calciner": {
            "max_heat_out": 5,
            "specific_heat_demand": 0.7,
            "fuel_type": "fossil",
            "eta_fossil": 0.9,
            "fossil_ng_share": 0.7,
            "calcination_emission_factor": 0.525,
            "ramp_up": 5,
            "ramp_down": 5,
        },
        "thermal_storage": {
            "capacity": 20,
            "max_power_charge": 10,
            "max_power_discharge": 20,
            "ramp_up": 10,
            "ramp_down": 20,
            "initial_soc": 0.2,
            "storage_type": "short-term_with_generator",
            "eta_electric": 0.97,
            "efficiency_charge": 0.9,
            "efficiency_discharge": 0.95,
            "storage_loss_rate": 0.02,
        },
    }
    # Cheap power early to encourage charging, expensive gas throughout to encourage
    # later discharging - gives the storage a reason to actually cycle.
    prices = [5, 5, 5, 300, 300, 300]
    forecaster = CementForecaster(
        pd.date_range("2023-01-01", periods=n, freq="h"),
        electricity_price=prices,
        fuel_prices={
            "natural_gas": [500] * n,
            "coal": [500] * n,
            "hydrogen": [500] * n,
            "co2": [10] * n,
        },
    )
    plant = make_plant(
        components, forecaster, demand=6, plant_id="test_storage_formula"
    )
    instance = solved_instance(plant)

    storage = instance.dsm_blocks["thermal_storage"]
    capacity = 20.0
    prev_soc_fraction = 0.2
    for t in instance.time_steps:
        power_in = pyo.value(storage.power_in[t])
        charge = pyo.value(storage.charge[t])
        discharge = pyo.value(storage.discharge[t])
        soc = pyo.value(storage.soc[t])

        # The electric heater's own conversion efficiency
        assert charge == pytest.approx(power_in * 0.97, abs=1e-6)

        prev_soc_abs = prev_soc_fraction * capacity
        expected_soc = (
            prev_soc_abs + 0.9 * charge - (1 / 0.95) * discharge - 0.02 * prev_soc_abs
        )
        assert soc == pytest.approx(expected_soc, abs=1e-5)
        prev_soc_fraction = soc / capacity

    # The storage actually cycled (charged in the cheap hours), not just sat idle
    total_charge = sum(pyo.value(storage.charge[t]) for t in instance.time_steps)
    assert total_charge > 1e-3


def test_full_plant_golden_integration(cement_components):
    """Every configured stage together: cross-check every link at every time step.

    Isolated tests elsewhere verify each formula individually; this test's job is to
    catch *interaction* bugs - e.g. cement_mill wiring breaking waste-heat recovery
    indexing, or thermal storage double-counting into the calciner's heat balance -
    that only show up when every optional component is present simultaneously.
    """
    n = 6
    demand = 15
    components = copy.deepcopy(cement_components)
    components["thermal_storage"] = {
        "capacity": 20,
        "max_power_charge": 10,
        "max_power_discharge": 20,
        "ramp_up": 10,
        "ramp_down": 20,
        "initial_soc": 0,
        "storage_type": "short-term_with_generator",
        "eta_electric": 0.97,
    }
    components["cement_mill"] = {
        "max_power": 25,
        "min_power": 0,
        "efficiency": 0.95,
        "specific_electricity_consumption": 0.04,
        "ramp_up": 25,
        "ramp_down": 25,
    }
    plant = make_plant(
        components,
        make_forecaster(n=n, prices=[80, 60, 40, 20, 60, 80]),
        demand=demand,
        plant_id="test_golden",
    )
    instance = solved_instance(plant)

    preheater = instance.dsm_blocks["preheater"]
    calciner = instance.dsm_blocks["calciner"]
    kiln = instance.dsm_blocks["kiln"]
    storage = instance.dsm_blocks["thermal_storage"]
    mill = instance.dsm_blocks["cement_mill"]

    total_clinker = 0.0
    for t in instance.time_steps:
        clinker = pyo.value(kiln.clinker_out[t])
        total_clinker += clinker

        # Raw meal -> calciner -> kiln mass chain
        assert pyo.value(calciner.clinker_out[t]) == pytest.approx(clinker, abs=1e-6)
        assert pyo.value(preheater.raw_meal_out[t]) == pytest.approx(
            clinker * RAW_MEAL_RATIO, abs=1e-6
        )

        # Waste heat recovery
        recovered = clinker * WASTE_HEAT * WASTE_HEAT_UTILISATION
        assert pyo.value(preheater.external_heat_in[t]) == pytest.approx(
            recovered, abs=1e-6
        )

        # Calciner heat balance including the thermal storage. The storage is in
        # "short-term_with_generator" mode, so it is charged from grid power (via
        # power_in), not from the calciner's own burner heat - effective_heat_in
        # therefore only adds the discharge, it does not subtract the charge.
        discharge = pyo.value(storage.discharge[t])
        assert pyo.value(calciner.effective_heat_in[t]) == pytest.approx(
            pyo.value(calciner.heat_out[t]) + discharge, abs=1e-6
        )

        # Cement mill grinds exactly the clinker produced
        assert pyo.value(mill.material_input[t]) == pytest.approx(clinker, abs=1e-6)
        assert pyo.value(mill.material_output[t]) == pytest.approx(
            clinker * 0.95, abs=1e-6
        )

        # Plant-level totals reconstructed from the components, independently of the
        # constraints that are supposed to enforce them
        expected_power = sum(
            pyo.value(getattr(block, "power_in")[t])
            for block in (preheater, calciner, kiln, storage, mill)
        ) + sum(
            pyo.value(getattr(block, "aux_power")[t])
            for block in (preheater, calciner, kiln)
        )
        assert pyo.value(instance.total_power_input[t]) == pytest.approx(
            expected_power, abs=1e-6
        )

        expected_cost = sum(
            pyo.value(getattr(block, "operating_cost")[t])
            for block in (preheater, calciner, kiln, storage, mill)
        )
        assert pyo.value(instance.variable_cost[t]) == pytest.approx(
            expected_cost, abs=1e-6
        )

    assert total_clinker == pytest.approx(demand, rel=1e-4)


# ---------------------------------------------------------------------------
# 3. Flexibility measures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flexibility_measure",
    [
        "cost_based_load_shift",
        "congestion_management_flexibility",
        "peak_load_shifting",
        "renewable_utilisation",
        "electricity_price_signal",
    ],
)
def test_flexibility_measures_produce_a_flex_profile(
    cement_components, flexibility_measure
):
    n = 6
    components = copy.deepcopy(cement_components)
    # An electric calciner gives the plant something to shift
    components["calciner"]["fuel_type"] = "electricity"
    plant = make_plant(
        components,
        make_forecaster(n=n, prices=[80, 60, 40, 20, 60, 80]),
        flexibility_measure=flexibility_measure,
        demand=20,
        plant_id=f"test_flex_{flexibility_measure}",
    )
    plant.determine_optimal_operation_with_flex()

    assert isinstance(plant.flex_power_requirement, FastSeries)
    assert len(plant.flex_power_requirement) == n


def test_peak_load_shifting_respects_the_peak_cap(cement_components):
    n = 6
    components = copy.deepcopy(cement_components)
    components["calciner"]["fuel_type"] = "electricity"
    plant = make_plant(
        components,
        make_forecaster(n=n, prices=[80, 60, 40, 20, 60, 80]),
        flexibility_measure="peak_load_shifting",
        demand=20,
        plant_id="test_peak_cap",
    )
    plant.determine_optimal_operation_with_flex()

    instance = plant.model.create_instance()
    instance = plant.switch_to_flex(instance)
    plant.solver.solve(instance, tee=False)

    cap = pyo.value(instance.peak_load_cap_value)
    for t in instance.time_steps:
        component_power = sum(
            pyo.value(instance.dsm_blocks[block].power_in[t])
            for block in instance.dsm_blocks
            if hasattr(instance.dsm_blocks[block], "power_in")
        ) + sum(
            pyo.value(instance.dsm_blocks[block].aux_power[t])
            for block in instance.dsm_blocks
            if hasattr(instance.dsm_blocks[block], "aux_power")
        )
        assert component_power <= cap + 1e-4


def test_peak_load_shifting_actually_lowers_a_former_peak_hour(cement_components):
    """Not just "respects the cap" - the peak hour's dispatch must genuinely change.

    Compares the flex dispatch against the unconstrained optimum directly: at least one
    hour that exceeded the cap before flexibility was applied must be strictly below its
    own former value after, proving the constraint is doing real work rather than
    being vacuously satisfied because nothing ever exceeded the cap to begin with.
    """
    n = 6
    components = copy.deepcopy(cement_components)
    components["calciner"]["fuel_type"] = "electricity"
    plant = make_plant(
        components,
        make_forecaster(n=n, prices=[80, 60, 40, 20, 200, 80]),
        flexibility_measure="peak_load_shifting",
        demand=20,
        plant_id="test_peak_shift_moves",
    )
    plant.determine_optimal_operation_without_flex()
    baseline = [plant.opt_power_requirement.iloc[t] for t in range(n)]
    cap = max(baseline) * (plant.peak_load_cap / 100)
    former_peaks = [t for t, v in enumerate(baseline) if v > cap]
    assert former_peaks, (
        "test setup must actually produce a peak hour to shift away from"
    )

    plant.determine_optimal_operation_with_flex()
    instance = plant.model.create_instance()
    instance = plant.switch_to_flex(instance)
    plant.solver.solve(instance, tee=False)

    def component_power(t):
        return sum(
            pyo.value(instance.dsm_blocks[block].power_in[t])
            for block in instance.dsm_blocks
            if hasattr(instance.dsm_blocks[block], "power_in")
        ) + sum(
            pyo.value(instance.dsm_blocks[block].aux_power[t])
            for block in instance.dsm_blocks
            if hasattr(instance.dsm_blocks[block], "aux_power")
        )

    for t in former_peaks:
        assert component_power(t) < baseline[t] - 1e-6
        assert component_power(t) <= cap + 1e-4

    # Total clinker demand is still met - the flexibility redistributes hours, it
    # does not shrink production.
    total_clinker = sum(pyo.value(instance.clinker_rate[t]) for t in range(n))
    assert total_clinker == pytest.approx(20, rel=1e-3)


def test_cost_based_load_shift_stays_within_the_exact_declared_tolerance(
    cement_components,
):
    """The flex-solve's actual cost must respect the numeric tolerance exactly.

    Computes the no-flex optimal cost precisely (``plant.total_cost``, the same value
    the model itself uses as the tolerance's baseline), then checks the flex-solve's
    realised cost against that exact number and the exact configured cost_tolerance -
    not just "some tolerance was probably respected".
    """
    n = 6
    components = copy.deepcopy(cement_components)
    components["calciner"]["fuel_type"] = "electricity"
    tolerance = 10  # matches make_plant's own default cost_tolerance
    plant = make_plant(
        components,
        make_forecaster(n=n, prices=[80, 60, 40, 20, 60, 80]),
        flexibility_measure="cost_based_load_shift",
        demand=20,
        plant_id="test_cost_tolerance_exact",
    )
    plant.determine_optimal_operation_without_flex()
    cost_opt = sum(plant.variable_cost_series.data)
    assert plant.total_cost == pytest.approx(cost_opt, rel=1e-9)

    plant.determine_optimal_operation_with_flex()
    cost_flex = sum(plant.flex_variable_cost_series.data)

    assert cost_flex <= cost_opt * (1 + tolerance / 100) + 1e-6


# ---------------------------------------------------------------------------
# 4. Rolling horizon
# ---------------------------------------------------------------------------


@pytest.fixture
def rh_config():
    """Minimal rolling-horizon config: 4h look-ahead, 2h commit, 2h step."""
    return {
        "horizon_mode": "rolling_horizon",
        "look_ahead_horizon": "4h",
        "commit_horizon": "2h",
        "rolling_step": "2h",
    }


def test_default_horizon_mode_is_full_horizon(cement_components):
    plant = make_plant(cement_components, make_forecaster(n=6), setup=False)
    assert plant.horizon_mode == "full_horizon"


def test_rolling_horizon_config_is_parsed(cement_components, rh_config):
    plant = make_plant(
        cement_components,
        make_forecaster(n=6),
        setup=False,
        dsm_optimisation_config=rh_config,
    )
    assert plant.horizon_mode == "rolling_horizon"
    assert plant._parse_duration_to_steps("4h") == 4


def _rh_plant(cement_components, rh_config, prices, demand=10, n=6, plant_id="test_rh"):
    components = copy.deepcopy(cement_components)
    components["calciner"]["fuel_type"] = "electricity"
    return make_plant(
        components,
        make_forecaster(n=n, prices=prices),
        demand=demand,
        plant_id=plant_id,
        dsm_optimisation_config=rh_config,
    )


def test_rolling_window_defers_production_to_cheap_look_ahead_hours(
    cement_components, rh_config
):
    """The committed hours stay idle when the look-ahead contains much cheaper hours."""
    plant = _rh_plant(
        cement_components,
        rh_config,
        prices=[200.0, 200.0, 200.0, 10.0, 10.0, 10.0],
        plant_id="test_rh_window",
    )

    triggered = plant._check_and_reoptimize_rolling_window(
        pd.Timestamp("2023-01-01 00:00")
    )
    assert triggered is True
    # Window 0 looks ahead over steps 0-3 and commits steps 0-1
    assert plant._rh_optimized_until_step == 2
    assert all(
        value == pytest.approx(0.0, abs=1e-4)
        for value in plant._rh_full_horizon_production[:2]
    )


def test_rolling_window_tracks_remaining_demand(cement_components, rh_config):
    """Production committed in earlier windows is deducted from the remaining demand."""
    plant = _rh_plant(
        cement_components,
        rh_config,
        prices=[10.0, 10.0, 200.0, 200.0, 200.0, 200.0],
        demand=10,
        plant_id="test_rh_demand",
    )

    plant._check_and_reoptimize_rolling_window(pd.Timestamp("2023-01-01 00:00"))
    produced_first_window = sum(plant._rh_full_horizon_production[:2])
    assert produced_first_window == pytest.approx(10, rel=1e-4)

    plant._check_and_reoptimize_rolling_window(pd.Timestamp("2023-01-01 02:00"))
    assert plant._rh_optimized_until_step == 4
    assert plant._rh_window_remaining_demand == pytest.approx(
        10 - produced_first_window, abs=1e-4
    )
    # Component-level results are recorded for the committed steps
    assert len(plant._component_operations) == 4


def test_rolling_window_carries_thermal_storage_state(cement_components, rh_config):
    """Thermal storage fill levels are carried from one committed window into the next."""
    components = copy.deepcopy(cement_components)
    components["thermal_storage"] = {
        "capacity": 20,
        "max_power_charge": 10,
        "max_power_discharge": 20,
        "ramp_up": 10,
        "ramp_down": 10,
        "initial_soc": 0,
        "storage_type": "short-term_with_generator",
    }
    plant = make_plant(
        components,
        make_forecaster(n=6, prices=[10.0] * 6),
        demand=6,
        plant_id="test_rh_state",
        dsm_optimisation_config=rh_config,
    )

    plant._check_and_reoptimize_rolling_window(pd.Timestamp("2023-01-01 00:00"))
    assert "soc" in plant._rh_init_states.get("thermal_storage", {})

    soc = plant._rh_init_states["thermal_storage"]["soc"]
    assert -1e-6 <= soc <= 1 + 1e-6


def test_rolling_horizon_full_day_meets_demand_every_window(cement_components):
    """Driving every window of a full 24h horizon must cover the whole day.

    Uses per-timestep minimums together with a global demand equal to their sum, so a
    solved model must both hit every hourly floor and hand back exactly the declared
    total once all eight 3h-step windows (6h look-ahead, 3h commit) have committed.
    Also checks the plant advances to the end of the horizon and that no window is left
    uncommitted (a stalled ``_rh_optimized_until_step`` would mean a window failed).
    """
    n = 24
    prices = [
        30,
        28,
        27,
        27,
        29,
        34,
        45,
        60,
        72,
        80,
        78,
        74,
        70,
        68,
        70,
        78,
        88,
        95,
        90,
        80,
        68,
        55,
        44,
        36,
    ]
    per_hour_min = [4.0] * n
    total_demand = sum(per_hour_min)

    components = copy.deepcopy(cement_components)
    components["calciner"]["fuel_type"] = "electricity"
    components["thermal_storage"] = {
        "capacity": 20,
        "max_power_charge": 10,
        "max_power_discharge": 20,
        "ramp_up": 10,
        "ramp_down": 10,
        "initial_soc": 0,
        "storage_type": "short-term_with_generator",
    }
    forecaster = make_forecaster(n=n, prices=prices, clinker_demand=per_hour_min)
    plant = make_plant(
        components,
        forecaster,
        demand=total_demand,
        plant_id="test_rh_full_day",
        dsm_optimisation_config={
            "horizon_mode": "rolling_horizon",
            "look_ahead_horizon": "6h",
            "commit_horizon": "3h",
            "rolling_step": "3h",
        },
    )

    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
    for ts in timestamps[::3]:  # one trigger per 3h commit step -> 8 windows
        plant._check_and_reoptimize_rolling_window(ts)

    # The plant advanced through every window to the very end of the horizon.
    assert plant._rh_optimized_until_step == n

    # Every hour met its declared floor, across all eight committed windows.
    shortfalls = [
        (t, v)
        for t, v in enumerate(plant._rh_full_horizon_production)
        if v < per_hour_min[t] - 1e-3
    ]
    assert shortfalls == [], f"hours below their minimum: {shortfalls}"

    # The cumulative committed production matches the declared total exactly.
    total_produced = sum(plant._rh_full_horizon_production)
    assert total_produced == pytest.approx(total_demand, rel=1e-3)

    # Component-level results were recorded for every committed step.
    assert len(plant._component_operations) == n

    # The thermal storage state carried across all eight windows stayed a valid
    # fraction of capacity (not just the state after the first hand-off).
    soc = plant._rh_init_states["thermal_storage"]["soc"]
    assert -1e-6 <= soc <= 1 + 1e-6


def test_rolling_horizon_works_for_a_standalone_mill():
    """A standalone cement mill (no kiln line) also converges under rolling horizon.

    Mirrors ``test_rolling_horizon_full_day_meets_demand_every_window`` but for a
    plant made of only a cement mill, proving the ``clinker_rate``-based
    generalization holds under multi-window rolling horizon too, rather than being
    correct only "by construction" in the full-horizon model.
    """
    n = 12
    prices = [60, 55, 50, 45, 10, 10, 10, 45, 50, 55, 60, 65]
    per_hour_min = [2.0] * n
    total_demand = sum(per_hour_min)

    components = {
        "cement_mill": {
            "max_power": 25,
            "min_power": 0,
            "efficiency": 1.0,
            "specific_electricity_consumption": 0.04,
            "ramp_up": 25,
            "ramp_down": 25,
        }
    }
    forecaster = make_forecaster(n=n, prices=prices, clinker_demand=per_hour_min)
    plant = make_plant(
        components,
        forecaster,
        demand=total_demand,
        plant_id="test_rh_standalone_mill",
        dsm_optimisation_config={
            "horizon_mode": "rolling_horizon",
            "look_ahead_horizon": "6h",
            "commit_horizon": "3h",
            "rolling_step": "3h",
        },
    )

    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
    for ts in timestamps[::3]:  # one trigger per 3h commit step -> 4 windows
        plant._check_and_reoptimize_rolling_window(ts)

    assert plant._rh_optimized_until_step == n

    shortfalls = [
        (t, v)
        for t, v in enumerate(plant._rh_full_horizon_production)
        if v < per_hour_min[t] - 1e-3
    ]
    assert shortfalls == [], f"hours below their minimum: {shortfalls}"

    total_produced = sum(plant._rh_full_horizon_production)
    assert total_produced == pytest.approx(total_demand, rel=1e-3)
    assert len(plant._component_operations) == n


def test_rolling_horizon_never_overproduces_once_demand_is_met():
    """Once the global demand is fully met early, later windows must produce exactly zero.

    The whole global demand is made cheap to satisfy entirely within the first window's
    committed hours (hours 0-2, priced far below the rest of the horizon). The window
    demand constraint (``sum(primary_output) <= remaining_demand``) then forces every
    later window's remaining_demand to exactly 0, which - since production is
    non-negative - forces every later hour's production to exactly 0 too, not just "on
    average" zero.
    """
    n = 9
    components = {
        "kiln": {
            "max_heat_out": 10,
            "specific_heat_demand": 1.0,
            "fuel_type": "electricity",
            "eta_electric": 1.0,
            "ramp_up": 10,
            "ramp_down": 10,
        }
    }
    prices = [10, 10, 10, 200, 200, 200, 200, 200, 200]
    plant = make_plant(
        components,
        make_forecaster(n=n, prices=prices),
        demand=10,
        plant_id="test_rh_no_overproduction",
        dsm_optimisation_config={
            "horizon_mode": "rolling_horizon",
            "look_ahead_horizon": "6h",
            "commit_horizon": "3h",
            "rolling_step": "3h",
        },
    )

    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
    for i, ts in enumerate(timestamps[::3]):
        plant._check_and_reoptimize_rolling_window(ts)
        if i == 0:
            # The whole demand was cheap enough to clear in the first committed block.
            produced_so_far = sum(plant._rh_full_horizon_production[:3])
            assert produced_so_far == pytest.approx(10, rel=1e-4)
        elif i == 1:
            # _rh_window_remaining_demand reflects what was remaining as this window's
            # solve started, i.e. it only shows the post-window0 value once window1
            # actually runs.
            assert plant._rh_window_remaining_demand == pytest.approx(0.0, abs=1e-6)

    assert plant._rh_optimized_until_step == n
    later_production = plant._rh_full_horizon_production[3:]
    assert later_production == pytest.approx([0.0] * (n - 3), abs=1e-6)

    total_produced = sum(plant._rh_full_horizon_production)
    assert total_produced == pytest.approx(10, rel=1e-4)


def test_cement_plant_with_none_components():
    """Initializing CementPlant with components=None should not raise AttributeError."""
    forecaster = make_forecaster(n=6)
    plant = CementPlant(
        id="test_none_components",
        unit_operator="test_operator",
        bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
        forecaster=forecaster,
        components=None,
        demand=0,
    )
    assert plant.components == {}


def test_cement_plant_invalid_forecaster_type():
    """Passing a non-CementForecaster instance must raise TypeError."""
    with pytest.raises(TypeError, match="forecaster must be of type CementForecaster"):
        CementPlant(
            id="test_invalid_forecaster",
            unit_operator="test_operator",
            bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
            forecaster="not_a_forecaster",
            components={},
        )


def test_thermal_process_stage_requires_co2_price():
    """ThermalProcessStage must reject a model missing co2_price."""
    from assume.units.dst_components import Kiln

    kiln = Kiln(max_heat_out=10, specific_heat_demand=1.7, time_steps=[0, 1])
    m = pyo.ConcreteModel()
    m.electricity_price = pyo.Param([0, 1], initialize=10.0)
    # co2_price is intentionally omitted from m
    with pytest.raises(ValueError, match="requires a 'co2_price' profile"):
        kiln.add_to_model(m, pyo.Block())


def test_component_schema_unique_keys_in_operations(cement_components):
    """Component operations dict must have distinct keys for preheater and raw_material_mill."""
    components = copy.deepcopy(cement_components)
    components["raw_material_mill"] = {
        "max_power": 10,
        "specific_electricity_consumption": 0.02,
    }
    forecaster = make_forecaster(n=6)
    plant = make_plant(
        components,
        forecaster,
        demand=10,
        plant_id="test_schema_keys",
        dsm_optimisation_config={
            "horizon_mode": "rolling_horizon",
            "look_ahead_horizon": "6h",
            "commit_horizon": "3h",
            "rolling_step": "3h",
        },
    )
    plant._check_and_reoptimize_rolling_window(pd.Timestamp("2023-01-01 00:00"))
    assert len(plant._component_operations) == 3
    op_row = plant._component_operations[0]
    assert "preheater_raw_meal_output" in op_row
    assert "raw_material_mill_output" in op_row


if __name__ == "__main__":
    pytest.main(["-s", __file__])
