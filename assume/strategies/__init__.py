# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.base import BaseStrategy, LearningStrategy
from assume.strategies.advanced_orders import (
    EnergyHeuristicFlexableBlockStrategy,
    EnergyHeuristicFlexableLinkedStrategy,
)
from assume.strategies.extended import EnergyNaiveOtcStrategy
from assume.strategies.flexable import (
    EnergyHeuristicFlexableStrategy,
    CapacityHeuristicBalancingNegStrategy,
    CapacityHeuristicBalancingPosStrategy,
)
from assume.strategies.flexable_storage import (
    StorageEnergyHeuristicFlexableStrategy,
    StorageCapacityHeuristicBalancingNegStrategy,
    StorageCapacityHeuristicBalancingPosStrategy,
)
from assume.strategies.naive_strategies import (
    DsmEnergyOptimizationStrategy,
    EnergyNaiveProfileStrategy,
    DsmEnergyNaiveRedispatchStrategy,
    EnergyNaiveRedispatchStrategy,
    EnergyNaiveStrategy,
    ExchangeEnergyNaiveStrategy,
    EnergyHeuristicElasticStrategy,
    DsmCapacityHeuristicBalancingPosStrategy,
    DsmCapacityHeuristicBalancingNegStrategy,
)
from assume.strategies.interactive_strategies import EnergyInteractiveStrategy
from assume.strategies.dmas_powerplant import EnergyOptimizationDmasStrategy
from assume.strategies.dmas_storage import StorageEnergyOptimizationDmasStrategy
from assume.strategies.portfolio_strategies import (
    UnitOperatorStrategy,
    UnitsOperatorDirectStrategy,
    UnitsOperatorEnergyHeuristicCournotStrategy,
)

# TODO remove after a few releases
deprecated_bidding_strategies: dict[str, type[BaseStrategy | UnitOperatorStrategy]] = {
    "naive_neg_reserve": EnergyNaiveStrategy,
    "naive_exchange": ExchangeEnergyNaiveStrategy,
    "naive_eom": EnergyNaiveStrategy,
    "elastic_demand": EnergyHeuristicElasticStrategy,
    "naive_pos_reserve": EnergyNaiveStrategy,
    "flexable_eom_storage": StorageEnergyHeuristicFlexableStrategy,
    "otc_strategy": EnergyNaiveOtcStrategy,
    "flexable_eom": EnergyHeuristicFlexableStrategy,
    "flexable_eom_block": EnergyHeuristicFlexableBlockStrategy,
    "flexable_neg_crm": CapacityHeuristicBalancingNegStrategy,
    "flexable_pos_crm": CapacityHeuristicBalancingPosStrategy,
    "flexable_eom_linked": EnergyHeuristicFlexableLinkedStrategy,
    "flexable_neg_crm_storage": StorageCapacityHeuristicBalancingNegStrategy,
    "flexable_pos_crm_storage": StorageCapacityHeuristicBalancingPosStrategy,
    "pos_crm_dsm": DsmCapacityHeuristicBalancingPosStrategy,
    "neg_crm_dsm": DsmCapacityHeuristicBalancingNegStrategy,
    "naive_redispatch": EnergyNaiveRedispatchStrategy,
    "naive_da_dsm": DsmEnergyOptimizationStrategy,
    "naive_redispatch_dsm": DsmEnergyNaiveRedispatchStrategy,
}

bidding_strategies: dict[str, type[BaseStrategy | UnitOperatorStrategy]] = {
    "powerplant_energy_naive": EnergyNaiveStrategy,
    "demand_energy_naive": EnergyNaiveStrategy,
    "powerplant_energy_naive_balancing": EnergyNaiveStrategy,
    "demand_energy_naive_balancing": EnergyNaiveStrategy,
    "demand_energy_heuristic_elastic": EnergyHeuristicElasticStrategy,
    "exchange_energy_naive": ExchangeEnergyNaiveStrategy,
    "demand_energy_naive_otc": EnergyNaiveOtcStrategy,
    "powerplant_energy_naive_otc": EnergyNaiveOtcStrategy,
    "powerplant_energy_heuristic_flexable": EnergyHeuristicFlexableStrategy,
    "powerplant_energy_heuristic_block": EnergyHeuristicFlexableBlockStrategy,
    "powerplant_energy_heuristic_linked": EnergyHeuristicFlexableLinkedStrategy,
    "powerplant_capacity_heuristic_balancing_neg": CapacityHeuristicBalancingNegStrategy,
    "powerplant_capacity_heuristic_balancing_pos": CapacityHeuristicBalancingPosStrategy,
    "storage_energy_heuristic_flexable": StorageEnergyHeuristicFlexableStrategy,
    "storage_capacity_heuristic_balancing_neg": StorageCapacityHeuristicBalancingNegStrategy,
    "storage_capacity_heuristic_balancing_pos": StorageCapacityHeuristicBalancingPosStrategy,
    "household_capacity_heuristic_balancing_pos": DsmCapacityHeuristicBalancingPosStrategy,
    "industry_capacity_heuristic_balancing_pos": DsmCapacityHeuristicBalancingPosStrategy,
    "household_capacity_heuristic_balancing_neg": DsmCapacityHeuristicBalancingNegStrategy,
    "industry_capacity_heuristic_balancing_neg": DsmCapacityHeuristicBalancingNegStrategy,
    "powerplant_energy_naive_redispatch": EnergyNaiveRedispatchStrategy,
    "demand_energy_naive_redispatch": EnergyNaiveRedispatchStrategy,
    "household_energy_optimization": DsmEnergyOptimizationStrategy,
    "industry_energy_optimization": DsmEnergyOptimizationStrategy,
    "household_energy_naive_redispatch": DsmEnergyNaiveRedispatchStrategy,
    "industry_energy_naive_redispatch": DsmEnergyNaiveRedispatchStrategy,
    "powerplant_energy_optimization_dmas": EnergyOptimizationDmasStrategy,
    "storage_energy_optimization_dmas": StorageEnergyOptimizationDmasStrategy,
    "units_operator_energy_heuristic_cournot": UnitsOperatorEnergyHeuristicCournotStrategy,
    "units_operator_direct": UnitsOperatorDirectStrategy,
    "powerplant_energy_naive_profile": EnergyNaiveProfileStrategy,
    "powerplant_energy_interactive": EnergyInteractiveStrategy,
}

try:
    from assume.strategies.learning_strategies import (
        EnergyLearningStrategy,
        EnergyLearningSingleBidStrategy,
        StorageEnergyLearningStrategy,
        RenewableEnergyLearningSingleBidStrategy,
    )

    deprecated_bidding_strategies["pp_learning"] = EnergyLearningStrategy
    deprecated_bidding_strategies["storage_learning"] = StorageEnergyLearningStrategy
    deprecated_bidding_strategies["renewable_eom_learning"] = (
        RenewableEnergyLearningSingleBidStrategy
    )
    deprecated_bidding_strategies["learning_advanced_orders"] = EnergyLearningStrategy

    bidding_strategies["powerplant_energy_learning"] = EnergyLearningStrategy
    bidding_strategies["powerplant_energy_learning_single_bid"] = (
        EnergyLearningSingleBidStrategy
    )
    bidding_strategies["storage_energy_learning"] = StorageEnergyLearningStrategy
    bidding_strategies["renewable_energy_learning_single_bid"] = (
        RenewableEnergyLearningSingleBidStrategy
    )


except ImportError:
    pass
