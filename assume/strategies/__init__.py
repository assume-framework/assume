# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.base import BaseStrategy, LearningStrategy
from assume.strategies.advanced_orders import flexableEOMBlock, flexableEOMLinked
from assume.strategies.extended import OTCStrategy
from assume.strategies.flexable import flexableEOM, flexableNegCRM, flexablePosCRM
from assume.strategies.flexable_storage import (
    flexableEOMStorage,
    flexableNegCRMStorage,
    flexablePosCRMStorage,
)
from assume.strategies.naive_strategies import (
    NaiveDADSMStrategy,
    NaiveProfileStrategy,
    NaiveRedispatchDSMStrategy,
    NaiveRedispatchStrategy,
    NaiveSingleBidStrategy,
    NaiveExchangeStrategy,
    ElasticDemandStrategy,
    DSM_PosCRM_Strategy,
    DSM_NegCRM_Strategy,
)
from assume.strategies.manual_strategies import SimpleManualTerminalStrategy
from assume.strategies.dmas_powerplant import DmasPowerplantStrategy
from assume.strategies.dmas_storage import DmasStorageStrategy
from assume.strategies.portfolio_strategies import (
    UnitOperatorStrategy,
    DirectUnitOperatorStrategy,
    CournotPortfolioStrategy,
)

bidding_strategies: dict[str, type[BaseStrategy | UnitOperatorStrategy]] = {
    #TODO: REMOVE BECAUSE OF DEPRECATION ####################################
    "naive_neg_reserve": NaiveSingleBidStrategy,
    "naive_exchange": NaiveExchangeStrategy,
    "naive_eom": NaiveSingleBidStrategy,
    "elastic_demand": ElasticDemandStrategy,
    "naive_pos_reserve": NaiveSingleBidStrategy,
    "flexable_eom_storage": flexableEOMStorage,
    "otc_strategy": OTCStrategy,
    "flexable_eom": flexableEOM,
    "flexable_eom_block": flexableEOMBlock,
    "flexable_neg_crm": flexableNegCRM,
    "flexable_pos_crm": flexablePosCRM,
    "flexable_eom_linked": flexableEOMLinked,
    "flexable_neg_crm_storage": flexableNegCRMStorage,
    "flexable_pos_crm_storage": flexablePosCRMStorage,
    "pos_crm_dsm": DSM_PosCRM_Strategy,
    "neg_crm_dsm": DSM_NegCRM_Strategy,
    "naive_redispatch": NaiveRedispatchStrategy,
    "naive_da_dsm": NaiveDADSMStrategy,
    "naive_redispatch_dsm": NaiveRedispatchDSMStrategy,
    # END OF REMOVE ################################################################


    "powerplant_energy_naive": NaiveSingleBidStrategy,
    "demand_energy_naive": NaiveSingleBidStrategy,
    "powerplant_energy_naive_balancing": NaiveSingleBidStrategy,
    "demand_energy_naive_balancing": NaiveSingleBidStrategy,
    "demand_energy_heuristic_elastic": ElasticDemandStrategy,
    "exchange_energy_naive": NaiveExchangeStrategy,
    "demand_energy_naive_otc": OTCStrategy,
    "powerplant_energy_naive_otc": OTCStrategy,
    "powerplant_energy_heuristic_flexable": flexableEOM,
    "powerplant_energy_heuristic_block": flexableEOMBlock,
    "powerplant_energy_heuristic_linked": flexableEOMLinked,
    "powerplant_capacity_heuristic_balancing_neg": flexableNegCRM,
    "powerplant_capacity_heuristic_balancing_pos": flexableNegCRM,
    "storage_energy_heuristic_flexable": flexableEOMStorage,
    "storage_capacity_heuristic_balancing_neg": flexableNegCRMStorage,
    "storage_capacity_heuristic_balancing_pos": flexablePosCRMStorage,
    "household_capacity_heuristic_balancing_pos": DSM_PosCRM_Strategy,
    "industry_capacity_heuristic_balancing_pos": DSM_PosCRM_Strategy,
    "household_capacity_heuristic_balancing_neg": DSM_NegCRM_Strategy,
    "industry_capacity_heuristic_balancing_neg": DSM_NegCRM_Strategy,
    "powerplant_energy_naive_redispatch": NaiveRedispatchStrategy,
    "demand_energy_naive_redispatch": NaiveRedispatchStrategy,
    "household_energy_naive": NaiveDADSMStrategy,
    "industry_energy_naive": NaiveDADSMStrategy,
    "household_energy_naive_redispatch": NaiveRedispatchDSMStrategy,
    "industry_energy_naive_redispatch": NaiveRedispatchDSMStrategy,
    "powerplant_energy_heuristic_dmas": DmasPowerplantStrategy,
    "storage_energy_heuristic_dmas": DmasStorageStrategy,
    "units_operator_energy_heuristic_cournot":CournotPortfolioStrategy,
    "units_operator_energy_naive_direct":DirectUnitOperatorStrategy,
    "powerplant_energy_naive_profile": NaiveProfileStrategy,
    "powerplant_energy_interactive":SimpleManualTerminalStrategy,
}

try:
    from assume.strategies.learning_strategies import (
        RLStrategy,
        RLStrategySingleBid,
        StorageRLStrategy,
        RenewableRLStrategy,
    )
    #TODO: REMOVE DEPRECATION###########################################
    bidding_strategies["pp_learning"] = RLStrategy
    bidding_strategies["storage_learning"] = StorageRLStrategy
    bidding_strategies["renewable_eom_learning"] = RenewableRLStrategy
    # END OF REMOVE DEPRECATION ###########################################

    bidding_strategies["powerplant_energy_learning"] = RLStrategy
    bidding_strategies["powerplant_energy_learning_single_bid"] = RLStrategySingleBid
    bidding_strategies["storage_energy_learning"] = StorageRLStrategy
    bidding_strategies["renewable_energy_learning_single_bid"] = RenewableRLStrategy


except ImportError:
    pass
