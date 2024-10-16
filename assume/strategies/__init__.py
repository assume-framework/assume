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
    NaiveDASteelplantStrategy,
    NaiveNegReserveStrategy,
    NaivePosReserveStrategy,
    NaiveProfileStrategy,
    NaiveRedispatchSteelplantStrategy,
    NaiveRedispatchStrategy,
    NaiveSingleBidStrategy,
)
from assume.strategies.manual_strategies import SimpleManualTerminalStrategy

bidding_strategies: dict[str, BaseStrategy] = {
    "naive_eom": NaiveSingleBidStrategy,
    "naive_dam": NaiveProfileStrategy,
    "naive_pos_reserve": NaivePosReserveStrategy,
    "naive_neg_reserve": NaiveNegReserveStrategy,
    "otc_strategy": OTCStrategy,
    "flexable_eom": flexableEOM,
    "flexable_eom_block": flexableEOMBlock,
    "flexable_eom_linked": flexableEOMLinked,
    "flexable_neg_crm": flexableNegCRM,
    "flexable_pos_crm": flexablePosCRM,
    "flexable_eom_storage": flexableEOMStorage,
    "flexable_neg_crm_storage": flexableNegCRMStorage,
    "flexable_pos_crm_storage": flexablePosCRMStorage,
    "naive_redispatch": NaiveRedispatchStrategy,
    "naive_da_steelplant": NaiveDASteelplantStrategy,
    "naive_steel_redispatch": NaiveRedispatchSteelplantStrategy,
    "manual_strategy": SimpleManualTerminalStrategy,
}

try:
    from assume.strategies.dmas_powerplant import DmasPowerplantStrategy
    from assume.strategies.dmas_storage import DmasStorageStrategy

    bidding_strategies["dmas_powerplant"] = DmasPowerplantStrategy
    bidding_strategies["dmas_storage"] = DmasStorageStrategy

except ImportError:
    pass


try:
    from assume.strategies.learning_advanced_orders import (
        RLAdvancedOrderStrategy,
    )
    from assume.strategies.learning_strategies import RLStrategy

    bidding_strategies["pp_learning"] = RLStrategy
    bidding_strategies["learning_advanced_orders"] = RLAdvancedOrderStrategy

except ImportError:
    pass
