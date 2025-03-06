# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.base import BaseStrategy, LearningStrategy
from assume.strategies.standard_advanced_orders import (
    EOMBlockPowerplant,
    EOMLinkedPowerplant,
)
from assume.strategies.extended import OTCStrategy
from assume.strategies.standard_powerplant import (
    EOMPowerplant,
    NCRMPowerplant,
    PCRMPowerplant,
)
from assume.strategies.standard_storage import (
    EOMStorage,
    NCRMStorage,
    PCRMStorage,
)
from assume.strategies.naive_strategies import (
    NaiveDADSMStrategy,
    NaiveProfileStrategy,
    NaiveRedispatchDSMStrategy,
    NaiveRedispatchStrategy,
    NaiveSingleBidStrategy,
    NaiveExchangeStrategy,
)
from assume.strategies.manual_strategies import SimpleManualTerminalStrategy
from assume.strategies.dmas_powerplant import DmasPowerplantStrategy
from assume.strategies.dmas_storage import DmasStorageStrategy


bidding_strategies: dict[str, BaseStrategy] = {
    "naive_eom_powerplant": NaiveSingleBidStrategy,
    "naive_eom_demand": NaiveSingleBidStrategy,
    "naive_eom_block_powerplant": NaiveProfileStrategy,
    "naive_pcrm": NaiveSingleBidStrategy,
    "naive_ncrm": NaiveSingleBidStrategy,
    "naive_exchange": NaiveExchangeStrategy,
    "otc_strategy": OTCStrategy,
    "eom_powerplant": EOMPowerplant,
    "eom_block_powerplant": EOMBlockPowerplant,
    "eom_linked_powerplant": EOMLinkedPowerplant,
    "ncrm_powerplant": NCRMPowerplant,
    "pcrm_powerplant": PCRMPowerplant,
    "eom_storage": EOMStorage,
    "ncrm_storage": NCRMStorage,
    "pcrm_storage": PCRMStorage,
    "naive_redispatch": NaiveRedispatchStrategy,
    "naive_eom_block_dsm": NaiveDADSMStrategy,
    "naive_redispatch_dsm": NaiveRedispatchDSMStrategy,
    "manual_strategy": SimpleManualTerminalStrategy,
    "dmas_powerplant": DmasPowerplantStrategy,
    "dmas_storage": DmasStorageStrategy,
}

try:
    from assume.strategies.learning_advanced_orders import (
        RLAdvancedOrderStrategy,
    )
    from assume.strategies.learning_strategies import (
        RLStrategy,
        StorageRLStrategy,
    )

    bidding_strategies["learning_powerplant"] = RLStrategy
    bidding_strategies["learning_storage"] = StorageRLStrategy
    bidding_strategies["learning_advanced_orders"] = RLAdvancedOrderStrategy

except ImportError:
    pass
