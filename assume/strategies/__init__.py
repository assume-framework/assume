# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.base import BaseStrategy, LearningStrategy
from assume.strategies.advanced_orders import flexableEOMBlock, flexableEOMLinked
from assume.strategies.dmas_powerplant import DmasPowerplantStrategy
from assume.strategies.dmas_storage import DmasStorageStrategy
from assume.strategies.extended import OTCStrategy
from assume.strategies.flexable import flexableEOM, flexableNegCRM, flexablePosCRM
from assume.strategies.flexable_storage import (
    flexableEOMStorage,
    flexableNegCRMStorage,
    flexablePosCRMStorage,
)
from assume.strategies.naive_strategies import (
    NaiveDAStrategy,
    NaiveNegReserveStrategy,
    NaivePosReserveStrategy,
    NaiveStrategy,
)

bidding_strategies: dict[str, BaseStrategy] = {
    "naive": NaiveStrategy,
    "naive_da": NaiveDAStrategy,
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
}
