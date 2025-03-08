# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.base import BaseStrategy, LearningStrategy
from assume.strategies.standard_advanced_orders import (
    StandardProfileEOMPowerplantStrategy,
)
from assume.strategies.extended import OTCStrategy
from assume.strategies.standard_powerplant import (
    StandardEOMPowerplantStrategy,
    StandardNegCRMPowerplantStrategy,
    StandardPosCRMPowerplantStrategy,
)
from assume.strategies.standard_storage import (
    StandardEOMStorageStrategy,
    StandardNegCRMStorageStrategy,
    StandardPosCRMStorageStrategy,
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
    "naive": NaiveSingleBidStrategy,
    "naive_profile": NaiveProfileStrategy,
    "naive_profile_dsm": NaiveDADSMStrategy,
    "naive_exchange": NaiveExchangeStrategy,
    "naive_redispatch": NaiveRedispatchStrategy,
    "naive_redispatch_dsm": NaiveRedispatchDSMStrategy,
    "standard_eom_powerplant": StandardEOMPowerplantStrategy,
    "standard_profile_eom_powerplant": StandardProfileEOMPowerplantStrategy,
    "standard_pos_crm_powerplant": StandardPosCRMPowerplantStrategy,
    "standard_neg_crm_powerplant": StandardNegCRMPowerplantStrategy,
    "standard_eom_storage": StandardEOMStorageStrategy,
    "standard_pos_crm_storage": StandardPosCRMStorageStrategy,
    "standard_neg_crm_storage": StandardNegCRMStorageStrategy,
    "dmas_powerplant": DmasPowerplantStrategy,
    "dmas_storage": DmasStorageStrategy,
    "misc_otc": OTCStrategy,
    "misc_manual": SimpleManualTerminalStrategy,
}

try:
    from assume.strategies.learning_advanced_orders import (
        LearningProfileEOMPowerplantStrategy,
    )
    from assume.strategies.learning_strategies import (
        LearningEOMPowerplantStrategy,
        LearningEOMStorageStrategy,
    )

    bidding_strategies["learning_eom_powerplant"] = LearningEOMPowerplantStrategy
    bidding_strategies["learning_eom_storage"] = LearningEOMStorageStrategy
    bidding_strategies["learning_profile_eom_powerplant"] = (
        LearningProfileEOMPowerplantStrategy
    )

except ImportError:
    pass
