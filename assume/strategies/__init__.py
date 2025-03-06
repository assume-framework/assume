# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.base import BaseStrategy, LearningStrategy
from assume.strategies.standard_advanced_orders import (
    StandardProfileEOMPowerplant,
)
from assume.strategies.extended import OTCStrategy
from assume.strategies.standard_powerplant import (
    StandardEOMPowerplant,
    StandardNCRMPowerplant,
    StandardPCRMPowerplant,
)
from assume.strategies.standard_storage import (
    StandardEOMStorage,
    StandardNCRMStorage,
    StandardPCRMStorage,
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
    "standard_eom_powerplant": StandardEOMPowerplant,
    "standard_profile_eom_powerplant": StandardProfileEOMPowerplant,
    "standard_pcrm_powerplant": StandardPCRMPowerplant,
    "standard_ncrm_powerplant": StandardNCRMPowerplant,
    "standard_eom_storage": StandardEOMStorage,
    "standard_pcrm_storage": StandardPCRMStorage,
    "standard_ncrm_storage": StandardNCRMStorage,
    "dmas_powerplant": DmasPowerplantStrategy,
    "dmas_storage": DmasStorageStrategy,
    "misc_otc": OTCStrategy,
    "misc_manual": SimpleManualTerminalStrategy,
}

try:
    from assume.strategies.learning_advanced_orders import (
        LearningProfileEOMPowerplant,
    )
    from assume.strategies.learning_strategies import (
        LearningEOMPowerplant,
        LearningEOMStorage,
    )

    bidding_strategies["learning_eom_powerplant"] = LearningEOMPowerplant
    bidding_strategies["learning_eom_storage"] = LearningEOMStorage
    bidding_strategies["learning_profile_eom_powerplant"] = LearningProfileEOMPowerplant

except ImportError:
    pass
