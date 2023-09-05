from assume.common.market_objects import market_mechanism

from .all_or_nothing import pay_as_bid_aon, pay_as_clear_aon
from .complex_clearing import pay_as_clear_complex
from .complex_clearing_dmas import complex_clearing_dmas
from .nodal_pricing import nodal_pricing_pyomo
from .simple import pay_as_bid, pay_as_clear

clearing_mechanisms: dict[str, market_mechanism] = {
    "pay_as_clear": pay_as_clear,
    "pay_as_bid": pay_as_bid,
    "pay_as_bid_all_or_nothing": pay_as_bid_aon,
    "pay_as_clear_all_or_nothing": pay_as_clear_aon,
    "nodal_pricing_pyomo": nodal_pricing_pyomo,
    "pay_as_clear_complex": pay_as_clear_complex,
    # "pay_as_clear_complex_dmas": complex_clearing_dmas,
}
