# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import random
from datetime import timedelta
from itertools import groupby
from operator import itemgetter

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.markets.base_market import MarketRole
from assume.markets.clearing_algorithms.simple import PayAsClearRole

logger = logging.getLogger(__name__)

class MitigatedRole(PayAsClearRole):
    def __init__(self, marketconfig: MarketConfig):
        
        super().__init__(marketconfig)
        self.impact_threshold = marketconfig.params.get("impact_threshold", lambda price: min(price + 100, price * 2))


    def clear(
        self, orderbook: Orderbook, market_products
    ) -> tuple[Orderbook, Orderbook, list[dict]]:
        """
        Performs electricity market clearing using a pay-as-clear mechanism. This means that the clearing price is the
        highest price that is still accepted. The clearing price is the same for all accepted orders.

        Args:
            orderbook (Orderbook): the orders to be cleared as an orderbook
            market_products (list[MarketProduct]): the list of products which are cleared in this clearing

        Returns:
            tuple: accepted orderbook, rejected orderbook and clearing meta data
        """
        # Bids with status
        not_mitigated_orderbook = [bid for bid in orderbook if bid.get('status', '') != 'mitigated']
        mitigated_orderbook = [bid for bid in orderbook if bid.get('status', '') != 'not_mitigated']

        not_mitigated_market_outcome = super().clear(not_mitigated_orderbook, market_products)
        mitigated_market_outcome = super().clear(mitigated_orderbook, market_products)
        #TODO: instead add marginal cost in the dict with the bids
        #TODO: install pre-commit
        
        accepted_orders = Orderbook()
        rejected_orders = Orderbook()
        meta = []
        flows = not_mitigated_market_outcome[3]  # flows are not considered in market-wide mitigation
        
        #TODO: check if this is working
        for i, product in enumerate(market_products):
            # if the clearing price of the not mitigated orderbook is above the impact threshold, then mitigate
            if not_mitigated_market_outcome[2][i]['max_price'] > self.impact_threshold(mitigated_market_outcome[2][i]['max_price']):
                accepted_orders.append(mitigated_market_outcome[0][i])
                rejected_orders.append(mitigated_market_outcome[1][i])
                meta.append(mitigated_market_outcome[2][i])
        
            else:
                accepted_orders.append(mitigated_market_outcome[0][i])
                rejected_orders.append(mitigated_market_outcome[1][i])
                meta.append(mitigated_market_outcome[2][i])
            
            #in both cases, extend the metadata to include not mitigated price for comparison
            meta[-1]['not_mitigated_price'] = not_mitigated_market_outcome[2][i]['max_price']

        return accepted_orders, rejected_orders, meta, flows