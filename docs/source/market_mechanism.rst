


Market Mechanism
================

A Market Mechanism is used to execute the clearing, scheduled by the MarketRole in base_market.py

The method signature for the market_mechanism is given as::

  def clearing_mechanism_name(
    market_agent: MarketRole,
    market_products: list[MarketProduct],
  ):
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta: list[Meta] = []
    return accepted_orders, rejected_orders, meta

The :code:`market_mechanism` is called by the MarketRole, which is the agent that is responsible for the market.
It is called with the :code:`market_agent` and the :code:`market_products`, which are the products that are traded in the current opening of the market.
This gives maximum flexbility as it allows to access properties from the MarketRole directly.
The :code:`market_mechanism` returns a list of accepted orders, a list of rejected orders and a list of meta information (for each tradable market product or trading zone, if needed).
The meta information is used to store information about the clearing, e.g. the min and max price, the cleared demand volume and supply volume, as well as the information about the cleared product.

In the Market Mechanism, the MarketRole is available to access the market configuration with :code:`market_agent.marketconfig` and the available Orders from previous clearings through :code:`market_agent.all_orders`.
In the future, the MarketMechanism will be a class which contains the additional information like grid information without changing the MarketRole.<>
