

'''
## Understanding clearing mechanisms in normal and edge cases

- 1 node
- 200 MW elastic demand with
    - y-intercept of the marginal utility at 20 €/MWh
    - slope: -0.1 (so the willingness to pay for the last MWh is ca. 0 €.)
- 200 MW elastic generation

marginal utility of demand and marginal cost of supply side can both take different forms:
- continous (linear)
- stepwise (blockwise / stairs)

clearing can be based on
- optimization with PyPSA
- complex clearing with ASSUME (optimization using pyomo)
- simple clearing with ASSUME (loop)

Edge cases are
- supply shortage (demand is price setting)
- marginal cost curve intersects marginal utility curve in a vertical segment (when blockwise curves)
'''
#%%
import pypsa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import assume

from dateutil import rrule as rr
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.markets.clearing_algorithms.simple import PayAsClearRole, PayAsBidRole
from assume.markets.clearing_algorithms.complex_clearing import ComplexClearingRole
from assume.markets.clearing_algorithms.complex_clearing_dmas import ComplexDmasClearingRole
from assume.common.utils import get_available_products

from datetime import datetime, timedelta

#%%
def plot_cost_utility_mc_mu(n, mc_type=int, mcq_round=3, make_demand_slope_negative=True):
    fig, ax = plt.subplots(ncols=2)
    p_max = n.generators.p_nom.max()
    p_gen = np.linspace(0, p_max,100)       # generation
    p_load = np.linspace(0, p_max, 100)     # load
    p_ax = p_gen                            # x-axis
    ax[0].set_title("Cost / Utility")
    ax[0].set_ylabel('€')
    ax[0].set_xlabel('MWh')
    ax[1].set_ylabel('€/MWh')
    ax[1].set_xlabel('MWh')
    ax[1].set_title("Marginal Cost / Marginal Utility")
    ax[0].plot(p_ax, [n.generators.marginal_cost['generator'] * p + n.generators.marginal_cost_quadratic['generator'] * p**2 for p in p_gen],
               label=f"generation {n.generators.loc['generator', 'marginal_cost'].astype(mc_type)} €/MWh * x \n+ {round(n.generators.loc['generator', 'marginal_cost_quadratic'], mcq_round)} €/MWh^2 * x^2")
    ax[1].plot(p_ax, [n.generators.marginal_cost['generator'] + 2 * n.generators.marginal_cost_quadratic['generator'] * p for p in p_gen],
               label=f"generation {n.generators.loc['generator', 'marginal_cost'].astype(mc_type)} €/MWh \n+ {round(2*n.generators.loc['generator', 'marginal_cost_quadratic'], mcq_round)} €/MWh * x")
    if make_demand_slope_negative:
        # y = ax - bx^2
        ax[0].plot(p_ax, [n.generators.marginal_cost['load'] * p - n.generators.marginal_cost_quadratic['load'] * p**2 for p in p_load],
               label=f"load {n.generators.loc['load', 'marginal_cost'].astype(mc_type)} €/MWh * x \n- {round(n.generators.loc['load', 'marginal_cost_quadratic'], mcq_round)} €/MWh^2 * x^2")
    
        # y = a - bx
        ax[1].plot(p_ax, [n.generators.marginal_cost['load'] - 2 * n.generators.marginal_cost_quadratic['load'] * p for p in p_load],
                label=f"load {n.generators.loc['load', 'marginal_cost'].astype(mc_type)} €/MWh \n- {round(2*n.generators.loc['load', 'marginal_cost_quadratic'], mcq_round)} €/MWh * x")
    else:
        # y = ax + bx^2
        ax[0].plot(p_ax, [n.generators.marginal_cost['load'] * p + n.generators.marginal_cost_quadratic['load'] * p**2 for p in p_load],
               label=f"load {n.generators.loc['load', 'marginal_cost'].astype(mc_type)} €/MWh * x \n+ {round(n.generators.loc['load', 'marginal_cost_quadratic'], mcq_round)} €/MWh^2 * x^2")
    
        # y = a + bx
        ax[1].plot(p_ax, [n.generators.marginal_cost['load'] + 2 * n.generators.marginal_cost_quadratic['load'] * p for p in p_load],
                label=f"load {n.generators.loc['load', 'marginal_cost'].astype(mc_type)} €/MWh \n+ {round(2*n.generators.loc['load', 'marginal_cost_quadratic'], mcq_round)} €/MWh * x")
    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()
    fig.suptitle("Cost / Utility and Marginal Cost / Marginal Utility")
    plt.tight_layout()
    return fig, ax

#%% -- Case 0: continuous marginal utility and marginal cost curves
# trivial case and in line with economic theory
n = pypsa.Network()

n.add("Carrier", "AC")
n.add("Bus", "main bus", carrier="AC")

#
n.add("Generator",
      name="generator",
      bus="main bus",
      p_nom=200,
      marginal_cost=0,
      marginal_cost_quadratic=1/30/2,
      carrier="AC",
      )
n.add("Generator",
      name="load",
      bus="main bus",
      p_nom=200,
      p_min_pu=-1,
      p_max_pu=0,
      marginal_cost=20,
      marginal_cost_quadratic=0.1/2,
      sign=1,
      carrier="AC",
      )
n.optimize(assign_all_duals=True)
fig, ax = plot_cost_utility_mc_mu(n, make_demand_slope_negative=True)

assert abs(round(n.objective, 0)) == (5*150/2) + (15*150/2)
print("PyPSA Optimization dispatched volume:", n.statistics.supply().sum())
print("PyPSA Optimization market price:", n.buses_t.marginal_price.loc['now', 'main bus'])

#%%
def plot_merit_order(bids, fig=None, ax=None, ls='-', bool_legend=True):
    demand_label = None
    supply_label = None
    x_label = None
    y_label = None
    title = None
    
    if ax is None:
        fig, ax = plt.subplots()
        demand_label = 'Demand Bids'
        supply_label = 'Supply Bids'
        x_label = 'Cumulative Power [MW]'
        y_label = 'Price [€/MWh]'
        title = 'Merit Order Curve'
    supply_bids = bids[bids['volume'] > 0]
    demand_bids = bids[bids['volume'] < 0]
    demand_bids['volume'] *= -1
    supply_bids = supply_bids.sort_values(by='price')
    demand_bids = demand_bids.sort_values(by='price', ascending=False)
    ax.stairs(supply_bids['price'],
                np.array([0] + supply_bids['volume'].cumsum().tolist()),
            label=supply_label,
            baseline=None,
            linestyle=ls,
            )
    ax.stairs(demand_bids['price'],
              np.array([0] + demand_bids['volume'].cumsum().tolist()),
        label=demand_label,
        baseline=None,
        linestyle=ls,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    # set grid behind the plotting data
    ax.set_axisbelow(True)
    ax.grid()
    if bool_legend:
        ax.legend()
    plt.tight_layout()
    return fig, ax

big_fig, big_ax = plt.subplots(ncols=2, nrows=3, sharey=True, sharex=True)
#%% -- Case 1: stepwise (blockwise) marginal utility and marginal cost curves
# get the supply offers as approximation of the marginal cost curve in 10 steps
n_supply_bids = 2
v_supply_bid = 200 / n_supply_bids
n_bids_demand = 3
v_bid_demand = -1 * 200 / n_bids_demand
demand_mc = 20
supply_mc = 0
demand_mcq = 0.15
supply_mcq = 1/20

def build_sorted_bids(n_supply_bids, v_supply_bid,
                      n_bids_demand, v_bid_demand,
                      demand_mc=20, demand_mcq=0.1,
                      supply_mc=0, supply_mcq=1/30):
    # supply
    marginal_costs = []
    for i in range(n_supply_bids):
        mc = supply_mc + i * v_supply_bid * supply_mcq
        marginal_costs.append(mc)
    sorted_supply = pd.Series(marginal_costs).sort_values().index
    sorted_mc = pd.Series(marginal_costs).loc[sorted_supply]
    sorted_volume_supply = pd.Series([v_supply_bid] * n_supply_bids, index=sorted_supply)

    # demand

    marginal_utility = []
    for i in range(n_bids_demand):
        mu = demand_mc + i * v_bid_demand * demand_mcq # + becaus volume is negative
        marginal_utility.append(mu)
    sorted_demand = pd.Series(marginal_utility).sort_values(ascending=False).index
    sorted_mu = pd.Series(marginal_utility).loc[sorted_demand]
    sorted_volume_demand = pd.Series([v_bid_demand] * n_bids_demand, index=sorted_demand)
    return sorted_mc, sorted_volume_supply, sorted_mu, sorted_volume_demand

sorted_mc, sorted_volume_supply, sorted_mu, sorted_volume_demand = build_sorted_bids(n_supply_bids, v_supply_bid,
                                                                              n_bids_demand, v_bid_demand,
                                                                              demand_mc, demand_mcq,
                                                                              supply_mc, supply_mcq)
def build_pypsa_stepwise(n_supply_bids, v_supply_bid, sorted_mc,
                         n_bids_demand, v_bid_demand, sorted_mu):
    n = pypsa.Network()
    n.add("Carrier", "AC")
    n.add("Bus", "main bus", carrier="AC")
    for i in range(n_supply_bids):
        n.add("Generator",
              name=f"sup_{i}",
              bus="main bus",
              p_nom=v_supply_bid,
              marginal_cost=sorted_mc.iloc[i],
              carrier="AC",
              )
    for i in range(n_bids_demand):
        n.add("Generator",
              name=f"dem_{i}",
              bus="main bus",
              p_nom=-v_bid_demand,
              p_min_pu=-1,
              p_max_pu=0,
              marginal_cost=sorted_mu.iloc[i],
              sign=1,
              carrier="AC",
              )
    return n

def reset_orderbook(orderbook):
    for bid in orderbook:
        bid['accepted_volume'] = 0
        bid['accepted_price'] = 0
    return orderbook

n = build_pypsa_stepwise(n_supply_bids, v_supply_bid, sorted_mc,
                         n_bids_demand, v_bid_demand, sorted_mu)

n.optimize(assign_all_duals=True)

# --- Solve with ASSUME simple market clearing
def build_assume_orderbook_marketconfig(sorted_mc, sorted_volume_supply,
                         sorted_mu, sorted_volume_demand):
    start = pd.Timestamp('2024-01-01 00:00:00')
    end = pd.Timestamp('2024-01-01 01:00:00')

    supply_bids = (
        pd.DataFrame(
            {   
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "node": 'node0',
                "price": sorted_mc,
                "volume": sorted_volume_supply,
                "bid_type" : "SB",
                }
                )
        .reset_index()
        .rename(columns={"index": "bid_id"})
    )
    # add sup_ to the bid ids
    supply_bids['bid_id'] = 'sup_' + supply_bids['bid_id'].astype(str)

    demand_bids = (
        pd.DataFrame(
            {   "start_time": start,
                "end_time": end,
                "only_hours": None,
                "node": 'node0',
                "price": sorted_mu,
                "volume": sorted_volume_demand,
                "bid_type" : "SB",
            }
        )
        .reset_index()
        .rename(columns={"index": "bid_id"})
    )
    # add dem_ to the bid ids
    demand_bids['bid_id'] = 'dem_' + demand_bids['bid_id'].astype(str)
    # create an orderbook containing all supply offers and demand bids
    orderbook = []
    orderbook.extend(supply_bids.to_dict('records'))
    orderbook.extend(demand_bids.to_dict('records'))

    marketconfig = MarketConfig(
        market_id="test",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
        additional_fields=["node"],
        opening_hours=rr.rrule(
            rr.HOURLY,
            dtstart=datetime(2024, 1, 1, 0, 0),
            until=datetime(2024, 1, 1, 1, 0),
            cache=True,
        ),
        opening_duration=timedelta(hours=1),
        volume_unit="MW",
        volume_tick=0.1,
        maximum_bid_volume=None,
        price_unit="€/MW",
        market_mechanism="PayAsClear",
    )

    mps = get_available_products(marketconfig.market_products, pd.Timestamp(start) - pd.Timedelta('1h'))
    return orderbook, marketconfig, mps
orderbook, marketconfig, mps = build_assume_orderbook_marketconfig(sorted_mc, sorted_volume_supply,
                                                                   sorted_mu, sorted_volume_demand)
pac = PayAsClearRole(marketconfig)
accepted_s, rejected_s, meta_s, flows_s = pac.clear(orderbook, mps)

orderbook = reset_orderbook(orderbook)
pab = PayAsBidRole(marketconfig)
accepted_pb, rejected_pb, meta_pb, flows_pb = pab.clear(orderbook, mps)

orderbook = reset_orderbook(orderbook)
cc = ComplexClearingRole(marketconfig)
accepted_cc, rejected_cc, meta_cc, flows_cc = cc.clear(orderbook, mps)

def show_results(n, meta_s, meta_pb, meta_cc, orderbook):
    # orerbook is a list of dicts, convert to dataframe
    bids_assume = pd.DataFrame(orderbook)
    fig, ax = plot_merit_order(bids_assume)
    ax.set_title(f'Merit Order Curve')
    print("PyPSA Optimization dispatched volume:", n.statistics.supply().sum())
    print("ASSUME simple PayAsClear dispatched volume:", meta_s[0]['supply_volume'])
    print("ASSUME simple PayAsBid dispatched volume:", meta_pb[0]['supply_volume'])
    print("ASSUME Complex Clearing dispatched volume:", meta_cc[0]['supply_volume'])
    print("PyPSA Optimization market price:", n.buses_t.marginal_price.loc['now', 'main bus'])
    print("ASSUME simple PayAsClear market price:", meta_s[0]['price'])
    print("ASSUME simple PayAsBid max. market price:", meta_pb[0]['max_price'])
    print("ASSUME Complex Clearing market price:", meta_cc[0]['price'])
#
show_results(n, meta_s, meta_pb, meta_cc, orderbook)
# this makes perfect sense: the intersection is in a horizontal segment of the marginal cost curve
big_fig, big_ax[0,0] = plot_merit_order(pd.DataFrame(orderbook),
                                        fig=big_fig,
                                        ax=big_ax[0,0],
                                        bool_legend=False,
                                        )

#%% --- Case 2: intersection of demand and supply cost curves in a vertical segment of the marginal cost curve
n_supply_bids = 2
v_supply_bid = 200 / n_supply_bids
n_bids_demand = 3
v_bid_demand = -1 * 200 / n_bids_demand
supply_mcq = 1/6

sorted_mc, sorted_volume_supply, sorted_mu, sorted_volume_demand = build_sorted_bids(n_supply_bids, v_supply_bid,
                                                                              n_bids_demand, v_bid_demand,
                                                                              demand_mc, demand_mcq,
                                                                              supply_mc, supply_mcq)
n = build_pypsa_stepwise(n_supply_bids, v_supply_bid, sorted_mc,
                         n_bids_demand, v_bid_demand, sorted_mu)
n.optimize(assign_all_duals=True)

orderbook, marketconfig, mps = build_assume_orderbook_marketconfig(sorted_mc, sorted_volume_supply,
                                                                   sorted_mu, sorted_volume_demand)


pac = PayAsClearRole(marketconfig)
accepted_s, rejected_s, meta_s, flows_s = pac.clear(orderbook, mps)

orderbook = reset_orderbook(orderbook)
pab = PayAsBidRole(marketconfig)
accepted_pb, rejected_pb, meta_pb, flows_pb = pab.clear(orderbook, mps)

orderbook = reset_orderbook(orderbook)
cc = ComplexClearingRole(marketconfig)
accepted_cc, rejected_cc, meta_cc, flows_cc = cc.clear(orderbook, mps)

show_results(n, meta_s, meta_pb, meta_cc, orderbook)
big_fig, big_ax[0,1] = plot_merit_order(pd.DataFrame(orderbook),
                                        fig=big_fig,
                                        ax=big_ax[0,1],
                                        bool_legend=False,
                                        )

#%% --- Case 3: horizontal overlap of supply and demand curves
n_supply_bids = 2
v_supply_bid = 200 / n_supply_bids
n_bids_demand = 3
v_bid_demand = -1 * 200 / n_bids_demand
supply_mcq = 1/10
demand_mcq = 3/20

sorted_mc, sorted_volume_supply, sorted_mu, sorted_volume_demand = build_sorted_bids(n_supply_bids, v_supply_bid,
                                                                              n_bids_demand, v_bid_demand,
                                                                              demand_mc, demand_mcq,
                                                                              supply_mc, supply_mcq)
n = build_pypsa_stepwise(n_supply_bids, v_supply_bid, sorted_mc,
                         n_bids_demand, v_bid_demand, sorted_mu)
n.optimize(assign_all_duals=True)
orderbook, marketconfig, mps = build_assume_orderbook_marketconfig(sorted_mc, sorted_volume_supply,
                                                                   sorted_mu, sorted_volume_demand)
pac = PayAsClearRole(marketconfig)
accepted_s, rejected_s, meta_s, flows_s = pac.clear(orderbook, mps)

orderbook = reset_orderbook(orderbook)
pab = PayAsBidRole(marketconfig)
accepted_pb, rejected_pb, meta_pb, flows_pb = pab.clear(orderbook, mps)

orderbook = reset_orderbook(orderbook)
cc = ComplexClearingRole(marketconfig)
accepted_cc, rejected_cc, meta_cc, flows_cc = cc.clear(orderbook, mps)
show_results(n, meta_s, meta_pb, meta_cc, orderbook)
# here Optimization and simple loop based clearing diverge
# this is to be expected, because the optimizer cannot see any benefit in matching more volume at zero welfare gain
# however, it should because serving more demand is a goal for itself
# and it is also done in EUPHEMIA (public description 18.12.2025, p. 62f)
big_fig, big_ax[1,0] = plot_merit_order(pd.DataFrame(orderbook),
                                        fig=big_fig,
                                        ax=big_ax[1,0],
                                        bool_legend=False,
                                        )


#%% --- Case 4: vertical overlap of supply and demand curves
n_supply_bids = 4
v_supply_bid = 200 / n_supply_bids
n_bids_demand = 2
v_bid_demand = -1 * 200 / n_bids_demand
demand_mcq = 0.1
supply_mcq = 0.1
supply_mc = 2.5
sorted_mc, sorted_volume_supply, sorted_mu, sorted_volume_demand = build_sorted_bids(n_supply_bids, v_supply_bid,
                                                                              n_bids_demand, v_bid_demand,
                                                                              demand_mc, demand_mcq,
                                                                              supply_mc, supply_mcq)
n = build_pypsa_stepwise(n_supply_bids, v_supply_bid, sorted_mc,
                         n_bids_demand, v_bid_demand, sorted_mu)
n.optimize(assign_all_duals=True)
orderbook, marketconfig, mps = build_assume_orderbook_marketconfig(sorted_mc, sorted_volume_supply,
                                                                   sorted_mu, sorted_volume_demand)

pac = PayAsClearRole(marketconfig)
accepted_s, rejected_s, meta_s, flows_s = pac.clear(orderbook, mps)

orderbook = reset_orderbook(orderbook)
pab = PayAsBidRole(marketconfig)
accepted_pb, rejected_pb, meta_pb, flows_pb = pab.clear(orderbook, mps)

orderbook = reset_orderbook(orderbook)
cc = ComplexClearingRole(marketconfig)
accepted_cc, rejected_cc, meta_cc, flows_cc = cc.clear(orderbook, mps)
show_results(n, meta_s, meta_pb, meta_cc, orderbook)
big_fig, big_ax[1,1] = plot_merit_order(pd.DataFrame(orderbook),
                                        fig=big_fig,
                                        ax=big_ax[1,1],
                                        bool_legend=False,
                                        )
#%% --- Case 5: demand shortage (supply is price setting)
n_supply_bids = 3
v_supply_bid = 200 / n_supply_bids
n_bids_demand = 2
v_bid_demand = -1 * 150 / n_bids_demand
demand_mcq = 1/10
supply_mcq = 15/200
demand_mc = 20
supply_mc = 0
sorted_mc, sorted_volume_supply, sorted_mu, sorted_volume_demand = build_sorted_bids(n_supply_bids, v_supply_bid,
                                                                              n_bids_demand, v_bid_demand,
                                                                              demand_mc, demand_mcq,
                                                                              supply_mc, supply_mcq)
n = build_pypsa_stepwise(n_supply_bids, v_supply_bid, sorted_mc,
                         n_bids_demand, v_bid_demand, sorted_mu)
n.optimize(assign_all_duals=True)
orderbook, marketconfig, mps = build_assume_orderbook_marketconfig(sorted_mc, sorted_volume_supply,
                                                                   sorted_mu, sorted_volume_demand)
pac = PayAsClearRole(marketconfig)
accepted_s, rejected_s, meta_s, flows_s = pac.clear(orderbook, mps)
orderbook = reset_orderbook(orderbook)
pab = PayAsBidRole(marketconfig)
accepted_pb, rejected_pb, meta_pb, flows_pb = pab.clear(orderbook, mps)
orderbook = reset_orderbook(orderbook)
cc = ComplexClearingRole(marketconfig)
accepted_cc, rejected_cc, meta_cc, flows_cc = cc.clear(orderbook, mps)
show_results(n, meta_s, meta_pb, meta_cc, orderbook)
big_fig, big_ax[2,0] = plot_merit_order(pd.DataFrame(orderbook),
                                        fig=big_fig,
                                        ax=big_ax[2,0],
                                        bool_legend=False,
                                        )

#%% --- Case 6: supply shortage (demand is price setting)
n_supply_bids = 2
v_supply_bid = 150 / n_supply_bids
n_bids_demand = 3
v_bid_demand = -1 * 200 / n_bids_demand
demand_mcq = 15/200
supply_mcq = 1/10
demand_mc = 20
supply_mc = 0
sorted_mc, sorted_volume_supply, sorted_mu, sorted_volume_demand = build_sorted_bids(n_supply_bids, v_supply_bid,
                                                                              n_bids_demand, v_bid_demand,
                                                                              demand_mc, demand_mcq,
                                                                              supply_mc, supply_mcq)
n = build_pypsa_stepwise(n_supply_bids, v_supply_bid, sorted_mc,
                         n_bids_demand, v_bid_demand, sorted_mu)
n.optimize(assign_all_duals=True)
orderbook, marketconfig, mps = build_assume_orderbook_marketconfig(sorted_mc, sorted_volume_supply,
                                                                   sorted_mu, sorted_volume_demand)
pac = PayAsClearRole(marketconfig)
accepted_s, rejected_s, meta_s, flows_s = pac.clear(orderbook, mps)
orderbook = reset_orderbook(orderbook)
pab = PayAsBidRole(marketconfig)
accepted_pb, rejected_pb, meta_pb, flows_pb = pab.clear(orderbook, mps)
orderbook = reset_orderbook(orderbook)
cc = ComplexClearingRole(marketconfig)
accepted_cc, rejected_cc, meta_cc, flows_cc = cc.clear(orderbook, mps)
show_results(n, meta_s, meta_pb, meta_cc, orderbook)
# ASSUME simple strategy fixes the market price at the last accepted supply bid (hard coded)
big_fig, big_ax[2,1] = plot_merit_order(pd.DataFrame(orderbook),
                                        fig=big_fig,
                                        ax=big_ax[2,1],
                                        bool_legend=False,
                                        )

#%% Finalize big figure
# big_fig.suptitle("Merit Order Curves of Different Clearing Cases", y=.02)
big_fig.tight_layout()
# set figure x and ylabels
big_ax[0,0].set_ylabel('Price [€/MWh]')
big_ax[1,0].set_ylabel('Price [€/MWh]')
big_ax[2,0].set_ylabel('Price [€/MWh]')
big_ax[2,0].set_xlabel('Cumulative Power [MW]')
big_ax[2,1].set_xlabel('Cumulative Power [MW]')

# annotate the case numbers to the axes in the top right of each subplot
cases = ['Case 1', 'Case 2', 'Case 3', 'Case 4', 'Case 5', 'Case 6']
for i in range(6):
    row = i // 2
    col = i % 2
    big_ax[row,col].text(0.95, 0.95, cases[i], transform=big_ax[row,col].transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right')
plt.show()
#%%
# plot last fig, ax again to then obtain the handles for the legend
fig, ax = plot_merit_order(pd.DataFrame(orderbook))
# add legend to the big figure
handles, labels = ax.get_legend_handles_labels()
big_fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=2)

#%%
import os
safe_path = 'C:\\Users\\Gunter\\Code\\Inconsistencies'
big_fig.savefig(os.path.join(safe_path, 'clearing_comparison_cases.png'), dpi=300, bbox_inches='tight')
big_fig.savefig(os.path.join(safe_path, 'clearing_comparison_cases.pdf'), dpi=300, bbox_inches='tight')

