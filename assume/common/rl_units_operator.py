import logging
from datetime import datetime
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import torch as th
import numpy as np

import pandas as pd
from mango import Role
from mango.messages.message import Performatives

from assume.common.market_objects import (
    ClearingMessage,
    MarketConfig,
    OpeningMessage,
    Order,
    Orderbook,
)
from assume.common.utils import aggregate_step_amount
from assume.strategies import BaseStrategy
from assume.units import BaseUnit
from assume.common import UnitsOperator

logger = logging.getLogger(__name__)


class RL_UnitsOperator(UnitsOperator):
    def __init__(
        self,
        available_markets: list[MarketConfig],
        opt_portfolio: tuple[bool, BaseStrategy] = None,
        world=None,
    ):
        super().__init__()

        self.bids_map = {}
        self.available_markets = available_markets
        self.registered_markets: dict[str, MarketConfig] = {}
        self.last_sent_dispatch = 0

        if opt_portfolio is None:
            self.use_portfolio_opt = False
            self.portfolio_strategy = None
        else:
            self.use_portfolio_opt = opt_portfolio[0]
            self.portfolio_strategy = opt_portfolio[1]

        self.valid_orders = []
        self.units: dict[str, BaseUnit] = {}

    def setup(self):
        self.id = self.context.aid
        self.context.subscribe_message(
            self,
            self.handle_opening,
            lambda content, meta: content.get("context") == "opening",
        )

        self.context.subscribe_message(
            self,
            self.handle_market_feedback,
            lambda content, meta: content.get("context") == "clearing",
        )

        for market in self.available_markets:
            if self.participate(market):
                self.register_market(market)
                self.registered_markets[market.name] = market



    def handle_market_feedback(self, content: ClearingMessage, meta: dict[str, str]):
        logger.debug(f"got market result: {content}")
        orderbook: Orderbook = content["orderbook"]
        for order in orderbook:
            order["market_id"] = content["market_id"]
            # map bid id to unit id
            order["unit_id"] = self.bids_map[order["bid_id"]]
        self.valid_orders.extend(orderbook)
        marketconfig = self.registered_markets[content["market_id"]]
        self.set_unit_dispatch(orderbook, marketconfig)
        self.write_actual_dispatch()

#TODO send back market feedback 
    def set_unit_dispatch(self, orderbook, marketconfig):
        """
        feeds the current market result back to the units
        this does not respect bids from multiple markets
        for the same time period
        """
        orderbook = list(sorted(orderbook, key=itemgetter("unit_id")))
        for unit_id, orders in groupby(orderbook, itemgetter("unit_id")):
            orders_l = list(orders)
            total_power = sum(map(itemgetter("volume"), orders_l))
            dispatch_plan = {"total_power": total_power}
            self.units[unit_id].set_dispatch_plan(
                dispatch_plan=dispatch_plan,
                start=orderbook[0]["start_time"],
                end=orderbook[0]["end_time"],
                product_type=marketconfig.product_type,
            )

    def write_actual_dispatch(self):
        """
        sends the actual aggregated dispatch curve
        works across multiple markets
        sends dispatch at or after it actually happens
        """
        last = self.last_sent_dispatch
        if self.context.current_timestamp == last:
            # stop if we exported at this time already
            return
        self.last_sent_dispatch = self.context.current_timestamp

        now = datetime.utcfromtimestamp(self.context.current_timestamp)
        start = datetime.utcfromtimestamp(last)

        market_dispatch = aggregate_step_amount(
            self.valid_orders, start, now, groupby=["market_id", "unit_id"]
        )
        unit_dispatch_dfs = []
        for unit_id, unit in self.units.items():
            data = pd.DataFrame(
                unit.execute_current_dispatch(start, now), columns=["power"]
            )
            data["unit"] = unit_id
            unit_dispatch_dfs.append(data)
        unit_dispatch = pd.concat(unit_dispatch_dfs)

        db_aid = self.context.data_dict.get("output_agent_id")
        db_addr = self.context.data_dict.get("output_agent_addr")
        if db_aid and db_addr:
            self.context.schedule_instant_acl_message(
                receiver_id=db_aid,
                receiver_addr=db_addr,
                content={
                    "context": "write_results",
                    "type": "market_dispatch",
                    "data": market_dispatch,
                },
            )
            if unit_dispatch_dfs:
                unit_dispatch = pd.concat(unit_dispatch_dfs)
                self.context.schedule_instant_acl_message(
                    receiver_id=db_aid,
                    receiver_addr=db_addr,
                    content={
                        "context": "write_results",
                        "type": "unit_dispatch",
                        "data": unit_dispatch,
                    },
                )

        self.valid_orders = list(
            filter(lambda x: x["end_time"] >= now, self.valid_orders)
        )

    async def submit_bids(self, opening: OpeningMessage):
        """
        formulates an orderbook and sends it to the market.
        This will handle optional portfolio processing

        Return:
        """

        products = opening["products"]
        market = self.registered_markets[opening["market_id"]]
        logger.debug(f"setting bids for {market.name}")
        orderbook = await self.formulate_bids(market, products)
        acl_metadata = {
            "performative": Performatives.inform,
            "sender_id": self.context.aid,
            "sender_addr": self.context.addr,
            "conversation_id": "conversation01",
        }
        await self.context.send_acl_message(
            content={
                "context": "submit_bids",
                "market": market.name,
                "orderbook": orderbook,
            },
            receiver_addr=market.addr,
            receiver_id=market.aid,
            acl_metadata=acl_metadata,
        )

 

    def initialize(self, t=0):
        if self.rl_agent:
            self.create_obs(t)

        for unit in self.units.items():
            unit.reset()


    async def formulate_bids(self, market: MarketConfig, products: list[tuple]):
        """
        Takes information from all units that the unit operator manages and
        formulates the bid to the market from that according to the bidding strategy.

        Return: OrderBook that is submitted as a bid to the market
        """

        orderbook: Orderbook = []

        # the given products just became available on our market
        # and we need to provide bids
        # [whole_next_hour, quarter1, quarter2, quarter3, quarter4]
        # algorithm should buy as much baseload as possible, then add up with quarters
        sorted_products = sorted(products, key=lambda p: (p[0] - p[1], p[0]))

        for product in sorted_products:

            order: Order = {
                "start_time": product[0],
                "end_time": product[1],
                "only_hours": product[2],
                "agent_id": (self.context.addr, self.context.aid),
            }

            actions = th.zeros(
            size=(len(self.units), self.world.act_dim),
            device=self.world.device,
            )

            for unit_id, unit in self.units.items():
                # take price from bidding strategy
                actions[unit_id, :] = unit.calculate_bids(
                    market_config=market,
                    product_tuple=product,
                )

            #convert actions from GPU onto CPU so that it is usable for the rest of the simulation
            #done for all actions once since this is a rather lengthly process
            actions = actions.clamp(-1, 1)
            actions = actions.squeeze().cpu().numpy()
            actions = actions.reshape(len(self.units), -1)

            if np.isnan(actions).any():
                raise ValueError("A NaN action happened.")


            for unit_id, unit in self.units.items():
                
                #append must run bid to oder book
                price = actions[unit_id, :].min() * unit.max_price
                volume = unit.minPower

                if market.volume_tick:
                    volume = round(volume / market.volume_tick)
                if market.price_tick:
                    price = round(price / market.price_tick)

                order_c = order.copy()
                order_c["volume"] = volume
                order_c["price"] = price
                order_c["bid_id"] = f"{unit_id}_{1}"
                orderbook.append(order_c)


                #append flexable bid to oder book
                price = actions[unit_id, :].max() * unit.max_price
                volume = unit.maxPower - unit.minPower

                if market.volume_tick:
                    volume = round(volume / market.volume_tick)
                if market.price_tick:
                    price = round(price / market.price_tick)

                order_c = order.copy()
                order_c["volume"] = volume
                order_c["price"] = price
                order_c["bid_id"] = f"{unit_id}_{2}"
                orderbook.append(order_c)

                self.bids_map[order_c["bid_id"]] = unit_id

        return orderbook


    def step(self):
        self.create_obs(self.world.currstep + 1)

        for unit_id, unit in self.units.items():
            unit.step(self.obs)

    # in rl_units operator in ASSUME
    # TODO consider that the last forecast_length time steps cant be used
    # TODO enable difference between actual load realisation and residual load forecast
    def create_obs(self):
        start = world.start
        end=world.end
        now= datetime.utcfromtimestamp(self.context.current_timestamp)
        delta_t=now-start #in metric of market
        obs = []
        forecast_len = 30 #in metric of market

        if delta_t < forecast_len:
            obs.extend(self.context.data_dict.get("res_demand_forecast")[-forecast_len + delta_t :])
            obs.extend(self.context.data_dict.get("res_demand_forecast")[: delta_t + forecast_len])

            obs.extend(self.context.data_dict.get("price_forecast")[-forecast_len + delta_t :])
            obs.extend(self.context.data_dict.get("price_forecast")[: delta_t + forecast_len])

        elif delta_t < (end-start) - forecast_len:
            obs.extend(self.context.data_dict.get("res_demand_forecast")[delta_t - forecast_len : delta_t])
            obs.extend(self.context.data_dict.get("res_demand_forecast")[delta_t : delta_t + forecast_len])

            obs.extend(self.context.data_dict.get("price_forecast")[delta_t - forecast_len : delta_t])
            obs.extend(self.context.data_dict.get("price_forecast")[delta_t : delta_t + forecast_len])

        else:
            obs.extend(self.context.data_dict.get("res_demand_forecast")[delta_t - forecast_len :])
            obs.extend(
                self.context.data_dict.get("res_demand_forecast")[: forecast_len * 2 - len(obs)]
            )

            obs.extend(self.context.data_dict.get("price_forecast")[t - forecast_len :])
            obs.extend(self.context.data_dict.get("price_forecast")[: forecast_len * 4 - len(obs)])

        self.obs = obs

#everything below is TODO

    def collect_experience(self):
        total_units = self.rl_algorithm.n_rl_agents
        obs = th.zeros((2, total_units, self.obs_dim), device=self.device)
        actions = th.zeros((total_units, self.act_dim), device=self.device)
        rewards = []

        for i, pp in enumerate(self.rl_algorithm.rl_agents):
            obs[0][i] = pp.curr_experience[0]
            obs[1][i] = pp.curr_experience[1]
            actions[i] = pp.curr_experience[2]
            rewards.append(pp.curr_experience[3])

        return obs, actions, rewards
    
    #this function is used in flexable but I want this in the RL Units operator
            if self.training:
                obs, actions, rewards = self.collect_experience()
                self.rl_algorithm.buffer.add(obs, actions, rewards)
                self.rl_algorithm.update_policy()

    # TODO check wiritng and if it should be in units operator and tensorboard
    def extract_rl_episode_info(self):
        total_rewards = 0
        total_profits = 0
        total_regrets = 0

        for unit in self.rl_powerplants:
            if unit.learning:
                total_rewards += sum(unit.rewards)
                total_profits += sum(unit.profits)
                total_regrets += sum(unit.regrets)

        for unit in self.rl_storages:
            if unit.learning:
                total_rewards += sum(unit.rewards)
                total_rewards += sum(unit.rewards)
                total_profits += sum(unit.profits)

        total_rl_units = (
            self.rl_algorithm.n_rl_agents
            if self.rl_algorithm is not None
            else len(self.rl_powerplants + self.rl_storages)
        )
        average_reward = total_rewards / total_rl_units / len(self.snapshots)
        average_profit = total_profits / total_rl_units / len(self.snapshots)
        average_regret = total_regrets / total_rl_units / len(self.snapshots)

        if self.training:
            self.tensorboard_writer.add_scalar(
                "Train/Average Reward", average_reward, self.episodes_done
            )
            self.tensorboard_writer.add_scalar(
                "Train/Average Profit", average_profit, self.episodes_done
            )
            self.tensorboard_writer.add_scalar(
                "Train/Average Regret", average_regret, self.episodes_done
            )
        else:
            self.rl_eval_rewards.append(average_reward)
            self.rl_eval_profits.append(average_profit)
            self.rl_eval_regrets.append(average_regret)

            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(
                    "Eval/Average Reward", average_reward, self.eval_episodes_done
                )
                self.tensorboard_writer.add_scalar(
                    "Eval/Average Profit", average_profit, self.eval_episodes_done
                )
                self.tensorboard_writer.add_scalar(
                    "Eval/Average Regret", average_regret, self.eval_episodes_done
                )
