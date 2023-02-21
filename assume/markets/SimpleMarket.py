import asyncio
from mango import RoleAgent, create_container, Role
from mango.util.clock import ExternalClock
from mango.messages.message import Performatives
from datetime import datetime
from dateutil import rrule
import pandas as pd
import numpy as np
import logging
from dateutil.parser import parse
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import TypedDict

logger = logging.getLogger(__name__)


class SimpleBid(TypedDict):
    price: float
    volume: float


class OneSidedMarketRole(Role):
    def __init__(self, demand=1000, receiver_ids=[]):
        super().__init__()
        self.demand = demand
        self.bids = []
        self.receiver_ids = receiver_ids

    def setup(self):
        self.context.results = []
        self.context.demands = []
        self.context.receiver_ids = self.receiver_ids
        self.context.demand = self.demand
        start = parse('202301010000')

        self.context.subscribe_message(
            self, self.handle_message, lambda content, meta: isinstance(content, dict)
        )
        # market acts every 15 minutes
        recurrency = rrule.rrule(rrule.MINUTELY, interval=15, dtstart=start)
        self.context.schedule_periodic_task(coroutine_func=self.clear_market, delay=900)

    async def clear_market(self):
        time = datetime.fromtimestamp(self.context.current_timestamp)
        i = time.hour + time.minute/60
        df = pd.DataFrame.from_dict(self.bids)
        self.bids = []
        price = 0
        demand = self.context.demand + 0.6 * self.context.demand*np.sin(i*np.pi/12)
        if not df.empty:
            # simple merit order calculation
            df = df.sort_values('price')
            df['cumsum'] = df['volume'].cumsum()
            filtered = df[df['cumsum'] >= demand]
            if filtered.empty:
                # demand could not be matched
                price = 100
            else:
                price = filtered['price'].values[0]
        self.context.results.append(price)
        self.context.demands.append(demand)
        acl_metadata = {
            'performative': Performatives.inform,
            'sender_id': self.context.aid,
            'sender_addr': self.context.addr,
            'conversation_id': 'conversation01'
        }
        for receiver_addr, receiver_id in self.context.receiver_ids:
            await self.context.send_acl_message(receiver_addr=receiver_addr,
                                                receiver_id=receiver_id,
                                                acl_metadata=acl_metadata,
                                                content={'message': f'Current time is {time}',
                                                         'data': df,
                                                         'price': price})

    def handle_message(self, content, meta):
        # content is SimpleBid
        content['sender_id']= meta['sender_id']
        self.bids.append(content)

    async def on_stop(self):
        logger.info(self.context.results)
        fig, ax1 = plt.subplots()
        plt.title('Simulation Results')
        ax1.plot(self.context.results, label='price')
        ax2 = ax1.twinx()
        ax2.plot(self.context.demands, label='demand', c='r')
        ax1.legend(loc='lower left', bbox_to_anchor= (0.8, 0.06), frameon=False)
        ax2.legend(loc='lower left', bbox_to_anchor= (0.8, 0.01), frameon=False)
        #plt.savefig('result.png')
        plt.show()


class BiddingRole(Role):
    def __init__(self, receiver_addr, receiver_id, volume=100, price=0.05):
        super().__init__()
        self.receiver_addr = receiver_addr
        self.receiver_id = receiver_id
        self.start = parse('202301010000')
        self.volume = volume
        self.price = price

    def setup(self):
        self.context.volume = self.volume
        self.context.price = self.price
        self.context.subscribe_message(
            self, self.handle_message, lambda content, meta: True
        )

    def handle_message(self, content, meta):
        # print(f'Received a message with the following content: {content}.')
        self.context.schedule_instant_task(coroutine=self.set_bids())

    async def set_bids(self):
        price = self.context.price + 0.01 * self.context.price * np.random.random()

        acl_metadata = {
            'performative': Performatives.inform,
            'sender_id': self.context.aid,
            'sender_addr': self.context.addr,
            'conversation_id': 'conversation01'
        }
        await self.context.send_acl_message(receiver_addr=self.receiver_addr,
                                            receiver_id=self.receiver_id,
                                            acl_metadata=acl_metadata,
                                            content={'price': price, 'volume': self.context.volume}
                                            )


async def main(start):
    clock = ExternalClock(start_time=start.timestamp())
    from mango.messages.codecs import JSON, PROTOBUF

    # works
    addr = [('127.0.0.1', 5555), ('127.0.0.1', 5556)]
    
    containers = []
    for ad in addr:
        c = await create_container(addr=ad, clock=clock, codec=PROTOBUF(generic_serializer=True))
        containers.append(c)
    market = RoleAgent(c)
    agents = []
    receiver_ids = []
    for i in range(17):
        ad = addr[i%len(addr)]
        c = containers[i%len(addr)]
        agent = RoleAgent(c)
        agent.add_role(BiddingRole(market.context.addr, market.aid, price=0.05*(i%9)))
        agents.append(agent)
        receiver_ids.append((ad, agent.aid))
    market.add_role(OneSidedMarketRole(demand=1000, receiver_ids=receiver_ids))

    if isinstance(clock, ExternalClock):
        for i in tqdm(range(1000)):
            await asyncio.sleep(0.01)
            clock.set_time(clock.time + 300)
    for c in containers:
        await c.shutdown()

if __name__ == '__main__':
    start = parse('202301010000')
    asyncio.run(main(start))
