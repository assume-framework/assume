import time
from multiprocessing import Process

from distributed_simulation.main import agent, agent_adresses, manager, tcp_host
from world_script import init

from assume import World


def run_distrib(n=1):
    man = Process(target=manager)
    for i in range(n - 1):
        agent_adresses.append((tcp_host, 9099 + i))
    ags = []
    for i in range(n):
        ag = Process(target=agent, args=(i, n))
        ags.append(ag)

    man.start()

    time.sleep(0.3)
    for ag in ags:
        ag.start()

    man.join()
    for ag in ags:
        ag.join()


def run_sync(n=1):
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    db_uri = ""
    world = World(database_uri=db_uri)
    world.loop.run_until_complete(init(world, n))
    world.run()


if __name__ == "__main__":
    results = []
    for n in [2, 3]:
        print("start simulation with", n)
        t = time.time()
        run_distrib(n)
        duration = time.time() - t
        results.append((n, "distrib", duration))

        t = time.time()
        run_sync(n)
        duration = time.time() - t
        results.append((n, "sync", duration))
    print(results)
    import json
    #results = [(1, 'distrib', 10.555368423461914), (1, 'sync', 8.060883045196533), (2, 'distrib', 10.997570276260376), (2, 'sync', 11.374260663986206), (3, 'distrib', 11.478757619857788), (3, 'sync', 14.855780601501465), (4, 'distrib', 12.92294454574585), (4, 'sync', 18.389983415603638), (5, 'distrib', 18.909234523773193), (5, 'sync', 24.365375518798828), (10, 'distrib', 27.40151834487915), (10, 'sync', 40.98110389709473), (20, 'distrib', 60.09907078742981), (20, 'sync', 74.69065427780151), (30, 'distrib', 101.84901762008667), (30, 'sync', 110.01267266273499), (40, 'distrib', 149.119637966156), (40, 'sync', 145.8321897983551), (50, 'distrib', 188.8543348312378), (50, 'sync', 189.90265107154846)]
    #          [(1, 'distrib', 12.669503450393677), (1, 'sync', 6.627783536911011), (2, 'distrib', 12.189964771270752), (2, 'sync', 9.727921962738037), (3, 'distrib', 11.60510540008545), (3, 'sync', 12.869911670684814), (4, 'distrib', 14.952376127243042), (4, 'sync', 16.52760100364685), (5, 'distrib', 15.294515371322632), (5, 'sync', 19.758689403533936)]
    # with factorial: [(2, 'distrib', 101.847238779068), (2, 'sync', 137.1397886276245), (3, 'distrib', 133.3344464302063), (3, 'sync', 183.33061480522156)]

    with open("runtime_tests.json", "w") as f:
        json.dump(results, f, indent=4)

    import json
    with open("runtime_tests.json", "r") as f:
        results = json.load(f)
    import pandas as pd

    df = pd.read_json("runtime_tests.json")
    df.columns = ["n", "type", "time"]

    # Separate data for each type
    distrib_data = df[df['type'] == 'distrib']
    sync_data = df[df['type'] == 'sync']

    import matplotlib.pyplot as plt
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(distrib_data['n'], distrib_data['time'], label='distrib')
    plt.plot(sync_data['n'], sync_data['time'], label='sync')
    plt.xlabel('n')
    plt.ylabel('time')
    plt.title('Time vs. n')
    plt.legend()
    plt.grid(True)
    plt.savefig("scalability.svg")
    plt.show()