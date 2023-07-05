import asyncio

from assume import World


def test_world():
    scenario = "example_01a"
    study_case = "example_01a"
    world = World(database_uri=None, export_csv_path=None)
    world.load_scenario(
        inputs_path="examples/inputs", scenario=scenario, study_case=study_case
    )

    assert world.db is None
    assert world.export_csv_path is None

    assert isinstance(world.loop, asyncio.BaseEventLoop)
    assert world.market_operators.keys()
    assert world.markets.keys()
    assert world.unit_operators.keys()
    assert world.forecast_providers.keys()
