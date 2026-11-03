# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime, timedelta

import dateutil.rrule as rr
import pytest
import yaml
from dateutil.relativedelta import relativedelta as rd

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.market_timing import check_market_timing, render_market_timing
from assume.scenario.loader_csv import load_market_configs
from assume_cli.cli import market_cli

# the diagrams are printed in each test, pytest shows them if a test fails
start = datetime(2019, 1, 1)


def daily_market(market_id, first_opening, opening_hours, first_delivery_hours):
    """A market opening daily, trading 24 hourly products."""
    return MarketConfig(
        market_id=market_id,
        opening_hours=rr.rrule(
            rr.DAILY, dtstart=first_opening, until=start + timedelta(days=3)
        ),
        opening_duration=timedelta(hours=opening_hours),
        market_products=[
            MarketProduct(rd(hours=1), 24, rd(hours=first_delivery_hours))
        ],
    )


def test_valid_setup():
    """
    Setup as in example_01d: the day-ahead market is cleared at 20:00,
    redispatch opens afterwards at 21:00 with the day-ahead results available.
    Both markets are cleared before their delivery starts at 00:00.
    """
    configs = [
        daily_market("EOM", start, opening_hours=20, first_delivery_hours=24),
        daily_market(
            "redispatch",
            start + timedelta(hours=21),
            opening_hours=2,
            first_delivery_hours=3,
        ),
    ]
    text = render_market_timing(configs, use_ascii=True)
    print(text)

    assert check_market_timing(configs) == []
    assert "EOM" in text and "redispatch" in text
    assert "WARNING" not in text and "ERROR" not in text


def test_opening_while_other_market_is_cleared():
    """
    The day-ahead market is cleared at 12:00, redispatch opens at the same time.
    As agents bid directly on opening, the day-ahead results are not available yet,
    although redispatch trades the same delivery hours, so a warning is raised.
    """
    configs = [
        daily_market("EOM", start, opening_hours=12, first_delivery_hours=12),
        daily_market(
            "redispatch",
            start + timedelta(hours=12),
            opening_hours=1,
            first_delivery_hours=1,
        ),
    ]
    text = render_market_timing(configs, use_ascii=True)
    print(text)

    issues = check_market_timing(configs)
    assert [(i.level, i.market_id, i.opening) for i in issues] == [
        ("warning", "redispatch", datetime(2019, 1, 1, 12))
    ]
    # the clash is shown in the graph and the warning is printed below
    assert "timing clashes" in text
    assert "WARNING" in text


def test_opening_while_other_market_is_cleared_for_other_delivery():
    """
    Same as above, but redispatch trades the day after the cleared day-ahead hours.
    The day-ahead results are not needed, so the same time opening is fine.
    """
    configs = [
        daily_market("EOM", start, opening_hours=12, first_delivery_hours=12),
        daily_market(
            "redispatch",
            start + timedelta(hours=12),
            opening_hours=1,
            first_delivery_hours=48,
        ),
    ]
    print(render_market_timing(configs, use_ascii=True))

    assert check_market_timing(configs) == []


def test_bidding_after_delivery_started():
    """
    The market is cleared at 13:00, but its delivery already starts at 12:00.
    This is physically impossible and an error.
    """
    configs = [daily_market("EOM", start, opening_hours=13, first_delivery_hours=12)]
    print(render_market_timing(configs, use_ascii=True))

    issues = check_market_timing(configs)
    assert [(i.level, i.market_id, i.clash_start, i.clash_end) for i in issues] == [
        ("error", "EOM", datetime(2019, 1, 1, 12), datetime(2019, 1, 1, 13))
    ]


def test_load_market_configs():
    """The market configs can be loaded from a config.yaml without loading the units."""
    configs = load_market_configs("examples/inputs/example_01c", "eom_and_crm")
    assert [config.market_id for config in configs] == ["EOM", "CRM_pos", "CRM_neg"]

    with pytest.raises(ValueError, match="not found"):
        load_market_configs("examples/inputs/example_01c", "unknown_case")


def test_market_cli(capsys):
    """The CLI renders the timing of a config and prints the timing issues."""
    market_cli(["examples/inputs/example_01c/config.yaml", "-c", "eom_and_crm"])
    out = capsys.readouterr().out
    assert "EOM" in out and "CRM_pos" in out
    assert "WARNING" in out

    market_cli(["examples/inputs/example_01a", "-c", "tiny", "--mermaid"])
    out = capsys.readouterr().out
    assert out.startswith("gantt")


def test_market_cli_exits_on_error(tmp_path):
    """The CLI exits with code 1 if the timing contains errors."""
    config = {
        "base": {
            "start_date": "2019-01-01 00:00",
            "end_date": "2019-01-03 00:00",
            "time_step": "1h",
            "markets_config": {
                "EOM": {
                    "operator": "EOM_operator",
                    "products": [
                        {"duration": "1h", "count": 24, "first_delivery": "1h"}
                    ],
                    "opening_frequency": "24h",
                    "opening_duration": "24h",
                    "market_mechanism": "pay_as_clear",
                }
            },
        }
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config))

    with pytest.raises(SystemExit) as exc_info:
        market_cli([str(config_file), "--ascii"])
    assert exc_info.value.code == 1
