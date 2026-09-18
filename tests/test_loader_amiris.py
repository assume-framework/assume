# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.scenario.loader_amiris import (
    get_send_receive_msgs_per_id,
    read_amiris_yaml,
)

FIXTURE_PATH = "tests/fixtures/amiris_scenario"


def test_read_amiris_yaml_resolves_includes():
    scenario = read_amiris_yaml(FIXTURE_PATH)

    # scalar !include of a single file, including a nested !include inside it
    assert scenario["Schema"] == {"Version": {"SchemaVersion": "0.1-test"}}

    # plain (non-included) content is untouched
    assert scenario["GeneralProperties"]["RunId"] == 1
    assert scenario["Agents"][0]["Type"] == "DemandTrader"

    # sequence-form !include with a glob pattern: an unflattened list of the
    # parsed content of each matched file, in path order
    assert scenario["Contracts"] == [
        {
            "Contracts": [
                {"SenderId": 100, "ReceiverId": 200, "ProductName": "EnergyExchange"}
            ]
        },
        {
            "Contracts": [
                {"SenderId": 300, "ReceiverId": [100, 400], "ProductName": "Bids"}
            ]
        },
    ]


def test_amiris_contracts_are_usable_by_downstream_code():
    # get_send_receive_msgs_per_id relies on amiris_scenario["Contracts"]
    # being that unflattened, per-file list - this is the actual consumer
    # of the !include output inside assume.
    scenario = read_amiris_yaml(FIXTURE_PATH)

    sends, receives = get_send_receive_msgs_per_id(100, scenario["Contracts"])

    # agent 100 is SenderId in contracts_a.yaml (EnergyExchange)
    # and (via a list) ReceiverId in contracts_b.yaml (Bids)
    assert [c["ProductName"] for c in sends] == ["EnergyExchange"]
    assert [c["ProductName"] for c in receives] == ["Bids"]
