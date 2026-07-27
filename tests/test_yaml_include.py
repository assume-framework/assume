# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Regression tests for :mod:`assume.scenario.yaml_include`, the in-house
replacement for the GPL-licensed ``pyyaml-include`` package.

These pin down the exact behaviour AMIRIS scenario files rely on, reverse
engineered from ``pyyaml-include`` 2.2 and from real scenario files at
https://gitlab.com/dlr-ve/esy/amiris/examples (e.g. ``Contracts: !include
["contracts/*.yaml", "Contracts"]``), so that swapping the implementation
does not change what :func:`assume.scenario.loader_amiris.read_amiris_yaml`
returns.
"""

import pytest
import yaml

from assume.scenario.yaml_include import Constructor

FIXTURE_PATH = "tests/fixtures/amiris_scenario"


def _loader_with_base_dir(base_dir: str) -> type:
    loader_cls = yaml.FullLoader
    yaml.add_constructor("!include", Constructor(base_dir=base_dir), loader_cls)
    return loader_cls


def test_scalar_include_loads_and_parses_single_file():
    data = yaml.load(
        'key: !include "schema.yaml"', Loader=_loader_with_base_dir(FIXTURE_PATH)
    )

    assert data == {"key": {"Version": {"SchemaVersion": "0.1-test"}}}


def test_nested_includes_resolve_relative_to_the_original_base_dir():
    # schema.yaml itself contains `Version: !include "meta/version.yaml"`.
    # That path only exists relative to FIXTURE_PATH, not relative to
    # schema.yaml's own directory, so this only works if nested includes
    # keep resolving against the original base_dir.
    data = yaml.load(
        'key: !include "schema.yaml"', Loader=_loader_with_base_dir(FIXTURE_PATH)
    )

    assert data["key"]["Version"] == {"SchemaVersion": "0.1-test"}


def test_sequence_include_expands_glob_into_a_list_of_parsed_files():
    data = yaml.load(
        'key: !include ["contracts/*.yaml", "AnyLabel"]',
        Loader=_loader_with_base_dir(FIXTURE_PATH),
    )

    # unflattened: one list entry per matched file, sorted by path, each
    # still wrapped in that file's own top-level "Contracts" key
    assert data["key"] == [
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


def test_scalar_include_also_expands_glob_without_a_sequence_wrapper():
    # pyyaml-include treats any urlpath containing wildcard characters as a
    # glob, regardless of whether it was written as a plain scalar or
    # wrapped in a YAML sequence.
    data = yaml.load(
        'key: !include "contracts/*.yaml"', Loader=_loader_with_base_dir(FIXTURE_PATH)
    )

    assert len(data["key"]) == 2
    assert all("Contracts" in entry for entry in data["key"])


def test_extra_sequence_entries_are_ignored():
    with_label = yaml.load(
        'key: !include ["contracts/*.yaml", "Contracts"]',
        Loader=_loader_with_base_dir(FIXTURE_PATH),
    )
    without_label = yaml.load(
        'key: !include "contracts/*.yaml"',
        Loader=_loader_with_base_dir(FIXTURE_PATH),
    )

    assert with_label == without_label


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        yaml.load(
            'key: !include "does-not-exist.yaml"',
            Loader=_loader_with_base_dir(FIXTURE_PATH),
        )


def test_glob_with_no_matches_returns_empty_list():
    data = yaml.load(
        'key: !include "no-such-dir/*.yaml"',
        Loader=_loader_with_base_dir(FIXTURE_PATH),
    )

    assert data["key"] == []
