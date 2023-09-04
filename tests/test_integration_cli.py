import pytest

from assume.cli import cli


@pytest.mark.slow
def test_cli():
    args = "-s example_01a -c base -db sqlite:///./examples/local_db/test_mini.db"
    cli(args.split(" "))
