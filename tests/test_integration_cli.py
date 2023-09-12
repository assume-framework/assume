import pytest

from assume.cli import cli


@pytest.mark.slow
def test_cli():
    args = "-s example_01a -c tiny -db sqlite:///./examples/local_db/test_mini.db"
    cli(args.split(" "))


@pytest.mark.slow
@pytest.mark.require_learning
def test_cli_learning():
    args = "-s example_01_rl -c tiny -db sqlite:///./examples/local_db/test_mini_rl.db"
    cli(args.split(" "))
