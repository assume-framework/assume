from assume.cli import cli


def test_cli():
    args = (
        "-s example_01a -c tiny_example -db sqlite:///./examples/local_db/test_mini.db"
    )
    cli(args.split(" "))
