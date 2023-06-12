from assume.cli import cli

scenarios = [
    "-s example_02 -c base_case_2019 -db postgresql://assume:assume@localhost:5432/assume",
    "-s example_02 -c ltm_case01 -db postgresql://assume:assume@localhost:5432/assume",
    "-s example_02 -c ltm_case05 -db postgresql://assume:assume@localhost:5432/assume",
    "-s example_02 -c ltm_case06 -db postgresql://assume:assume@localhost:5432/assume",
    "-s example_02 -c ltm_case07 -db postgresql://assume:assume@localhost:5432/assume",
    "-s example_02 -c ltm_case08 -db postgresql://assume:assume@localhost:5432/assume",
    "-s example_02 -c ltm_case09 -db postgresql://assume:assume@localhost:5432/assume",
    "-s example_02 -c ltm_case10 -db postgresql://assume:assume@localhost:5432/assume",
]
for args in scenarios:
    cli(args.split(" "))
