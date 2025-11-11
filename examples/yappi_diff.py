# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pstats
from pathlib import Path

import pandas as pd


def load_yappi_profile(path: str) -> pd.DataFrame:
    """Load a pstats .profile file into a DataFrame."""
    stats = pstats.Stats(path)

    data = []
    # stats.stats is a dict: {(filename, lineno, funcname): (cc, nc, tt, ct, callers)}
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, lineno, func_name = func
        # Create a readable function ID
        func_id = f"{filename}:{func_name}:{lineno}"

        data.append(
            {
                "function": func_id,
                "ncall": nc,  # number of calls
                "tsub": tt,  # total time (excluding subcalls)
                "ttot": ct,  # cumulative time (including subcalls)
                "tavg": ct / nc if nc > 0 else 0,  # average time per call
            }
        )

    df = pd.DataFrame(data).set_index("function")
    return df


def compare_profiles(paths: list[str]) -> pd.DataFrame:
    """Load multiple .profile files and compute diffs."""
    dfs = []
    file_names = []

    for path in paths:
        # Extract filename without extension
        file_name = Path(path).stem  # e.g., "02a" from "02a.profile"
        file_names.append(file_name)

        df = load_yappi_profile(path)
        df = df.add_suffix(f"_{file_name}")
        dfs.append(df)

    # Merge on function name (index)
    merged = pd.concat(dfs, axis=1).fillna(0)

    # Compute deltas (vs first file)
    if len(paths) > 1:
        base_name = file_names[0]
        for col in ["ncall", "ttot", "tsub", "tavg"]:
            base = merged[f"{col}_{base_name}"]
            for file_name in file_names[1:]:
                merged[f"{col}_diff_{file_name}-{base_name}"] = (
                    merged[f"{col}_{file_name}"] - base
                )
                # Avoid division by zero for percentage
                with pd.option_context("mode.use_inf_as_na", True):
                    merged[f"{col}_pct_{file_name}-{base_name}"] = (
                        merged[f"{col}_diff_{file_name}-{base_name}"]
                        / base.replace(0, float("nan"))
                    ) * 100

    return merged, file_names


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple pstats .profile files using pandas."
    )
    parser.add_argument(
        "profiles", nargs="+", help="Paths to .profile files to compare"
    )
    parser.add_argument("--output", "-o", help="Output CSV file (optional)")
    parser.add_argument(
        "--sort",
        "-s",
        help="Column to sort by (default: tsub_diff if available, or tsub)",
    )
    parser.add_argument("--top", "-t", type=int, default=10, help="Show top N rows")
    parser.add_argument(
        "--metric",
        "-m",
        default="tsub",
        choices=["ttot", "tsub", "tavg", "ncall"],
        help="Metric to display in summary (default: tsub)",
    )
    args = parser.parse_args()

    df, file_names = compare_profiles(args.profiles)

    # Determine sort column
    sort_col = args.sort
    if not sort_col and len(file_names) > 1:
        # Default to first diff column based on selected metric
        sort_col = f"{args.metric}_diff_{file_names[1]}-{file_names[0]}"
    elif not sort_col:
        sort_col = f"{args.metric}_{file_names[0]}"

    if sort_col in df.columns:
        df = df.sort_values(by=sort_col, ascending=False)
    else:
        print(
            f"Warning: Sort column '{sort_col}' not found. Available columns: {list(df.columns)}"
        )

    # Display summary
    pd.set_option("display.max_rows", args.top)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # Build column list dynamically based on file names and selected metric
    summary_cols = []

    # Add base values for first two files
    for name in file_names[:2]:
        if f"{args.metric}_{name}" in df.columns:
            summary_cols.append(f"{args.metric}_{name}")

    # Add diff and percentage columns
    if len(file_names) > 1:
        diff_col = f"{args.metric}_diff_{file_names[1]}-{file_names[0]}"
        pct_col = f"{args.metric}_pct_{file_names[1]}-{file_names[0]}"
        if diff_col in df.columns:
            summary_cols.append(diff_col)
        if pct_col in df.columns:
            summary_cols.append(pct_col)

    print(f"\n{'='*80}")
    print(f"Profile Comparison - Metric: {args.metric} - Sorted by: {sort_col}")
    print(f"{'='*80}\n")
    print(df[summary_cols].head(args.top))

    if args.output:
        df.to_csv(args.output)
        print(f"\n✅ Saved full comparison to {args.output}")


if __name__ == "__main__":
    main()
