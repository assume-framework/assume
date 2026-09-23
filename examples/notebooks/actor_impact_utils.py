# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Extraction/caching helpers for the new-demand-actor EOM/aFRR price-impact analysis.

Reads the per-case batch output archives under ``examples/outputs/<case>/*.tar.gz``
(gzip, one archive per finished scenario) and the no-actor baseline
``examples/outputs/steel.tar.zst`` (zstd, all 24 family/year scenarios bundled), and
caches parsed, aggregated results as pickles under ``examples/outputs/_analysis_cache``
so re-running the notebook while the batch is still producing archives only does work
for scenarios that are new since the last run.

Every extraction reads directly out of the tar stream (``tarfile.extractfile`` for the
gzip archives, ``tar --zstd`` piped straight into pandas for the zstd baseline) --
nothing multi-gigabyte is ever written to disk.
"""

from __future__ import annotations

import re
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

NOTEBOOKS_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = NOTEBOOKS_DIR.parent / "outputs"
INPUTS_DIR = NOTEBOOKS_DIR.parent / "inputs"
CACHE_DIR = OUTPUTS_DIR / "_analysis_cache"
BASELINE_ARCHIVE = OUTPUTS_DIR / "steel.tar.zst"

FAMILIES = [
    "aktuellepolitiken",
    "fokusH2",
    "fokusstrom",
    "hohenachfrage",
    "niedrigenachfrage",
    "technologiemix",
]
YEARS = [2030, 2035, 2040, 2045]

PRICE_TAKER_CASES = [
    "AktuellePolitiken_RN",
    "FokusH2_RN",
    "FokusStrom_RN",
    "HoheNachfrage_RN",
    "NachfrageNiedrig_RN",
    "Technologiemix_RN",
    "WCMean",
    "WCTail",
]
BID_CASES = [f"{c}_BID" for c in PRICE_TAKER_CASES]
ALL_CASES = PRICE_TAKER_CASES + BID_CASES

# RN cases are risk-neutral and matched to their own market (demand identical across
# family subfolders, varies only by year); WCMean/WCTail are risk-averse and evaluated
# against every family.
RN_CASES = [c for c in PRICE_TAKER_CASES if c not in ("WCMean", "WCTail")]

ARCHIVE_RE = re.compile(r"^(?P<family>.+)_(?P<year>20\d\d)_base_case_(?P=year)\.tar\.gz$")

MARKETS = ["EOM", "CRM_energy_neg"]
CAPACITY_MARKETS = ["CRM_capacity_pos", "CRM_capacity_neg"]
AFRR_ENERGY_MARKETS = ["CRM_energy_pos", "CRM_energy_neg"]

# Actor identification in market_orders.csv unit_id -- see notebook methodology cell
# for how these were verified against demand_units.csv / forecasts_df.csv.
STEEL_RE = re.compile(r"^demand_P100000120")
CEMENT_RE = re.compile(r"^demand_P10000012[45]")
AVMAR_RE = re.compile(r"^demand_AVMAR_")
CATEGORIES = ["steel", "cement", "aviation_maritime"]

ORDERS_USECOLS = ["unit_id", "market_id", "start_time", "volume", "accepted_volume"]
META_USECOLS = ["time", "market_id", "price"]


def classify_unit(unit_id) -> str | None:
    if not isinstance(unit_id, str):
        return None
    if STEEL_RE.match(unit_id):
        return "steel"
    if CEMENT_RE.match(unit_id):
        return "cement"
    if AVMAR_RE.match(unit_id):
        return "aviation_maritime"
    return None


def discover_archives() -> pd.DataFrame:
    """List currently-finished with-actor scenario archives (re-glob each call)."""
    rows = []
    for case in ALL_CASES:
        case_dir = OUTPUTS_DIR / case
        if not case_dir.is_dir():
            continue
        for f in sorted(case_dir.glob("*.tar.gz")):
            m = ARCHIVE_RE.match(f.name)
            if not m:
                continue
            rows.append(
                {
                    "case": case,
                    "family": m["family"],
                    "year": int(m["year"]),
                    "is_bid": case.endswith("_BID"),
                    "path": f,
                }
            )
    return pd.DataFrame(rows, columns=["case", "family", "year", "is_bid", "path"])


def _warmup_safe_start(year: int) -> pd.Timestamp:
    """Guard against a possible prepended Dec-31 warm-up day (see notebook note)."""
    return pd.Timestamp(year=year, month=1, day=1)


def _cache_path(*parts: str) -> Path:
    p = CACHE_DIR.joinpath(*parts[:-1], f"{parts[-1]}.pkl")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Baseline (no actors) -- examples/outputs/steel.tar.zst
# --------------------------------------------------------------------------- #


def _ensure_baseline_cached(cache_ns: str, markets: list[str], force: bool = False) -> None:
    """Extract+cache market_meta.csv for all 24 baseline scenarios in one pass.

    steel.tar.zst and cement.tar.zst are confirmed demand-identical (0 differing
    columns), so either works as the no-actor baseline; we use steel.tar.zst.

    Pipes `zstd -dc` into `tarfile` so the 4.8GB archive is decompressed in a
    single sequential streaming pass (member-by-member via tarfile, not written
    to disk), rather than re-scanning it once per scenario. Each distinct
    `cache_ns`/`markets` pair (energy markets, capacity markets, ...) does its
    own such pass the first time it's needed.
    """
    missing = {
        (family, year)
        for family in FAMILIES
        for year in YEARS
        if force or not _cache_path(cache_ns, f"{family}_{year}").exists()
    }
    if not missing:
        return
    if not BASELINE_ARCHIVE.exists():
        raise FileNotFoundError(f"baseline archive not found: {BASELINE_ARCHIVE}")

    dctx_cmd = ["zstd", "-dc", str(BASELINE_ARCHIVE)]
    zstd_proc = subprocess.Popen(dctx_cmd, stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=zstd_proc.stdout, mode="r|") as tf:
            for member in tf:
                if not member.name.endswith("/market_meta.csv"):
                    continue
                # member.name like steel/<family>_<year>_base_case_<year>/market_meta.csv
                sub = member.name.split("/")[1]
                m = ARCHIVE_RE.match(sub + ".tar.gz")
                if not m:
                    continue
                key = (m["family"], int(m["year"]))
                if key not in missing:
                    continue
                import io

                raw = tf.extractfile(member).read()
                df = pd.read_csv(io.BytesIO(raw), usecols=META_USECOLS)
                df = df[df["market_id"].isin(markets)].copy()
                df["time"] = pd.to_datetime(df["time"])
                df = df[df["time"] >= _warmup_safe_start(key[1])]
                df.to_pickle(_cache_path(cache_ns, f"{key[0]}_{key[1]}"))
                missing.discard(key)
                if not missing:
                    break
    finally:
        if zstd_proc.stdout:
            zstd_proc.stdout.close()
        zstd_proc.wait()

    if missing:
        raise RuntimeError(f"baseline archive did not contain market_meta.csv for: {sorted(missing)}")


def ensure_baseline_meta_cached(force: bool = False) -> None:
    _ensure_baseline_cached("baseline_meta", MARKETS, force=force)


def ensure_baseline_capacity_cached(force: bool = False) -> None:
    _ensure_baseline_cached("baseline_capacity", CAPACITY_MARKETS, force=force)


def ensure_baseline_afrr_energy_cached(force: bool = False) -> None:
    _ensure_baseline_cached("baseline_afrr_energy", AFRR_ENERGY_MARKETS, force=force)


def load_baseline_price(family: str, year: int) -> pd.DataFrame:
    ensure_baseline_meta_cached()
    return pd.read_pickle(_cache_path("baseline_meta", f"{family}_{year}"))


def load_baseline_capacity_price(family: str, year: int) -> pd.DataFrame:
    ensure_baseline_capacity_cached()
    return pd.read_pickle(_cache_path("baseline_capacity", f"{family}_{year}"))


def load_baseline_afrr_energy_price(family: str, year: int) -> pd.DataFrame:
    ensure_baseline_afrr_energy_cached()
    return pd.read_pickle(_cache_path("baseline_afrr_energy", f"{family}_{year}"))


# --------------------------------------------------------------------------- #
# With-actor scenarios -- examples/outputs/<case>/<family>_<year>_base_case_<year>.tar.gz
# --------------------------------------------------------------------------- #


def _open_member(archive_path: Path, suffix: str):
    with tarfile.open(archive_path, "r:gz") as tf:
        member = next((m for m in tf.getmembers() if m.name.endswith(suffix)), None)
        if member is None:
            raise FileNotFoundError(f"{suffix} not found in {archive_path}")
        return tf.extractfile(member).read()


def _load_actor_meta(
    case: str, family: str, year: int, archive_path: Path, cache_ns: str, markets: list[str]
) -> pd.DataFrame:
    cp = _cache_path(cache_ns, case, f"{family}_{year}")
    if cp.exists():
        return pd.read_pickle(cp)

    import io

    raw = _open_member(archive_path, "/market_meta.csv")
    df = pd.read_csv(io.BytesIO(raw), usecols=META_USECOLS)
    df = df[df["market_id"].isin(markets)].copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"] >= _warmup_safe_start(year)]
    df.to_pickle(cp)
    return df


def load_actor_price(case: str, family: str, year: int, archive_path: Path) -> pd.DataFrame:
    return _load_actor_meta(case, family, year, archive_path, "actor_meta", MARKETS)


def load_actor_capacity_price(case: str, family: str, year: int, archive_path: Path) -> pd.DataFrame:
    return _load_actor_meta(case, family, year, archive_path, "actor_capacity_meta", CAPACITY_MARKETS)


def load_actor_afrr_energy_price(case: str, family: str, year: int, archive_path: Path) -> pd.DataFrame:
    return _load_actor_meta(case, family, year, archive_path, "actor_afrr_energy_meta", AFRR_ENERGY_MARKETS)


def load_actor_orders_agg(case: str, family: str, year: int, archive_path: Path) -> pd.DataFrame:
    """Per-(hour, market, category) sum of actor bid volume vs. accepted volume.

    Streams market_orders.csv straight out of the tar (chunksize-limited) filtering
    to the handful of actor unit_id prefixes as it goes, so the multi-GB file is
    never held in memory at once.
    """
    cp = _cache_path("actor_orders_agg", case, f"{family}_{year}")
    if cp.exists():
        return pd.read_pickle(cp)

    with tarfile.open(archive_path, "r:gz") as tf:
        member = next((m for m in tf.getmembers() if m.name.endswith("/market_orders.csv")), None)
        if member is None:
            raise FileNotFoundError(f"market_orders.csv not found in {archive_path}")
        fobj = tf.extractfile(member)

        parts = []
        # No usecols: some storage-unit rows are written with fewer columns than
        # the header (a ragged-CSV artifact of ASSUME's output writer, not a
        # corrupted file -- verified against the raw bytes). Combining usecols
        # with on_bad_lines hits a pandas C-engine bug (IndexError inside
        # _concatenate_chunks); reading all columns and subsetting after avoids it.
        for chunk in pd.read_csv(fobj, chunksize=1_000_000, on_bad_lines="skip", low_memory=False):
            chunk = chunk[ORDERS_USECOLS]
            chunk = chunk[chunk["market_id"].isin(MARKETS)]
            category = chunk["unit_id"].map(classify_unit)
            chunk = chunk[category.notna()].copy()
            if chunk.empty:
                continue
            chunk["category"] = category[category.notna()]
            parts.append(
                chunk.groupby(["start_time", "market_id", "category"], as_index=False)[
                    ["volume", "accepted_volume"]
                ].sum()
            )

    if parts:
        agg = pd.concat(parts, ignore_index=True)
        agg = agg.groupby(["start_time", "market_id", "category"], as_index=False)[
            ["volume", "accepted_volume"]
        ].sum()
    else:
        agg = pd.DataFrame(columns=["start_time", "market_id", "category", "volume", "accepted_volume"])

    agg["start_time"] = pd.to_datetime(agg["start_time"])
    agg = agg[agg["start_time"] >= _warmup_safe_start(year)]
    agg.to_pickle(cp)
    return agg


SYSTEM_ORDERS_USECOLS = ["market_id", "start_time", "accepted_volume"]


def load_system_energy_neg_volume(case: str, family: str, year: int, archive_path: Path) -> pd.DataFrame:
    """System-wide (every participant, not just the new actors) hourly accepted
    volume in CRM_energy_neg -- one row per hour: (start_time, system_accepted_volume).

    Separate streaming pass from load_actor_orders_agg (fewer usecols, no
    classify_unit), kept independent so it doesn't touch that function's
    already-cached results.
    """
    cp = _cache_path("system_energy_neg_volume", case, f"{family}_{year}")
    if cp.exists():
        return pd.read_pickle(cp)

    with tarfile.open(archive_path, "r:gz") as tf:
        member = next((m for m in tf.getmembers() if m.name.endswith("/market_orders.csv")), None)
        if member is None:
            raise FileNotFoundError(f"market_orders.csv not found in {archive_path}")
        fobj = tf.extractfile(member)

        # Supply-side (powerplants reducing output, positive) and demand-side
        # (existing demand + the new actors, negative) accepted volumes balance
        # per hour (verified: sum(positive) + sum(negative) ~= 0 to float noise),
        # since one side's accepted volume is the other's by construction. Sum
        # just the demand side (negative) so this is the true cleared volume for
        # the hour, not 2x it from counting both sides of the same trade.
        parts = []
        # See load_actor_orders_agg for why usecols is dropped in favor of
        # reading all columns and subsetting after.
        for chunk in pd.read_csv(fobj, chunksize=1_000_000, on_bad_lines="skip", low_memory=False):
            chunk = chunk[SYSTEM_ORDERS_USECOLS]
            chunk = chunk[(chunk["market_id"] == "CRM_energy_neg") & (chunk["accepted_volume"] < 0)]
            if chunk.empty:
                continue
            chunk = chunk.copy()
            chunk["accepted_volume"] = chunk["accepted_volume"].abs()
            parts.append(chunk.groupby("start_time", as_index=False)["accepted_volume"].sum())

    if parts:
        agg = pd.concat(parts, ignore_index=True).groupby("start_time", as_index=False)["accepted_volume"].sum()
    else:
        agg = pd.DataFrame(columns=["start_time", "accepted_volume"])

    agg["start_time"] = pd.to_datetime(agg["start_time"])
    agg = agg[agg["start_time"] >= _warmup_safe_start(year)]
    agg = agg.rename(columns={"accepted_volume": "system_accepted_volume"})
    agg.to_pickle(cp)
    return agg


def build_energy_neg_share_panel(archives: pd.DataFrame) -> pd.DataFrame:
    """Per (case, family, year, hour): system-wide CRM_energy_neg accepted volume
    alongside each actor category's own accepted volume that hour.

    Only CRM_energy_neg -- the new actors never bid into CRM_energy_pos (blank
    bidding_CRM_energy_pos in demand_units.csv for all of them), so a "positive"
    counterpart would be a flat 0% share for every category, every quarter, by
    construction, and isn't included.
    """
    frames = []
    for row in archives.itertuples():
        sys_vol = load_system_energy_neg_volume(row.case, row.family, row.year, row.path)
        actor_agg = load_actor_orders_agg(row.case, row.family, row.year, row.path)
        actor_agg = actor_agg[actor_agg["market_id"] == "CRM_energy_neg"]
        piv = (
            actor_agg.pivot_table(
                index="start_time", columns="category", values="accepted_volume", aggfunc="sum", fill_value=0
            )
            .abs()
            .reindex(columns=CATEGORIES, fill_value=0.0)
        )
        merged = sys_vol.merge(piv, on="start_time", how="left")
        merged[CATEGORIES] = merged[CATEGORIES].fillna(0.0)
        merged["case"] = row.case
        merged["family"] = row.family
        merged["year"] = row.year
        merged["quarter"] = merged["start_time"].dt.quarter
        frames.append(merged)
    if not frames:
        return pd.DataFrame(
            columns=["start_time", "system_accepted_volume", *CATEGORIES, "case", "family", "year", "quarter"]
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Panel assembly
# --------------------------------------------------------------------------- #


def build_capacity_panel(archives: pd.DataFrame) -> pd.DataFrame:
    """One row per (case, family, year, market, product): actor capacity price, baseline capacity price.

    New actors don't bid into CRM_capacity_pos/neg at all (blank bidding_CRM_capacity_pos/neg
    in demand_units.csv for steel/cement/AVMAR) -- this is a descriptive view of the
    background capacity market, not an actor-impact panel, but the baseline is still
    joined in so a quick with-vs-without check is possible.
    """
    frames = []
    for row in archives.itertuples():
        actor_df = load_actor_capacity_price(row.case, row.family, row.year, row.path)
        base_df = load_baseline_capacity_price(row.family, row.year)
        merged = actor_df.merge(
            base_df, on=["time", "market_id"], how="inner", suffixes=("_actor", "_baseline")
        )
        merged["case"] = row.case
        merged["family"] = row.family
        merged["year"] = row.year
        merged["is_bid"] = row.is_bid
        merged["quarter"] = merged["time"].dt.quarter
        frames.append(merged)
    if not frames:
        return pd.DataFrame(
            columns=[
                "time",
                "market_id",
                "price_actor",
                "price_baseline",
                "case",
                "family",
                "year",
                "is_bid",
                "quarter",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def build_afrr_energy_panel(archives: pd.DataFrame) -> pd.DataFrame:
    """One row per (case, family, year, market, hour): actor aFRR energy price, baseline price.

    CRM_energy_neg is the market the new actors actually bid into (down-regulation
    demand); CRM_energy_pos has no actor bids at all, so it's included only for the
    same paired-subplot layout as Section 6's capacity chart, not as an actor-impact
    market.
    """
    frames = []
    for row in archives.itertuples():
        actor_df = load_actor_afrr_energy_price(row.case, row.family, row.year, row.path)
        base_df = load_baseline_afrr_energy_price(row.family, row.year)
        merged = actor_df.merge(
            base_df, on=["time", "market_id"], how="inner", suffixes=("_actor", "_baseline")
        )
        merged["case"] = row.case
        merged["family"] = row.family
        merged["year"] = row.year
        merged["is_bid"] = row.is_bid
        merged["quarter"] = merged["time"].dt.quarter
        frames.append(merged)
    if not frames:
        return pd.DataFrame(
            columns=[
                "time",
                "market_id",
                "price_actor",
                "price_baseline",
                "case",
                "family",
                "year",
                "is_bid",
                "quarter",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def build_price_panel(archives: pd.DataFrame) -> pd.DataFrame:
    """One row per (case, family, year, market, hour): actor price, baseline price, delta."""
    frames = []
    for row in archives.itertuples():
        actor_df = load_actor_price(row.case, row.family, row.year, row.path)
        base_df = load_baseline_price(row.family, row.year)
        merged = actor_df.merge(
            base_df, on=["time", "market_id"], how="inner", suffixes=("_actor", "_baseline")
        )
        merged["case"] = row.case
        merged["family"] = row.family
        merged["year"] = row.year
        merged["is_bid"] = row.is_bid
        merged["delta"] = merged["price_actor"] - merged["price_baseline"]
        frames.append(merged)
    if not frames:
        return pd.DataFrame(
            columns=[
                "time",
                "market_id",
                "price_actor",
                "price_baseline",
                "case",
                "family",
                "year",
                "is_bid",
                "delta",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def build_orders_panel(archives: pd.DataFrame) -> pd.DataFrame:
    """One row per (case, family, year, market, hour, category): actor bid vs. cleared volume."""
    frames = []
    for row in archives.itertuples():
        agg = load_actor_orders_agg(row.case, row.family, row.year, row.path)
        if agg.empty:
            continue
        agg = agg.copy()
        agg["case"] = row.case
        agg["family"] = row.family
        agg["year"] = row.year
        agg["is_bid"] = row.is_bid
        frames.append(agg)
    if not frames:
        return pd.DataFrame(
            columns=[
                "start_time",
                "market_id",
                "category",
                "volume",
                "accepted_volume",
                "case",
                "family",
                "year",
                "is_bid",
            ]
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Redispatch volume/cost and renewable curtailment
# --------------------------------------------------------------------------- #

VRE_TECHNOLOGY_LABEL = {
    "solar": "Solar PV",
    "onshore wind": "Wind Onshore",
    "offshore wind AC": "Wind Offshore",
    "offshore wind DC": "Wind Offshore",
}
CURTAILMENT_SOURCES = ["Wind Offshore", "Wind Onshore", "Solar PV"]

REDISPATCH_COLUMNS = ["market_id", "start_time", "unit_id", "accepted_volume", "accepted_price"]


def _load_vre_technology_map(case: str, family: str, year: int) -> dict[str, str]:
    """unit name -> {Wind Offshore, Wind Onshore, Solar PV} from that scenario's own
    powerplant_units.csv (technology mix is scenario-specific, not shared across
    cases/years, so this isn't cached at module scope)."""
    path = INPUTS_DIR / case / f"{family}_{year}" / "powerplant_units.csv"
    df = pd.read_csv(path, usecols=["name", "technology"])
    df = df[df["technology"].isin(VRE_TECHNOLOGY_LABEL)]
    return dict(zip(df["name"], df["technology"].map(VRE_TECHNOLOGY_LABEL)))


def load_redispatch_summary(case: str, family: str, year: int, archive_path: Path) -> pd.DataFrame:
    """One-row-per-scenario summary: system redispatch volume/cost + VRE curtailment by source.

    Redispatch up- and down-instructions balance system-wide per hour (verified:
    sum(positive) + sum(negative) ~= 0 to float noise, same as CRM_energy_neg's
    buyer/seller balance) -- "redispatch volume" and its cost-weighted average price
    are computed from the positive (upward) side only, so up and down aren't both
    counted for the same underlying congestion-relief action. Curtailment is a
    different thing: the downward (negative) redispatch specifically applied to
    wind/solar units, i.e. renewable output cut for grid congestion, which is
    naturally one-sided and needs no such adjustment.
    """
    cp = _cache_path("redispatch_summary", case, f"{family}_{year}")
    if cp.exists():
        return pd.read_pickle(cp)

    vre_map = _load_vre_technology_map(case, family, year)

    with tarfile.open(archive_path, "r:gz") as tf:
        member = next((m for m in tf.getmembers() if m.name.endswith("/market_orders.csv")), None)
        if member is None:
            raise FileNotFoundError(f"market_orders.csv not found in {archive_path}")
        fobj = tf.extractfile(member)

        pos_volume = 0.0
        pos_cost_x_volume = 0.0
        curtailment_mwh = dict.fromkeys(CURTAILMENT_SOURCES, 0.0)

        # See load_actor_orders_agg for why usecols is dropped in favor of
        # reading all columns and subsetting after (ragged-row pandas bug).
        for chunk in pd.read_csv(fobj, chunksize=1_000_000, on_bad_lines="skip", low_memory=False):
            chunk = chunk[REDISPATCH_COLUMNS]
            chunk = chunk[chunk["market_id"] == "redispatch"]
            if chunk.empty:
                continue

            pos = chunk[chunk["accepted_volume"] > 0]
            pos_volume += pos["accepted_volume"].sum()
            pos_cost_x_volume += (pos["accepted_price"] * pos["accepted_volume"]).sum()

            neg = chunk[chunk["accepted_volume"] < 0].copy()
            if not neg.empty:
                neg["technology"] = neg["unit_id"].map(vre_map)
                neg = neg[neg["technology"].notna()]
                if not neg.empty:
                    by_tech = neg.groupby("technology")["accepted_volume"].sum().abs()
                    for tech, val in by_tech.items():
                        curtailment_mwh[tech] += val

    row = {
        "case": case,
        "family": family,
        "year": year,
        "redispatch_volume_twh": pos_volume / 1e6,
        "avg_redispatch_cost": (pos_cost_x_volume / pos_volume) if pos_volume else float("nan"),
    }
    for tech in CURTAILMENT_SOURCES:
        row[f"{tech}_curtailment_twh"] = curtailment_mwh[tech] / 1e6

    summary = pd.DataFrame([row])
    summary.to_pickle(cp)
    return summary


def build_redispatch_panel(archives: pd.DataFrame) -> pd.DataFrame:
    """One row per (case, family, year): redispatch volume/cost + VRE curtailment by source."""
    frames = [
        load_redispatch_summary(row.case, row.family, row.year, row.path) for row in archives.itertuples()
    ]
    if not frames:
        cols = ["case", "family", "year", "redispatch_volume_twh", "avg_redispatch_cost"] + [
            f"{t}_curtailment_twh" for t in CURTAILMENT_SOURCES
        ]
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)
