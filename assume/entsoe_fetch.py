# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
ENTSO-E Transparency Platform 
Fetches:
- A81 Amounts under contract (17.1.B)
- A89 Prices of procured reserves (17.1.C)

Works around API document-limit (Reason 999) by querying month-by-month.
"""

import os
import time
import requests
import xmltodict
import pandas as pd

# ================== CONFIG ==================
ENTSOE_TOKEN = "49e549fc-a751-48a2-bae3-4d30c0a68e59"  # consider moving to $Env:ENTSOE_TOKEN and regenerating
CONTROL_AREA_FR = "10YFR-RTE------C"                   # France (RTE) control area EIC
# German TSOs – use as Scheduling Areas (SCA)
CONTROL_AREAS_DE = {
    "50Hertz":   "10YDE-VE-------2",
    "Amprion":   "10YDE-RWENET---I",
    "TenneTDE":  "10YDE-EON------1",
    "TransnetBW":"10YDE-ENBW-----N",
}
# Continental Europe Synchronous Area (for FCR fallback)
SYNCH_AREA_CE = "10YEU-CONT-SYNC0"

YEAR = 2024
PRODUCTS = {"FCR": "A95", "aFRR": "A96", "mFRR": "A97"}
BASE = "https://web-api.tp.entsoe.eu/api"
TYPE_MARKET_AGREEMENT = "A01"  # daily (robust across TSOs)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============== HELPERS ====================
def _try_domain_keys(params_base, domain_eic, domain_keys=("schedulingArea_Domain","controlArea_Domain")):
    """
    Try the given EIC with several domain parameter names (SCA first, then ControlArea).
    Returns: xml text of the first successful (HTTP 200) call.
    Raises HTTPError if none succeed.
    """
    last_err = None
    for key in domain_keys:
        params = dict(params_base)
        params[key] = domain_eic
        try:
            return _api_get(params)
        except requests.HTTPError as e:
            last_err = e
            continue
    if last_err:
        raise last_err

def _ensure_cols(df):
    for c in ["time_utc", "direction", "quantity_MW", "price_EUR_per_MW_h"]:
        if c not in df.columns:
            df[c] = None
    return df

def _merge_amt_prc(df_amt, df_prc):
    df_amt = _ensure_cols(df_amt)
    df_prc = _ensure_cols(df_prc)
    if df_amt.empty and df_prc.empty:
        return pd.DataFrame(columns=["time_utc","direction","quantity_MW","price_EUR_per_MW_h"])
    if df_amt.empty:
        df = df_prc.copy()
        if "quantity_MW" not in df.columns: df["quantity_MW"] = None
        return df
    if df_prc.empty:
        df = df_amt.copy()
        if "price_EUR_per_MW_h" not in df.columns: df["price_EUR_per_MW_h"] = None
        return df
    return pd.merge(
        df_amt[["time_utc","direction","quantity_MW"]],
        df_prc[["time_utc","direction","price_EUR_per_MW_h"]],
        on=["time_utc","direction"], how="outer"
    )

def _api_get(params, max_retries=3, sleep_s=1.0):
    if not ENTSOE_TOKEN:
        raise SystemExit("No ENTSOE_TOKEN found. Set environment variable ENTSOE_TOKEN with your API key.")
    params["securityToken"] = ENTSOE_TOKEN
    last = None
    for attempt in range(1, max_retries + 1):
        r = requests.get(BASE, params=params, timeout=120)
        last = r
        if r.status_code == 200:
            return r.text
        time.sleep(sleep_s * attempt)
    txt = ""
    try:
        txt = last.text
    except Exception:
        pass
    raise requests.HTTPError(f"{last.status_code} for {last.url}\nResponse:\n{txt}", response=last)

def _parse_balancing_xml_to_df(xml_text):
    doc = xmltodict.parse(xml_text, dict_constructor=dict)
    root = doc.get("Balancing_MarketDocument")
    if not root:
        return pd.DataFrame(columns=["time_utc", "quantity_MW", "price_EUR_per_MW_h", "direction"])
    series = root.get("TimeSeries")
    if not series:
        return pd.DataFrame(columns=["time_utc", "quantity_MW", "price_EUR_per_MW_h", "direction"])
    if isinstance(series, dict):
        series = [series]

    rows = []
    for ts in series:
        dir_code = ts.get("flowDirection.direction") or "A03"
        direction = {"A01": "UP", "A02": "DOWN", "A03": "NA"}.get(dir_code, "NA")

        periods = ts.get("Period")
        if not periods:
            continue
        if isinstance(periods, dict):
            periods = [periods]

        for per in periods:
            ti = per.get("timeInterval", {}) or {}
            start = ti.get("start")
            res = per.get("resolution", "PT60M")
            points = per.get("Point")
            if not points:
                continue
            if isinstance(points, dict):
                points = [points]

            try:
                t0 = pd.Timestamp(start)
            except Exception:
                t0 = None

            for p in points:
                pos = int(p.get("position", "1"))
                qty = p.get("quantity")
                price = p.get("procurement_Price.amount")

                ts_utc = pd.Timestamp(start) if start else None
                if t0 is not None and res.startswith("PT"):
                    hours = 1.0
                    try:
                        if "H" in res:
                            h = res.replace("PT", "").replace("H", "")
                            hours = float(h) if h else 1.0
                        elif "M" in res:
                            m = res.replace("PT", "").replace("M", "")
                            minutes = float(m) if m else 60.0
                            hours = minutes / 60.0
                    except Exception:
                        hours = 1.0
                    ts_utc = t0 + pd.to_timedelta((pos - 1) * hours, unit="h")

                rows.append(
                    {
                        "time_utc": ts_utc,
                        "quantity_MW": float(qty) if qty is not None else None,
                        "price_EUR_per_MW_h": float(price) if price is not None else None,
                        "direction": direction,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["time_utc", "quantity_MW", "price_EUR_per_MW_h", "direction"])
    df = df.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)
    return df

def _month_ranges(year):
    ranges = []
    for m in range(1, 13):
        start = pd.Timestamp(year=year, month=m, day=1, hour=0, minute=0, tz="UTC")
        end = (start + pd.offsets.MonthBegin(1))
        ranges.append((start.strftime("%Y%m%d%H%M"), end.strftime("%Y%m%d%H%M")))
    return ranges

# ========== API WRAPPERS (chunked) ==========
def fetch_amounts_chunked(domain_eic, business_type, year, type_market="A01", is_fcr=False):
    all_parts = []
    for (ps, pe) in _month_ranges(year):
        base = {
            "documentType": "A81",
            "type_MarketAgreement.Type": type_market,
            "periodStart": ps,
            "periodEnd": pe,
        }
        base_bt = dict(base)
        base_bt["businessType"] = business_type
        try:
            xml = _try_domain_keys(base_bt, domain_eic)
            df = _parse_balancing_xml_to_df(xml)
            if not df.empty:
                all_parts.append(df)
        except requests.HTTPError as e:
            if is_fcr:
                try:
                    base_fcr = dict(base_bt)  # keep businessType=A95
                    xml = _try_domain_keys(base_fcr, SYNCH_AREA_CE, domain_keys=("synchronousArea_Domain",))
                    df = _parse_balancing_xml_to_df(xml)
                    if not df.empty:
                        all_parts.append(df)
                        continue
                except requests.HTTPError:
                    pass
            print(f"  [warn] A81 {business_type} {ps[:6]}: {e}")
    return pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()

def fetch_prices_chunked(domain_eic, business_type, year, type_market="A01", is_fcr=False):
    all_parts = []
    for (ps, pe) in _month_ranges(year):
        base = {
            "documentType": "A89",
            "type_MarketAgreement.Type": type_market,
            "periodStart": ps,
            "periodEnd": pe,
        }
        try:
            xml = _try_domain_keys(base, domain_eic)  # SCA then controlArea
            df = _parse_balancing_xml_to_df(xml)
            if not df.empty:
                all_parts.append(df)
                continue
        except requests.HTTPError:
            try:
                base_bt = dict(base)
                base_bt["businessType"] = business_type
                xml = _try_domain_keys(base_bt, domain_eic)
                df = _parse_balancing_xml_to_df(xml)
                if not df.empty:
                    all_parts.append(df)
                    continue
            except requests.HTTPError as e2:
                if is_fcr:
                    try:
                        xml = _try_domain_keys(base, SYNCH_AREA_CE, domain_keys=("synchronousArea_Domain",))
                        df = _parse_balancing_xml_to_df(xml)
                        if not df.empty:
                            all_parts.append(df)
                            continue
                    except requests.HTTPError:
                        pass
                print(f"  [warn] A89 {business_type} {ps[:6]}: {e2}")
    return pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()

# ================== MAIN (DE) ==================
def get_DE_balancing_for_2024():
    results = {}
    for name, bt in PRODUCTS.items():
        is_fcr = (name == "FCR")
        all_tso_frames = []
        print(f"\n=== Fetching {name} ({bt}) DE 2024 per Scheduling Area ===")
        for tso, eic in CONTROL_AREAS_DE.items():
            print(f"  - {tso} (SCA EIC {eic})")
            df_amt = fetch_amounts_chunked(eic, bt, YEAR, TYPE_MARKET_AGREEMENT, is_fcr=is_fcr)
            if df_amt.empty: print(f"    [warn] No amount data for {tso}.")
            df_prc = fetch_prices_chunked(eic, bt, YEAR, TYPE_MARKET_AGREEMENT, is_fcr=is_fcr)
            if df_prc.empty: print(f"    [warn] No price data for {tso}.")
            df = _merge_amt_prc(df_amt, df_prc)
            if not df.empty:
                df = df.sort_values("time_utc").reset_index(drop=True)
                df["product"] = name
                df["tso"] = tso
                df = df[["time_utc","product","tso","direction","quantity_MW","price_EUR_per_MW_h"]]
            out_path = os.path.join(SCRIPT_DIR, f"entsoe_DE_{YEAR}_{name}_{tso}.csv")
            df.to_csv(out_path, index=False)
            print(f"    ✅ Saved: {out_path} (rows={len(df)})")
            all_tso_frames.append(df)

        # Optional DE aggregate
        if any(not d.empty for d in all_tso_frames):
            big = pd.concat(all_tso_frames, ignore_index=True)
            agg_q = big.groupby(["time_utc","direction"], as_index=False)["quantity_MW"].sum(min_count=1)
            def _agg_price(g):
                vals = g["price_EUR_per_MW_h"].dropna().unique()
                return vals[0] if len(vals) == 1 else float("nan")
            agg_p = big.groupby(["time_utc","direction"]).apply(_agg_price).reset_index(name="price_EUR_per_MW_h")
            de_all = pd.merge(agg_q, agg_p, on=["time_utc","direction"], how="outer").sort_values("time_utc")
            de_all["product"] = name
            de_all["tso"] = "ALL"
            de_all = de_all[["time_utc","product","tso","direction","quantity_MW","price_EUR_per_MW_h"]]
        else:
            de_all = pd.DataFrame(columns=["time_utc","product","tso","direction","quantity_MW","price_EUR_per_MW_h"])

        out_path = os.path.join(SCRIPT_DIR, f"entsoe_DE_{YEAR}_{name}_ALL.csv")
        de_all.to_csv(out_path, index=False)
        print(f"  ✅ Saved aggregate: {out_path} (rows={len(de_all)})")
        results[name] = {"TSO": all_tso_frames, "ALL": de_all}
    return results

# ================== MAIN (FR) ==================
def get_FR_balancing_for_2024():
    results = {}
    for name, bt in PRODUCTS.items():
        print(f"\n=== Fetching {name} ({bt}) FR 2024 month-by-month ===")
        df_amt = fetch_amounts_chunked(CONTROL_AREA_FR, bt, YEAR, TYPE_MARKET_AGREEMENT, is_fcr=(name=="FCR"))
        if df_amt.empty:
            print(f"  [warn] No amount data returned for {name} across 2024.")
        df_prc = fetch_prices_chunked(CONTROL_AREA_FR, bt, YEAR, TYPE_MARKET_AGREEMENT, is_fcr=(name=="FCR"))
        if df_prc.empty:
            print(f"  [warn] No price data returned for {name} across 2024.")

        df = _merge_amt_prc(df_amt, df_prc)
        if not df.empty and "time_utc" in df.columns:
            df = df.sort_values("time_utc").reset_index(drop=True)
        df["product"] = name
        df = df[["time_utc", "product", "direction", "quantity_MW", "price_EUR_per_MW_h"]]

        out_path = os.path.join(SCRIPT_DIR, f"entsoe_FR_{YEAR}_{name}.csv")
        df.to_csv(out_path, index=False)
        print(f"  ✅ Saved: {out_path}  (rows={len(df)})")
        results[name] = df
    return results

# ================== DRIVER =====================
if __name__ == "__main__":
    data = get_DE_balancing_for_2024()
    for product, content in data.items():
        print(f"\nPreview {product} – DE aggregate:")
        print(content["ALL"].head(5).to_string(index=False))
