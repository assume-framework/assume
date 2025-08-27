# -*- coding: utf-8 -*-
"""
ENTSO-E Transparency Platform – Sweden 2024 (FCR, aFRR, mFRR)
Fetches:
- A81 Amounts under contract (TR 17.1.B)
- A89 Prices of procured reserves (TR 17.1.C)

Robustness:
- Month-by-month to avoid "requested data exceeds limit" (Reason 999)
- Tries multiple Type_MarketAgreement.Type: A01 (Daily), A02 (Weekly), A03 (Monthly)
- Merges amounts & prices on [time_utc, direction]
"""

import os
import time
import requests
import xmltodict
import pandas as pd

# ================== CONFIG ==================
# 1) Use environment variable (recommended) or paste the token string here
ENTSOE_TOKEN = os.getenv("ENTSOE_TOKEN", "49e549fc-a751-48a2-bae3-4d30c0a68e59")

# Sweden control area (Svenska kraftnät)
CONTROL_AREA_SE = "10YSE-1--------K"

YEAR = 2024
PRODUCTS = {"FCR": "A95", "aFRR": "A96", "mFRR": "A97"}
BASE = "https://web-api.tp.entsoe.eu/api"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# We will cycle through these if a month returns no data
TMA_CANDIDATES = ["A01", "A02", "A03"]  # Daily, Weekly, Monthly

# =============== HELPERS ====================
def _api_get(params, max_retries=3, sleep_s=1.0):
    if not ENTSOE_TOKEN or ENTSOE_TOKEN == "PUT_YOUR_TOKEN_HERE":
        raise SystemExit("Missing ENTSOE_TOKEN. Set $Env:ENTSOE_TOKEN or paste it into the script.")
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
    """Parse Balancing_MarketDocument into DataFrame with columns:
       time_utc, quantity_MW, price_EUR_per_MW_h, direction (UP/DOWN/NA)
    """
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
        dcode = ts.get("flowDirection.direction") or "A03"
        direction = {"A01": "UP", "A02": "DOWN", "A03": "NA"}.get(dcode, "NA")

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
                    # Position stepping for PT1H / PT60M etc.
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

                rows.append({
                    "time_utc": ts_utc,
                    "quantity_MW": float(qty) if qty is not None else None,
                    "price_EUR_per_MW_h": float(price) if price is not None else None,
                    "direction": direction,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["time_utc", "quantity_MW", "price_EUR_per_MW_h", "direction"])
    return df.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)

def _month_ranges(year):
    """Return [(periodStart, periodEnd)] per month in UTC, as yyyymmddHHMM."""
    ranges = []
    for m in range(1, 13):
        start = pd.Timestamp(year=year, month=m, day=1, hour=0, minute=0, tz="UTC")
        end = (start + pd.offsets.MonthBegin(1))
        ranges.append((start.strftime("%Y%m%d%H%M"), end.strftime("%Y%m%d%H%M")))
    return ranges

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

# ========== API WRAPPERS (chunked) ==========
def fetch_amounts_SE_chunked(control_area_eic, business_type, year):
    """A81 amounts. Try product-specific AND combined B&C, over Daily/Weekly/Monthly."""
    all_parts = []
    for (ps, pe) in _month_ranges(year):
        got_any = False
        # try Daily, then Weekly, then Monthly
        for tma in TMA_CANDIDATES:
            # --- V1: product-specific (A95/A96/A97) ---
            params_v1 = {
                "documentType": "A81",
                "type_MarketAgreement.Type": tma,
                "businessType": business_type,  # A95/A96/A97
                "controlArea_Domain": control_area_eic,
                "periodStart": ps,
                "periodEnd": pe,
            }
            try:
                xml = _api_get(params_v1)
                df = _parse_balancing_xml_to_df(xml)
                if not df.empty:
                    all_parts.append(df); got_any = True; break
            except requests.HTTPError:
                pass

            # --- V2: combined B&C view (common for some TSOs) ---
            params_v2 = {
                "documentType": "A81",
                "type_MarketAgreement.Type": tma,
                "businessType": "B95",   # procured capacity
                "processType": "A52",    # procurement
                "controlArea_Domain": control_area_eic,
                "periodStart": ps,
                "periodEnd": pe,
            }
            try:
                xml = _api_get(params_v2)
                df = _parse_balancing_xml_to_df(xml)
                if not df.empty:
                    all_parts.append(df); got_any = True; break
            except requests.HTTPError:
                pass

        if not got_any:
            print(f"  [warn] A81 {business_type} {ps[:6]}: no data (TMA tried: {TMA_CANDIDATES})")
    return pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()


def fetch_prices_SE_chunked(control_area_eic, business_type, year):
    """A89 prices. Try without businessType, then with A95/A96/A97, over Daily/Weekly/Monthly."""
    all_parts = []
    for (ps, pe) in _month_ranges(year):
        got_any = False
        for tma in TMA_CANDIDATES:
            base = {
                "documentType": "A89",
                "type_MarketAgreement.Type": tma,
                "controlArea_Domain": control_area_eic,
                "periodStart": ps,
                "periodEnd": pe,
            }
            # P1: broad (no businessType)
            try:
                xml = _api_get(base)
                df = _parse_balancing_xml_to_df(xml)
                if not df.empty:
                    all_parts.append(df); got_any = True; break
            except requests.HTTPError:
                pass
            # P2: with product-specific businessType
            try:
                base_bt = dict(base)
                base_bt["businessType"] = business_type
                xml = _api_get(base_bt)
                df = _parse_balancing_xml_to_df(xml)
                if not df.empty:
                    all_parts.append(df); got_any = True; break
            except requests.HTTPError:
                pass

        if not got_any:
            print(f"  [warn] A89 {business_type} {ps[:6]}: no data (TMA tried: {TMA_CANDIDATES})")
    return pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()


# ================== MAIN (SE) ==================
def get_SE_balancing_for_2024():
    results = {}
    for name, bt in PRODUCTS.items():
        print(f"\n=== Fetching {name} ({bt}) SE 2024 month-by-month ===")

        df_amt = fetch_amounts_SE_chunked(CONTROL_AREA_SE, bt, YEAR)
        if df_amt.empty:
            print(f"  [warn] No amount data returned for {name} across 2024.")

        df_prc = fetch_prices_SE_chunked(CONTROL_AREA_SE, bt, YEAR)
        if df_prc.empty:
            print(f"  [warn] No price data returned for {name} across 2024.")

        df = _merge_amt_prc(df_amt, df_prc)
        if not df.empty and "time_utc" in df.columns:
            df = df.sort_values("time_utc").reset_index(drop=True)
        df["product"] = name
        df = df[["time_utc","product","direction","quantity_MW","price_EUR_per_MW_h"]]

        out_path = os.path.join(SCRIPT_DIR, f"entsoe_SE_{YEAR}_{name}.csv")
        df.to_csv(out_path, index=False)
        print(f"  ✅ Saved: {out_path}  (rows={len(df)})")
        results[name] = df
    return results

# ================== DRIVER =====================
if __name__ == "__main__":
    data = get_SE_balancing_for_2024()
    for product, df in data.items():
        print(f"\nPreview {product} – Sweden:")
        print(df.head(5).to_string(index=False))