"""Load WHOOP daily data (CSV export) and generate a synthetic stand-in with known ground truth.

Expected input is the WHOOP data-export style file ("physiological_cycles.csv"): one row per cycle with
columns such as 'Cycle start time', 'Recovery score %', 'Resting heart rate (bpm)',
'Heart rate variability (ms)'. Column matching is case-insensitive and fuzzy; pass `date_col` / `value_col`
explicitly if your export (e.g. from the WHOOP API / MCP server) uses other names."""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = {
    # key: (substrings that identify the column, transform, inverse, axis label)
    "hrv": (("heart rate variability", "hrv", "rmssd"), np.log, np.exp, "HRV [ms]"),
    "recovery": (("recovery score", "recovery"), lambda x: x, lambda x: x, "recovery score [%]"),
    "rhr": (("resting heart rate", "rhr"), lambda x: x, lambda x: x, "resting HR [bpm]"),
}
DATE_HINTS = ("cycle start", "start time", "date", "day", "timestamp", "created")


def _find(cols, hints, exclude=()):
    low = {c: c.lower() for c in cols}
    for h in hints:
        for c, l in low.items():
            if h in l and not any(e in l for e in exclude):
                return c
    return None


def load_whoop_csv(path, metric="hrv", date_col=None, value_col=None):
    """Return a DataFrame indexed by calendar day (gaps filled with NaN) with columns raw, y (transformed)."""
    df = pd.read_csv(path)
    key = METRICS[metric]
    value_col = value_col or _find(df.columns, key[0], exclude=("end",))
    date_col = date_col or _find(df.columns, DATE_HINTS, exclude=("end", "timezone"))
    if value_col is None or date_col is None:
        raise ValueError(f"could not identify columns for '{metric}'. Columns found: {list(df.columns)}")
    d = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce", utc=False),
                      "raw": pd.to_numeric(df[value_col], errors="coerce")}).dropna(subset=["date"])
    try:
        d["date"] = d["date"].dt.tz_localize(None)
    except TypeError:
        pass
    d["date"] = d["date"].dt.normalize()
    d = d.groupby("date", as_index=True)["raw"].mean()
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
    d.loc[d <= 0] = np.nan
    out = pd.DataFrame({"raw": d})
    out["y"] = key[1](out["raw"].astype(float))
    return out


def synthetic_hrv(n=140, seed=7, start="2026-06-01"):
    """Synthetic daily ln(HRV): drifting baseline + weekday effect (Saturday dip) + noise + missing days.
    Ground truth is returned so the estimator can be checked against it."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    level = 4.05 + np.cumsum(rng.normal(0, 0.012, n)) - 0.0012 * np.arange(n)
    wk_true = {0: 0.02, 1: 0.03, 2: 0.01, 3: 0.0, 4: -0.03, 5: -0.14, 6: 0.11}   # Mon..Sun, sums to ~0
    wk = np.array([wk_true[d.weekday()] for d in dates])
    y = level + wk + rng.normal(0, 0.09, n)
    y[rng.choice(n, 6, replace=False)] = np.nan
    df = pd.DataFrame({"y": y, "raw": np.exp(y), "level_true": level, "season_true": wk}, index=dates)
    return df, wk_true
