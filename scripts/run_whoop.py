"""Kalman smoothing of a WHOOP daily series: slow baseline (random-walk level) + weekday effect.

    python scripts/run_whoop.py                          # synthetic demo with known ground truth
    python scripts/run_whoop.py --csv data/whoop.csv     # your export (metric: hrv | recovery | rhr)
    python scripts/run_whoop.py --csv data/whoop.csv --metric recovery

Writes figures/w*.png and results/whoop_<source>_<metric>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dcmotor import style  # noqa: E402
from dcmotor.kalman import fit_level_seasonal  # noqa: E402
from dcmotor.whoop import METRICS, load_whoop_csv, synthetic_hrv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=None, help="WHOOP export CSV; omit for the synthetic demo")
ap.add_argument("--metric", default="hrv", choices=list(METRICS))
ap.add_argument("--date-col", default=None)
ap.add_argument("--value-col", default=None)
args = ap.parse_args()

style.apply()
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

synthetic = args.csv is None
if synthetic:
    df, wk_true = synthetic_hrv()
    metric, src = "hrv", "synthetic"
else:
    df = load_whoop_csv(args.csv, args.metric, args.date_col, args.value_col)
    metric, src, wk_true = args.metric, "whoop", None
_, fwd, inv, ylabel = METRICS[metric]
tag = "SYNTHETIC DEMO DATA, not WHOOP" if synthetic else "WHOOP export"
y = df["y"].to_numpy(float)
n_obs, n_days = int(np.sum(~np.isnan(y))), len(y)
if n_obs < 28:
    sys.exit(f"need at least 4 weeks of data for a weekly-seasonal model (found {n_obs} observed days)")

# ------------------------------------------------------------------ fit (maximum likelihood) and smooth
fit = fit_level_seasonal(y, period=7)
xs, Ps = fit["x_smooth"], fit["P_smooth"]
level, level_sd = xs[:, 0], np.sqrt(Ps[:, 0, 0])
season, season_var = xs[:, 1], Ps[:, 1, 1]
dates = df.index

# weekday effect: average the smoothed seasonal state over all days of that weekday
wd = np.array([d.weekday() for d in dates])
names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
eff = np.array([season[wd == k].mean() for k in range(7)])
eff_pct = (np.exp(eff) - 1) * 100 if metric == "hrv" else None

resid = (fit["filt"]["v"] / np.sqrt(fit["filt"]["S"]))[14:]   # skip the first two weeks (initial-state burn-in)
resid = resid[~np.isnan(resid)]
out = dict(source=src, metric=metric, days=n_days, observed=n_obs, missing=n_days - n_obs,
           variances=fit["params"], loglik=float(fit["loglik"]),
           weekday_effect_log_or_raw={k: float(v) for k, v in zip(names, eff)},
           weekday_effect_pct=None if eff_pct is None else {k: float(v) for k, v in zip(names, eff_pct)},
           standardised_innovations=dict(mean=float(resid.mean()), std=float(resid.std()),
                                         lag1_acf=float(np.corrcoef(resid[:-1], resid[1:])[0, 1])),
           level_start=float(inv(level[0])), level_end=float(inv(level[-1])))
if synthetic:
    ok = ~np.isnan(y)
    out["ground_truth_check"] = dict(
        level_rmse_smoothed=float(np.sqrt(np.mean((level - df["level_true"]) ** 2))),
        level_rmse_raw_observations=float(np.sqrt(np.mean((y[ok] - df["level_true"].to_numpy()[ok]) ** 2))),
        weekday_effect_true={names[k]: wk_true[k] for k in range(7)},
        weekday_effect_max_abs_error=float(max(abs(eff[k] - wk_true[k]) for k in range(7))))

# ------------------------------------------------------------------ figure 1: baseline + data
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(dates, inv(y), ".", color=style.MUTED, ms=5, alpha=0.8)
ax.fill_between(dates, inv(level - 1.96 * level_sd), inv(level + 1.96 * level_sd), color=style.BLUE, alpha=0.15, lw=0)
ax.plot(dates, inv(level), color=style.BLUE, lw=2.0)
if synthetic:
    ax.plot(dates, inv(df["level_true"]), color=style.INK, lw=1.0, ls="--")
handles = [plt.Line2D([], [], marker=".", ls="", color=style.MUTED, label="daily value"),
           plt.Line2D([], [], color=style.BLUE, lw=2.0, label="smoothed baseline (95 % band)")]
if synthetic:
    handles.append(plt.Line2D([], [], color=style.INK, lw=1.0, ls="--", label="true baseline"))
ax.legend(handles=handles, loc="best")
ax.set(title=f"{ylabel.split(' [')[0]}: baseline estimated by a Kalman smoother  ({tag})", ylabel=ylabel)
fig.autofmt_xdate()
style.save(fig, FIG / f"w1_{src}_{metric}_baseline.png")

# ------------------------------------------------------------------ figure 2: weekday effect
fig, ax = plt.subplots(figsize=(8, 3.9))
xk = np.arange(7)
vals = eff_pct if eff_pct is not None else eff
unit = "% vs weekly average" if eff_pct is not None else ylabel.split("[")[-1].rstrip("]") + " vs weekly average"
ax.axhline(0, color=style.AXIS, lw=0.9)
ax.vlines(xk, 0, vals, color=style.BLUE, lw=3)
ax.plot(xk, vals, "o", color=style.BLUE, ms=7, mec=style.SURFACE, mew=1.5)
if synthetic:
    tr = np.array([wk_true[k] for k in range(7)])
    tr = (np.exp(tr) - 1) * 100
    ax.plot(xk, tr, "D", color=style.INK, ms=5, mfc=style.SURFACE)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=style.BLUE, label="estimated"),
                       plt.Line2D([], [], marker="D", ls="", mfc=style.SURFACE, color=style.INK, label="true (synthetic)")], loc="upper left")
for k, v in zip(xk, vals):
    ax.annotate(f"{v:+.1f}", (k, v), xytext=(0, 9 if v >= 0 else -14), textcoords="offset points", ha="center", fontsize=8.5, color=style.INK2)
ax.set_xticks(xk, names)
ax.set_ylim(min(vals.min() * 1.25, -3), max(vals.max() * 1.3, 3) + (4 if synthetic else 0))
ax.set(title=f"Weekday effect on {ylabel.split(' [')[0]} ({tag})", ylabel=unit)
style.save(fig, FIG / f"w2_{src}_{metric}_weekday.png")

(RES / f"whoop_{src}_{metric}.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=1, default=float))
