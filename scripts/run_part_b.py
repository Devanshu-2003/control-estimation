"""Part B: Kalman filtering and LQG on the simulated DC motor.

Run from the repo root:  python scripts/run_part_b.py   (run_part_a.py need not be run first)
Writes figures/b*.png and results/part_b_metrics.json.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dcmotor import MotorParams, style  # noqa: E402
from dcmotor.design import LQRIntegral, simulate, tune_lqr, tune_pid  # noqa: E402
from dcmotor.kalman import motor_kf  # noqa: E402

style.apply()
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

p = MotorParams()
NOISE = dict(sigma=2.0, sigma_v=0.1, sigma_T=0.005)         # true noise injected by the simulator
KF_KW = dict(sigma_meas=2.0, sigma_v=0.1, sigma_T=0.005)    # what the filter is told (matched)

pid, _ = tune_pid(p)
lqr0, _ = tune_lqr(p)
Kx, Kz = lqr0.Kx, lqr0.Kz


def lqg(three_state=False, sigma_dw=0.002, **kw):
    return LQRIntegral(Kx, Kz, estimator=motor_kf(p, disturbance_state=three_state, sigma_dw=sigma_dw, **{**KF_KW, **kw}))


def rmse(a):
    return float(np.sqrt(np.mean(np.square(a))))


out = {"noise": NOISE}

# ------------------------------------------------------------------ B1: what the filter estimates
res = simulate(lqg(), p, T=2.0, seed=3, **NOISE)
t = res.t
w_hat, i_hat = res.xhat[:, 1], res.xhat[:, 0]
m = (t >= 0.05) & (t < 1.0)
out["estimation_2state"] = dict(
    speed_rmse_raw_measurement=rmse((res.y - res.w)[m]), speed_rmse_kf=rmse((w_hat - res.w)[m]),
    current_rmse_kf=rmse((i_hat - res.i)[m]), current_std_true=float(np.std(res.i[m])))

fig, ax = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
z = t <= 0.3
ax[0].plot(t[z], res.y[z], ".", color=style.MUTED, ms=3, alpha=0.7)
ax[0].plot(t[z], res.w[z], color=style.INK, lw=1.0)
ax[0].plot(t[z], w_hat[z], color=style.AQUA, lw=1.8)
ax[0].set(ylabel="speed [rad/s]", title="Speed: measurement, estimate and truth")
ax[0].legend(handles=[Line2D([], [], marker=".", ls="", color=style.MUTED, label="encoder reading (noise σ = 2 rad/s)"),
                      Line2D([], [], color=style.INK, lw=1.0, label="true speed"),
                      Line2D([], [], color=style.AQUA, lw=1.8, label="Kalman estimate")], loc="lower right")
ax[1].plot(t[z], res.i[z], color=style.INK, lw=1.0)
ax[1].plot(t[z], i_hat[z], color=style.AQUA, lw=1.8, ls="--")
ax[1].set(ylabel="current [A]", xlabel="time [s]", title="Armature current: never measured, estimated from the speed signal")
ax[1].legend(handles=[Line2D([], [], color=style.INK, lw=1.0, label="true current"),
                      Line2D([], [], color=style.AQUA, lw=1.8, ls="--", label="Kalman estimate")], loc="upper right")
fig.tight_layout()
style.save(fig, FIG / "b1_kalman_estimates.png")

# ------------------------------------------------------------------ B2: closed loop under noise, PID vs LQG (20 seeds)
def window_metrics(r, lo=0.5, hi=1.0):
    k = (r.t >= lo) & (r.t < hi)
    return dict(speed_err_rms=rmse(r.w[k] - r.r[k]), voltage_std=float(np.std(r.u[k])))


runs = {"PID": [], "LQG": []}
for seed in range(20):
    for name, c in (("PID", pid), ("LQG", lqg())):
        r = simulate(c, p, T=2.0, seed=seed, **NOISE)
        met = window_metrics(r)
        met["load_dip_rads"] = float(np.max(r.r[r.t >= 1.0] - r.w[r.t >= 1.0]))
        runs[name].append(met)
out["closed_loop_noise_20seeds"] = {
    n: {k: dict(mean=float(np.mean([m[k] for m in v])), std=float(np.std([m[k] for m in v]))) for k in v[0]}
    for n, v in runs.items()}

demo = {"PID": simulate(pid, p, T=2.0, seed=3, **NOISE), "LQG (LQR + Kalman)": simulate(lqg(), p, T=2.0, seed=3, **NOISE)}
cols = {"PID": style.BLUE, "LQG (LQR + Kalman)": style.ORANGE}
fig, ax = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
for k, r in demo.items():
    z = (r.t >= 0.5) & (r.t < 1.0)
    ax[0].plot(r.t[z], r.w[z], color=cols[k], label=k)
    ax[1].plot(r.t[z], r.u[z], color=cols[k], lw=1.0)
ax[0].axhline(100, color=style.REF, ls="--", lw=1.0)
ax[0].set(ylabel="true speed [rad/s]", title="Steady state with sensor and process noise")
ax[0].legend(loc="upper right")
ax[1].set(ylabel="applied voltage [V]", xlabel="time [s]")
fig.tight_layout()
style.save(fig, FIG / "b2_closed_loop_noise.png")

# ------------------------------------------------------------------ B3: filter consistency (NIS, innovation whiteness)
ctrl = lqg()  # keep a handle on the estimator: it stores the innovations
longrun = simulate(ctrl, p, T=20.0, seed=11, d_fn=lambda t: 0.0, **NOISE)
nis = np.array(ctrl.est.nis)[1000:]
innov = np.array(ctrl.est.innov)[1000:]
S = np.array(ctrl.est.S)[1000:]
N = len(nis)
lo, hi = chi2.ppf([0.025, 0.975], 1)
frac_in = float(np.mean((nis >= lo) & (nis <= hi)))
nu = innov / np.sqrt(S)
lags = np.arange(1, 31)
acf = np.array([np.corrcoef(nu[:-l], nu[l:])[0, 1] for l in lags])
band = 1.96 / np.sqrt(N)
out["consistency"] = dict(samples=int(N), mean_nis=float(nis.mean()), mean_nis_expected=1.0,
                          mean_nis_95pct_band=[1 - 1.96 * np.sqrt(2 / N), 1 + 1.96 * np.sqrt(2 / N)],
                          fraction_nis_inside_chi2_95=frac_in, acf_lag1=float(acf[0]),
                          acf_fraction_inside_band=float(np.mean(np.abs(acf) <= band)), acf_band=float(band))

win = 500
roll = np.convolve(nis, np.ones(win) / win, mode="valid")
tt = longrun.t[1000:][win - 1:]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
ax[0].plot(tt, roll, color=style.AQUA)
b = 1.96 * np.sqrt(2 / win)
ax[0].axhspan(1 - b, 1 + b, color=style.GRID, alpha=0.7, lw=0)
ax[0].axhline(1.0, color=style.REF, lw=0.9)
ax[0].annotate("expected 1.0 ± 95 % band", (tt[0], 1 + b), xytext=(4, 5), textcoords="offset points", fontsize=8.5, color=style.INK2)
ax[0].set(title=f"NIS, 500-sample mean (overall {nis.mean():.3f})", xlabel="time [s]", ylabel="normalised innovation²", ylim=(0.6, 1.4))
ax[1].axhspan(-band, band, color=style.GRID, alpha=0.7, lw=0)
ax[1].vlines(lags, 0, acf, color=style.AQUA, lw=2)
ax[1].plot(lags, acf, "o", color=style.AQUA, ms=4, mec=style.SURFACE)
ax[1].axhline(0, color=style.AXIS, lw=0.8)
ax[1].set(title="Innovation autocorrelation", xlabel="lag [samples]", ylabel="correlation")
ax[1].annotate("shaded: 95 % band for white noise", (1, band), xytext=(4, 6), textcoords="offset points", fontsize=8.5, color=style.INK2)
fig.tight_layout()
style.save(fig, FIG / "b3_filter_consistency.png")

# ------------------------------------------------------------------ B4: unknown load torque as a filter state
ctrls = {"2-state filter (no torque state)": lqg(False), "3-state filter (torque as random-walk state)": lqg(True)}
sim4 = {k: simulate(c, p, T=2.0, seed=5, **NOISE) for k, c in ctrls.items()}
k3 = sim4["3-state filter (torque as random-walk state)"]
k2 = sim4["2-state filter (no torque state)"]
post = (k3.t >= 1.3) & (k3.t < 2.0)
out["load_torque_estimation"] = dict(
    torque_estimate_mean_after_step=float(k3.xhat[post, 2].mean()), torque_true=0.02,
    current_bias_2state_A=float(np.mean(k2.xhat[post, 0] - k2.i[post])),
    current_bias_3state_A=float(np.mean(k3.xhat[post, 0] - k3.i[post])),
    current_rmse_2state_A=rmse(k2.xhat[post, 0] - k2.i[post]), current_rmse_3state_A=rmse(k3.xhat[post, 0] - k3.i[post]))
sweep = {}
for sdw in (0.0005, 0.002, 0.008):
    runs_d = [simulate(lqg(True, sigma_dw=sdw), p, T=1.5, seed=sd, **NOISE) for sd in range(30)]
    tt_ = runs_d[0].t
    d_all = np.array([r.xhat[:, 2] for r in runs_d])
    d_mean = d_all.mean(axis=0)                               # ensemble mean removes the noise-crossing bias
    up = np.where((tt_ >= 1.0) & (d_mean >= 0.018))[0]
    sweep[str(sdw)] = dict(torque_noise_std_before_step=float(np.mean(np.std(d_all[:, (tt_ > 0.5) & (tt_ < 1.0)], axis=1))),
                           time_to_90pct_s=float(tt_[up[0]] - 1.0) if len(up) else None)
out["load_torque_estimation"]["sigma_dw_sweep"] = sweep

def movmean(x, n=20):
    return np.convolve(x, np.ones(n) / n, mode="valid")

fig, ax = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True)
ax[0].plot(k3.t, k3.d, color=style.REF, ls="--", lw=1.0)
ax[0].plot(k3.t, k3.xhat[:, 2], color=style.AQUA, lw=1.4)
ax[0].set(ylabel="load torque [N·m]", title="Load torque inferred from the speed signal alone")
ax[0].legend(handles=[Line2D([], [], color=style.REF, ls="--", lw=1.0, label="true load torque"),
                      Line2D([], [], color=style.AQUA, lw=1.4, label="3-state Kalman estimate")], loc="upper left")
for key, ls in (("2-state filter (no torque state)", "--"), ("3-state filter (torque as random-walk state)", "-")):
    r = sim4[key]
    ax[1].plot(r.t[19:], movmean(r.xhat[:, 0] - r.i), color=style.AQUA, ls=ls, lw=1.5)
ax[1].axhline(0, color=style.AXIS, lw=0.8)
ax[1].axvline(1.0, color=style.AXIS, lw=0.8)
ax[1].annotate("load step", (1.0, 0.07), xytext=(5, 0), textcoords="offset points", fontsize=8.5, color=style.INK2)
ax[1].legend(handles=[Line2D([], [], color=style.AQUA, ls="--", lw=1.5, label="2-state filter (no torque state)"),
                      Line2D([], [], color=style.AQUA, ls="-", lw=1.5, label="3-state filter (torque state)")], loc="lower left")
ax[1].set(ylabel="current error î − i [A]", xlabel="time [s]", title="Current-estimate error around the load step (20-sample mean)")
fig.tight_layout()
style.save(fig, FIG / "b4_load_torque_state.png")

(RES / "part_b_metrics.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=1, default=float))
