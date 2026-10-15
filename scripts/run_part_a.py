"""Part A: open-loop analysis, PID and LQR(+integral) design, comparison and robustness.

Run from the repo root:  python scripts/run_part_a.py
Writes figures/a*.png and results/part_a_metrics.json.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import control as ct
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dcmotor import MotorParams, state_space, transfer_function  # noqa: E402
from dcmotor import style  # noqa: E402
from dcmotor.design import (PID, LQRIntegral, loop_margins, lqr_loop_tf, simulate,  # noqa: E402
                            step_metrics, tune_lqr, tune_pid)

style.apply()
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

p = MotorParams()
G = transfer_function(p)
A, B, Bd, C = state_space(p)
out = {}

# ------------------------------------------------------------------ 1. open-loop analysis
poles = ct.poles(G)
out["open_loop"] = dict(poles=[float(x.real) for x in poles], dc_gain_rads_per_V=float(ct.dcgain(G)),
                        no_load_speed_at_vmax_rads=float(ct.dcgain(G) * p.v_max),
                        time_constants_s=[float(-1 / x.real) for x in poles])

fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))
a = ax[0, 0]
a.axhline(0, color=style.AXIS, lw=0.8)
a.plot(poles.real, poles.imag, "x", color=style.BLUE, ms=9, mew=2)
for x in poles:
    a.annotate(f"s = {x.real:.1f}", (x.real, 0), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8.5, color=style.INK2)
a.set(title="Open-loop poles (both real, stable)", xlabel="Re(s)  [rad/s]", ylabel="Im(s)", ylim=(-1, 1), xlim=(-230, 20))
a.set_yticks([])

a = ax[0, 1]
tt = np.linspace(0, 0.5, 600)
_, yy = ct.step_response(G * p.v_max, tt)
a.plot(tt, yy, color=style.BLUE)
a.axhline(out["open_loop"]["no_load_speed_at_vmax_rads"], color=style.REF, ls="--", lw=0.9)
a.annotate(f"no-load speed at 12 V = {out['open_loop']['no_load_speed_at_vmax_rads']:.0f} rad/s",
           (0.5, out["open_loop"]["no_load_speed_at_vmax_rads"]), xytext=(-4, -12), textcoords="offset points", ha="right", fontsize=8.5, color=style.INK2)
a.set(title="Open-loop response to a 12 V step", xlabel="time [s]", ylabel="speed [rad/s]")

w = np.logspace(0, 4, 600)
mag, ph, _ = ct.frequency_response(G, w)
ax[1, 0].semilogx(w, 20 * np.log10(mag), color=style.BLUE)
ax[1, 0].set(title="Plant Bode: magnitude", xlabel="frequency [rad/s]", ylabel="|G| [dB]")
ax[1, 1].semilogx(w, np.degrees(np.unwrap(ph)), color=style.BLUE)
ax[1, 1].set(title="Plant Bode: phase", xlabel="frequency [rad/s]", ylabel="phase [deg]")
fig.suptitle("DC motor plant  w(s)/V(s)", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout()
style.save(fig, FIG / "a1_plant_open_loop.png")

# ------------------------------------------------------------------ 2. design
pid, r_pid = tune_pid(p)
lqr, r_lqr = tune_lqr(p)
out["pid"] = dict(Kp=pid.Kp, Ki=pid.Ki, Kd=pid.Kd, N=pid.N, Tt=pid.Tt)
out["lqr"] = dict(Kx_i=float(lqr.Kx[0, 0]), Kx_w=float(lqr.Kx[0, 1]), Kz=float(lqr.Kz))

L_pid = pid.tf() * G
L_lqr = lqr_loop_tf(lqr.Kx[0], lqr.Kz, p)
out["pid"]["margins"] = {k: float(v) for k, v in loop_margins(L_pid).items()}
out["lqr"]["margins"] = {k: float(v) for k, v in loop_margins(L_lqr).items()}
out["pid"]["closed_loop_poles"] = [complex(x).real for x in ct.poles(ct.feedback(L_pid, 1))]
out["lqr"]["closed_loop_poles"] = [complex(x).real for x in np.linalg.eigvals(
    np.block([[A, np.zeros((2, 1))], [-C, np.zeros((1, 1))]]) - np.vstack([B, [[0.0]]]) @ np.hstack([lqr.Kx, [[lqr.Kz]]]))]

# ------------------------------------------------------------------ 3. step + load disturbance
res = {"PID": simulate(pid, p), "LQR": simulate(lqr, p)}
out["step_100rads"] = {k: {m: float(v) for m, v in step_metrics(r).items()} for k, r in res.items()}

fig, ax = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
t = res["PID"].t
ax[0].plot(t, res["PID"].r, color=style.REF, ls="--", lw=1.0)
ax[0].plot(t, res["PID"].w, color=style.BLUE, label="PID (speed feedback only)")
ax[0].plot(t, res["LQR"].w, color=style.ORANGE, label="LQR + integral (full state)")
ax[0].axvline(1.0, color=style.AXIS, lw=0.8)
ax[0].annotate("load step 0.02 N·m", (1.0, 60), xytext=(5, 0), textcoords="offset points", fontsize=8.5, color=style.INK2)
ax[0].annotate("reference 100 rad/s", (1.9, 100), xytext=(0, 7), textcoords="offset points", ha="right", fontsize=8.5, color=style.INK2)
ax[0].set(ylabel="speed [rad/s]", title="Step and load-disturbance response (noise-free)")
ax[0].legend(loc="lower right")
ax[1].plot(t, res["PID"].u_cmd, color=style.BLUE)
ax[1].plot(t, res["LQR"].u_cmd, color=style.ORANGE)
ax[1].axhline(p.v_max, color=style.REF, ls=":", lw=0.9)
ax[1].annotate("12 V supply limit", (1.9, p.v_max), xytext=(0, -9), textcoords="offset points", ha="right", fontsize=8.5, color=style.INK2)
ax[1].set(ylabel="command [V]", xlabel="time [s]", ylim=(0, 14))
fig.tight_layout()
style.save(fig, FIG / "a2_step_and_disturbance.png")

# zoom on the first 0.4 s makes the rise visible
fig, ax = plt.subplots(figsize=(9, 3.6))
for k, c, lab in (("PID", style.BLUE, "PID"), ("LQR", style.ORANGE, "LQR + integral")):
    m = t <= 0.4
    ax.plot(t[m], res[k].w[m], color=c, label=lab)
ax.plot(t[t <= 0.4], res["PID"].r[t <= 0.4], color=style.REF, ls="--", lw=1.0)
ax.axhspan(98, 102, color=style.GRID, alpha=0.6, lw=0)
ax.annotate("±2 % settling band", (0.4, 102), xytext=(-2, 6), textcoords="offset points", ha="right", fontsize=8.5, color=style.INK2)
ax.set(title="First 0.4 s of the step", xlabel="time [s]", ylabel="speed [rad/s]")
ax.legend(loc="lower right")
style.save(fig, FIG / "a2b_step_zoom.png")

# ------------------------------------------------------------------ 4. loop Bode with margins
w = np.logspace(-1, 3.5, 800)
fig, ax = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
for L, c, lab in ((L_pid, style.BLUE, "PID loop"), (L_lqr, style.ORANGE, "LQR loop (broken at plant input)")):
    mg, pf, _ = ct.frequency_response(L, w)
    ax[0].semilogx(w, 20 * np.log10(np.squeeze(mg)), color=c, label=lab)
    ax[1].semilogx(w, np.degrees(np.unwrap(np.squeeze(pf))), color=c)
ax[0].axhline(0, color=style.REF, lw=0.9)
ax[0].set(ylabel="|L| [dB]", title="Loop transfer functions", ylim=(-60, 70))
ax[0].legend(loc="upper right")
ax[1].axhline(-180, color=style.REF, lw=0.9, ls="--")
ax[1].set(ylabel="phase [deg]", xlabel="frequency [rad/s]")
txt = []
for k, lab in (("pid", "PID"), ("lqr", "LQR")):
    mg = out[k]["margins"]
    gm = "inf" if not np.isfinite(mg["gain_margin_db"]) else f"{mg['gain_margin_db']:.1f} dB"
    txt.append(f"{lab}: phase margin {mg['phase_margin_deg']:.0f}° at {mg['wcp']:.0f} rad/s, gain margin {gm}")
ax[1].annotate("\n".join(txt), (0.02, 0.05), xycoords="axes fraction", fontsize=8.5, color=style.INK2, va="bottom")
fig.tight_layout()
style.save(fig, FIG / "a3_loop_bode.png")

# ------------------------------------------------------------------ 5. saturation / anti-windup (large step)
big = dict(r_fn=lambda t: 180.0, d_fn=lambda t: 0.0, T=1.2)
pid_nowu = PID(pid.Kp, pid.Ki, pid.Kd, antiwindup=False)
sims = {"PID, anti-windup": simulate(pid, p, **big), "PID, no anti-windup": simulate(pid_nowu, p, **big),
        "LQR + integral": simulate(lqr, p, **big)}
out["large_step_180rads"] = {}
for k, r in sims.items():
    out["large_step_180rads"][k] = dict(overshoot_pct=float(max(0.0, (r.w.max() - 180) / 180 * 100)),
                                        peak_speed_rads=float(r.w.max()))
cols = {"PID, anti-windup": (style.BLUE, "-"), "PID, no anti-windup": (style.AQUA, "--"), "LQR + integral": (style.ORANGE, "-")}
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
ax[0].plot(sims["LQR + integral"].t, sims["LQR + integral"].r, color=style.REF, ls="--", lw=1.0)
for k, r in sims.items():
    ax[0].plot(r.t, r.w, color=cols[k][0], ls=cols[k][1], label=k)
    ax[1].plot(r.t, r.u, color=cols[k][0], ls=cols[k][1])
ax[0].set(ylabel="speed [rad/s]", title="Large step (180 rad/s) with a 12 V supply limit")
ax[0].legend(loc="lower right")
ax[1].set(ylabel="applied voltage [V]", xlabel="time [s]", ylim=(0, 14))
fig.tight_layout()
style.save(fig, FIG / "a4_saturation_antiwindup.png")

# ------------------------------------------------------------------ 6. robustness to plant mismatch
cases = {"J x0.5": dict(J=p.J * 0.5), "J x2": dict(J=p.J * 2.0), "R x0.8": dict(R=p.R * 0.8), "R x1.2": dict(R=p.R * 1.2),
         "K x0.8": dict(Kt=p.Kt * 0.8, Ke=p.Ke * 0.8), "K x1.2": dict(Kt=p.Kt * 1.2, Ke=p.Ke * 1.2)}
out["robustness"] = {"PID": {}, "LQR": {}}
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
for a, (name, ctrl, col) in zip(ax, (("PID", pid, style.BLUE), ("LQR", lqr, style.ORANGE))):
    nom = simulate(ctrl, p, T=1.0, d_fn=lambda t: 0.0)
    for cname, kw in cases.items():
        r = simulate(ctrl, replace(p, **kw), T=1.0, d_fn=lambda t: 0.0)
        a.plot(r.t, r.w, color=col, alpha=0.35, lw=1.1)
        m = step_metrics(r, t_load=1.0)
        out["robustness"][name][cname] = dict(overshoot_pct=float(m["overshoot_pct"]), settle_s=float(m["settle_s"]))
    a.plot(nom.t, nom.w, color=col, lw=2.0)
    a.plot(nom.t, nom.r, color=style.REF, ls="--", lw=1.0)
    a.set(title=f"{name}: nominal (bold) and 6 mismatched plants", xlabel="time [s]")
    a.set_xlim(0, 0.6)
ax[0].set_ylabel("speed [rad/s]")
fig.tight_layout()
style.save(fig, FIG / "a5_robustness.png")

(RES / "part_a_metrics.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps({k: out[k] for k in ("open_loop", "pid", "lqr", "step_100rads", "large_step_180rads", "robustness")}, indent=1, default=float))
