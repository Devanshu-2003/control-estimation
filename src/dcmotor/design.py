"""Controllers (PID with anti-windup, LQR with integral action), a sampled-data
simulator and the metrics/tuning code used to compare them on equal terms."""
from __future__ import annotations

from dataclasses import dataclass, field

import control as ct
import numpy as np
from scipy.optimize import minimize

from .plant import MotorParams, discretize, state_space, transfer_function

TS = 1e-3  # controller/plant sample time [s]


# --------------------------------------------------------------------------- controllers
class PID:
    """Discrete PID. Derivative acts on the measurement through a first-order filter
    (no set-point kick); the integrator uses back-calculation anti-windup."""

    needs_state = False

    def __init__(self, Kp, Ki, Kd=0.0, N=500.0, Tt=None, Ts=TS, antiwindup=True):
        self.Kp, self.Ki, self.Kd, self.N, self.Ts = Kp, Ki, Kd, N, Ts
        # back-calculation time constant: default to the integral time Ti = Kp/Ki (floored at 5 samples)
        self.Tt = Tt if Tt is not None else max(Kp / max(Ki, 1e-9), 5 * Ts)
        self.antiwindup = antiwindup
        self.reset()

    def reset(self):
        self.I = 0.0
        self.D = 0.0
        self.y_prev = 0.0
        self._u_cmd = 0.0

    def step(self, r, y, x_true=None):
        Ts = self.Ts
        e = r - y
        self.D = (self.D - self.Kd * self.N * (y - self.y_prev)) / (1.0 + self.N * Ts)
        self.y_prev = y
        u = self.Kp * e + self.I + self.D
        self._u_cmd, self._e = u, e
        return u

    def post(self, u_applied):
        aw = (u_applied - self._u_cmd) / self.Tt if self.antiwindup else 0.0
        self.I += self.Ts * (self.Ki * self._e + aw)

    # linear (unsaturated) controller transfer function, for margin analysis
    def tf(self):
        s = ct.tf("s")
        return self.Kp + self.Ki / s + self.Kd * self.N * s / (s + self.N)


class LQRIntegral:
    """Full-state LQR on [i, w, z] with z' = r - w (integral of speed error).

    State comes from `x_source`: 'true' (ideal full-state feedback) or an estimator object
    exposing .update(y) / .predict(u) / .x (e.g. dcmotor.kalman.KalmanFilter) -> LQG."""

    needs_state = True

    def __init__(self, Kx, Kz, Ts=TS, estimator=None, Tt=0.01):
        self.Kx, self.Kz, self.Ts, self.est, self.Tt = np.asarray(Kx).reshape(1, -1), float(Kz), Ts, estimator, Tt
        self.reset()

    def reset(self):
        self.z = 0.0
        self._u_cmd = 0.0
        self._e = 0.0
        if self.est is not None:
            self.est.reset()

    def step(self, r, y, x_true=None):
        if self.est is not None:
            self.est.update(y)
            x = self.est.x[:2]
            self._xhat = self.est.x.copy()
        else:
            x = x_true
            self._xhat = None
        self._e = r - y
        u = -(self.Kx @ x).item() - self.Kz * self.z
        self._u_cmd = u
        return u

    def post(self, u_applied):
        # back-calculation anti-windup: bleed z so that u_cmd = -Kz z + ... is pulled back toward the applied voltage
        # (d u / d z = -Kz, so the correction is divided by -Kz)
        self.z += self.Ts * (self._e + (u_applied - self._u_cmd) / (self.Tt * (-self.Kz)))
        if self.est is not None:
            self.est.predict(u_applied)


# --------------------------------------------------------------------------- simulator
@dataclass
class SimResult:
    t: np.ndarray
    r: np.ndarray
    w: np.ndarray        # true speed
    y: np.ndarray        # measured speed
    i: np.ndarray        # true current
    u_cmd: np.ndarray
    u: np.ndarray        # applied (saturated) voltage
    d: np.ndarray        # load torque
    xhat: np.ndarray | None = None
    extras: dict = field(default_factory=dict)


def simulate(ctrl, p: MotorParams = MotorParams(), T=2.0, Ts=TS, r_fn=lambda t: 100.0,
             d_fn=lambda t: 0.02 * (t >= 1.0), sigma=0.0, seed=0, saturate=True,
             sigma_v=0.0, sigma_T=0.0):
    """sigma: speed-sensor noise [rad/s]; sigma_v / sigma_T: per-sample white noise on the voltage [V] and load torque [N m]."""
    A, B, Bd, C = state_space(p)
    Ad, Bdd, Bdd_d, Cd = discretize(A, B, C, Ts, Bd=Bd)
    rng = np.random.default_rng(seed)
    n = int(round(T / Ts))
    ctrl.reset()
    x = np.zeros(2)
    out = {k: np.zeros(n) for k in ("t", "r", "w", "y", "i", "u_cmd", "u", "d")}
    xh = [] if getattr(ctrl, "est", None) is not None else None
    for k in range(n):
        t = k * Ts
        r, d = r_fn(t), d_fn(t)
        y = x[1] + sigma * rng.standard_normal()
        u_cmd = ctrl.step(r, y, x.copy())
        u = float(np.clip(u_cmd, -p.v_max, p.v_max)) if saturate else u_cmd
        ctrl.post(u)
        if xh is not None:
            xh.append(ctrl._xhat)
        for key, val in zip(out, (t, r, x[1], y, x[0], u_cmd, u, d)):
            out[key][k] = val
        nv = sigma_v * rng.standard_normal() if sigma_v > 0 else 0.0
        nT = sigma_T * rng.standard_normal() if sigma_T > 0 else 0.0
        x = Ad @ x + Bdd[:, 0] * (u + nv) + Bdd_d[:, 0] * (d + nT)
    return SimResult(**out, xhat=None if xh is None else np.array(xh))


# --------------------------------------------------------------------------- metrics
def step_metrics(res: SimResult, t_load=1.0, band=0.02):
    """Step-response and load-rejection metrics from a noise-free simulation."""
    t, w, r = res.t, res.w, res.r
    pre = t < t_load
    r_final = r[pre][-1]
    wp = w[pre]
    tp = t[pre]
    overshoot = max(0.0, (wp.max() - r_final) / r_final * 100.0)
    try:
        t10 = tp[np.argmax(wp >= 0.1 * r_final)]
        t90 = tp[np.argmax(wp >= 0.9 * r_final)]
        rise = t90 - t10
    except Exception:
        rise = np.nan
    outside = np.where(np.abs(wp - r_final) > band * r_final)[0]
    settle = tp[outside[-1] + 1] if len(outside) and outside[-1] + 1 < len(tp) else (0.0 if not len(outside) else np.nan)
    itae = np.trapezoid(tp * np.abs(r_final - wp), tp)
    post = t >= t_load
    if not post.any():  # run ended before the load step
        dip, recov = np.nan, np.nan
    else:
        dip = float(np.max(r[post] - w[post]))
        tt = t[post]
        ee = np.abs(r[post] - w[post])
        bad = np.where(ee > 1.0)[0]  # recovered when within 1 rad/s of the reference
        recov = (tt[bad[-1] + 1] - t_load) if len(bad) and bad[-1] + 1 < len(tt) else (0.0 if not len(bad) else np.nan)
    return dict(overshoot_pct=overshoot, rise_s=rise, settle_s=settle, itae=itae,
                peak_u_V=float(np.max(np.abs(res.u_cmd[pre]))), load_dip_rads=dip, load_recovery_s=recov,
                sse_rads=float(abs(r[-1] - w[-50:].mean())))


# --------------------------------------------------------------------------- tuning
SPEC = dict(overshoot_pct=5.0, peak_u_V=12.0, pm_deg=60.0)


def _cost(m, pm=None, spec=SPEC):
    c = m["itae"] + 0.5 * m["load_dip_rads"] + 2.0 * (0 if np.isnan(m["load_recovery_s"]) else m["load_recovery_s"])
    c += 50.0 * max(0.0, m["overshoot_pct"] - spec["overshoot_pct"])
    c += 5.0 * max(0.0, m["peak_u_V"] - spec["peak_u_V"])
    if pm is not None:
        c += 0.5 * max(0.0, spec["pm_deg"] - pm)
    if np.isnan(m["settle_s"]):
        c += 1e3
    return c


def loop_margins(L):
    gm, pm, wcg, wcp = ct.margin(L)
    return dict(gain_margin_db=20 * np.log10(gm) if np.isfinite(gm) else np.inf, phase_margin_deg=pm,
                wcg=wcg, wcp=wcp)


def tune_pid(p=MotorParams(), spec=SPEC, x0=(-0.5, 1.0, -3.0)):
    """Minimise ITAE + load-rejection cost subject to overshoot / peak-voltage / phase-margin penalties."""
    G = transfer_function(p)

    def build(z):
        Kp, Ki, Kd = 10 ** z[0], 10 ** z[1], 10 ** z[2]
        return PID(Kp, Ki, Kd)

    def f(z):
        if not (-4 < z[0] < 2 and -4 < z[1] < 4 and -6 < z[2] < 1):
            return 1e9
        c = build(z)
        res = simulate(c, p, T=2.0)
        m = step_metrics(res)
        pm = loop_margins(c.tf() * G)["phase_margin_deg"]
        return _cost(m, pm if np.isfinite(pm) else 0.0, spec)

    best = minimize(f, np.array(x0, float), method="Nelder-Mead",
                    options=dict(xatol=1e-3, fatol=1e-4, maxiter=400))
    return build(best.x), best


def lqr_design(p=MotorParams(), q_w=1.0, q_i=0.0, q_z=1.0, rho=1.0):
    """Continuous-time LQR on the integral-augmented model. Returns (Kx, Kz, closed-loop A)."""
    A, B, Bd, C = state_space(p)
    Aa = np.block([[A, np.zeros((2, 1))], [-C, np.zeros((1, 1))]])
    Ba = np.vstack([B, [[0.0]]])
    Q = np.diag([q_i, q_w, q_z])
    K, S, E = ct.lqr(Aa, Ba, Q, np.array([[rho]]))
    return K[0, :2], K[0, 2], (Aa - Ba @ K), Aa, Ba


def lqr_loop_tf(Kx, Kz, p=MotorParams()):
    """Loop transfer function broken at the plant input: L(s) = K_aug (sI - A_aug)^-1 B_aug."""
    _, _, _, Aa, Ba = lqr_design(p)
    K = np.hstack([Kx, [Kz]]).reshape(1, -1)
    return ct.ss(Aa, Ba, K, 0)


def tune_lqr(p=MotorParams(), spec=SPEC, x0=(-1.0, 1.0)):
    """Search log10(q_w), log10(q_z) (rho = 1 fixed, q_i = 0) with the same cost as the PID."""
    def build(z):
        Kx, Kz, *_ = lqr_design(p, q_w=10 ** z[0], q_z=10 ** z[1])
        return LQRIntegral(Kx, Kz)

    def f(z):
        if not (-8 < z[0] < 6 and -4 < z[1] < 8):
            return 1e9
        try:
            c = build(z)
        except Exception:
            return 1e9
        m = step_metrics(simulate(c, p, T=2.0))
        return _cost(m, None, spec)

    best = minimize(f, np.array(x0, float), method="Nelder-Mead",
                    options=dict(xatol=1e-3, fatol=1e-4, maxiter=300))
    return build(best.x), best
