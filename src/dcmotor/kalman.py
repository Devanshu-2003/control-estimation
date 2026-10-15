"""Kalman filtering: an online filter class (for closed-loop LQG), batch filter/RTS smoother with
missing-data handling and maximum-likelihood tuning (for the WHOOP time series), and motor models."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .plant import MotorParams, discretize, state_space


# --------------------------------------------------------------------------- online filter
class KalmanFilter:
    """Linear Kalman filter, Joseph-form covariance update.

    Order of use inside a control loop at step k:  update(y_k) -> compute u_k -> predict(u_k).
    Records innovations and their covariance so the filter can be checked for consistency (NIS)."""

    def __init__(self, Ad, Bd, C, Q, R, x0=None, P0=None):
        self.Ad, self.Bd, self.C = np.atleast_2d(Ad), np.atleast_2d(Bd), np.atleast_2d(C)
        self.Q, self.R = np.atleast_2d(Q), np.atleast_2d(R)
        n = self.Ad.shape[0]
        self._x0 = np.zeros(n) if x0 is None else np.asarray(x0, float)
        self._P0 = np.eye(n) if P0 is None else np.asarray(P0, float)
        self.reset()

    def reset(self):
        self.x, self.P = self._x0.copy(), self._P0.copy()
        self.innov, self.S, self.nis = [], [], []

    def predict(self, u=0.0):
        self.x = self.Ad @ self.x + self.Bd[:, 0] * u
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q

    def update(self, y):
        if y is None or np.isnan(y):
            self.innov.append(np.nan); self.S.append(np.nan); self.nis.append(np.nan)
            return
        y = np.atleast_1d(y)
        v = y - self.C @ self.x
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ v
        I_KC = np.eye(len(self.x)) - K @ self.C
        self.P = I_KC @ self.P @ I_KC.T + K @ self.R @ K.T
        self.innov.append(float(v[0])); self.S.append(float(S[0, 0])); self.nis.append(float(v @ np.linalg.solve(S, v)))


def motor_kf(p: MotorParams = MotorParams(), Ts=1e-3, sigma_meas=2.0, sigma_v=0.1, sigma_T=0.005,
             disturbance_state=False, sigma_dw=0.002, P0_diag=None):
    """Kalman filter for the DC motor. Measurement: speed only. States [i, w] (or [i, w, T_load]).

    The process-noise covariance is built from the same per-sample voltage / torque noise that the
    simulator injects, so with matched settings the filter is exactly consistent."""
    A, B, Bd, C = state_space(p)
    Ad, Bdd, Bdd_d, Cd = discretize(A, B, C, Ts, Bd=Bd)
    W = Bdd @ Bdd.T * sigma_v ** 2 + Bdd_d @ Bdd_d.T * sigma_T ** 2
    R = np.array([[sigma_meas ** 2]])
    if not disturbance_state:
        P0 = np.diag([1.0, 25.0]) if P0_diag is None else np.diag(P0_diag)
        return KalmanFilter(Ad, Bdd, Cd, W, R, np.zeros(2), P0)
    A3 = np.block([[Ad, Bdd_d], [np.zeros((1, 2)), np.ones((1, 1))]])
    B3 = np.vstack([Bdd, [[0.0]]])
    C3 = np.hstack([Cd, [[0.0]]])
    Q3 = np.zeros((3, 3))
    Q3[:2, :2] = W
    Q3[2, 2] = sigma_dw ** 2
    P0 = np.diag([1.0, 25.0, 1e-4]) if P0_diag is None else np.diag(P0_diag)
    return KalmanFilter(A3, B3, C3, Q3, R, np.zeros(3), P0)


# --------------------------------------------------------------------------- batch filter / smoother
def kalman_batch(F, H, Q, R, y, x0, P0):
    """Time-invariant Kalman filter over a series y (NaN = missing -> prediction only).

    Returns dict with predicted/filtered means and covariances, innovations, S and the log-likelihood."""
    n, T = F.shape[0], len(y)
    xp, Pp = np.zeros((T, n)), np.zeros((T, n, n))
    xf, Pf = np.zeros((T, n)), np.zeros((T, n, n))
    v, S = np.full(T, np.nan), np.full(T, np.nan)
    ll = 0.0
    x, P = x0.copy(), P0.copy()
    for t in range(T):
        xp[t], Pp[t] = x, P
        if np.isnan(y[t]):
            xf[t], Pf[t] = x, P
        else:
            St = (H @ P @ H.T).item() + R
            K = (P @ H.T / St).ravel()
            vt = y[t] - (H @ x).item()
            x = x + K * vt
            P = P - np.outer(K, H @ P)
            P = 0.5 * (P + P.T)
            xf[t], Pf[t], v[t], S[t] = x, P, vt, St
            ll += -0.5 * (np.log(2 * np.pi * St) + vt ** 2 / St)
        x, P = F @ x, F @ P @ F.T + Q
    return dict(xp=xp, Pp=Pp, xf=xf, Pf=Pf, v=v, S=S, loglik=ll)


def rts_smooth(F, filt):
    """Rauch-Tung-Striebel fixed-interval smoother on the output of kalman_batch."""
    xf, Pf, xp, Pp = filt["xf"], filt["Pf"], filt["xp"], filt["Pp"]
    T, n = xf.shape
    xs, Ps = xf.copy(), Pf.copy()
    for t in range(T - 2, -1, -1):
        J = Pf[t] @ F.T @ np.linalg.pinv(Pp[t + 1])
        xs[t] = xf[t] + J @ (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T
    return xs, Ps


# --------------------------------------------------------------------------- structural model: level + weekly seasonal
def level_seasonal_model(var_level, var_season, var_obs, period=7):
    """y_t = level_t + season_t + noise;  level: random walk;  season: dummy form with sum-to-zero constraint."""
    m = period - 1
    n = 1 + m
    F = np.zeros((n, n))
    F[0, 0] = 1.0
    F[1, 1:] = -1.0
    for j in range(2, n):
        F[j, j - 1] = 1.0
    H = np.zeros((1, n)); H[0, 0] = 1.0; H[0, 1] = 1.0
    Q = np.zeros((n, n)); Q[0, 0] = var_level; Q[1, 1] = var_season
    return F, H, Q, var_obs


def fit_level_seasonal(y, period=7):
    """Maximum-likelihood fit of (var_level, var_season, var_obs) with a diffuse-ish initial state."""
    ybar = np.nanmean(y)
    yvar = np.nanvar(y)
    n = period
    x0 = np.zeros(n); x0[0] = ybar
    P0 = np.eye(n) * 1e1 * max(yvar, 1e-6)

    def nll(theta):
        vl, vs, vo = np.exp(theta)
        F, H, Q, R = level_seasonal_model(vl, vs, vo, period)
        return -kalman_batch(F, H, Q, R, y, x0, P0)["loglik"]

    th0 = np.log([yvar * 0.01, yvar * 0.001, yvar * 0.5])
    best = minimize(nll, th0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=800))
    vl, vs, vo = np.exp(best.x)
    F, H, Q, R = level_seasonal_model(vl, vs, vo, period)
    filt = kalman_batch(F, H, Q, R, y, x0, P0)
    xs, Ps = rts_smooth(F, filt)
    return dict(params=dict(var_level=vl, var_season=vs, var_obs=vo), loglik=-best.fun, F=F, H=H, Q=Q, R=R,
                filt=filt, x_smooth=xs, P_smooth=Ps, x0=x0, P0=P0)
