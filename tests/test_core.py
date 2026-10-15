import sys
from pathlib import Path

import control as ct
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dcmotor import MotorParams, state_space, transfer_function  # noqa: E402
from dcmotor.design import PID, LQRIntegral, lqr_design, simulate, step_metrics  # noqa: E402
from dcmotor.kalman import fit_level_seasonal, motor_kf  # noqa: E402
from dcmotor.whoop import load_whoop_csv, synthetic_hrv  # noqa: E402

p = MotorParams()


def test_plant_poles_and_gain():
    G = transfer_function(p)
    poles = np.sort(np.real(ct.poles(G)))
    assert poles == pytest.approx([-186.5637, -13.9363], abs=1e-3)
    assert ct.dcgain(G) == pytest.approx(p.Kt / (p.b * p.R + p.Kt * p.Ke))


def test_statespace_matches_transfer_function():
    A, B, _, C = state_space(p)
    G1 = ct.ss2tf(ct.ss(A, B, C, 0))
    assert ct.dcgain(G1) == pytest.approx(ct.dcgain(transfer_function(p)))
    assert np.sort(np.real(np.linalg.eigvals(A))) == pytest.approx(np.sort(np.real(ct.poles(transfer_function(p)))), abs=1e-6)


def test_pid_and_lqr_track_without_steady_state_error():
    pid = PID(0.1187, 2.111, 5.2e-4)
    Kx, Kz, Acl, *_ = lqr_design(p, q_w=10 ** 0.557, q_z=10 ** 3.386)
    assert np.all(np.real(np.linalg.eigvals(Acl)) < 0)
    for c in (pid, LQRIntegral(Kx, Kz)):
        m = step_metrics(simulate(c, p))
        assert m["sse_rads"] < 1e-3          # integral action, also after the load step
        assert m["overshoot_pct"] < 6.0


def test_anti_windup_reduces_overshoot():
    big = dict(r_fn=lambda t: 180.0, d_fn=lambda t: 0.0, T=1.2)
    with_aw = simulate(PID(0.1187, 2.111, 5.2e-4), p, **big).w.max()
    without = simulate(PID(0.1187, 2.111, 5.2e-4, antiwindup=False), p, **big).w.max()
    assert with_aw < without - 10


def test_kalman_filter_is_consistent():
    Kx, Kz, *_ = lqr_design(p, q_w=10 ** 0.557, q_z=10 ** 3.386)
    kf = motor_kf(p, sigma_meas=2.0, sigma_v=0.1, sigma_T=0.005)
    simulate(LQRIntegral(Kx, Kz, estimator=kf), p, T=10.0, seed=2, d_fn=lambda t: 0.0, sigma=2.0, sigma_v=0.1, sigma_T=0.005)
    nis = np.array(kf.nis)[1000:]
    assert abs(nis.mean() - 1.0) < 3 * np.sqrt(2 / len(nis)) * 1.5   # inside roughly a 99 % band for the mean of chi2(1)


def test_seasonal_smoother_recovers_weekday_effect():
    df, truth = synthetic_hrv(seed=5)
    fit = fit_level_seasonal(df["y"].to_numpy())
    wd = np.array([d.weekday() for d in df.index])
    est = np.array([fit["x_smooth"][wd == k, 1].mean() for k in range(7)])
    assert max(abs(est[k] - truth[k]) for k in range(7)) < 0.06


def test_whoop_loader_handles_export_format_and_gaps():
    df = load_whoop_csv(ROOT / "data" / "sample_whoop_format_SYNTHETIC.csv", "hrv")
    assert df["y"].notna().sum() > 80
    assert df.index.is_monotonic_increasing and (df.index[1] - df.index[0]).days == 1
    assert df["raw"].isna().any()      # missing days become NaN, not dropped
