"""Armature-controlled DC motor: continuous model, discretisation and analysis helpers.

Electrical:   L di/dt = V - R i - Ke w
Mechanical:   J dw/dt = Kt i - b w - T_load

State x = [i, w]^T, input u = V (volts), measured output y = w (rad/s),
disturbance d = T_load (N m).
"""
from __future__ import annotations

from dataclasses import dataclass

import control as ct
import numpy as np
from scipy.signal import cont2discrete


@dataclass(frozen=True)
class MotorParams:
    R: float = 1.0       # armature resistance [ohm]
    L: float = 5e-3      # armature inductance [H]
    Kt: float = 0.05     # torque constant [N m / A]
    Ke: float = 0.05     # back-EMF constant [V s / rad] (equals Kt in SI)
    J: float = 2e-4      # rotor + load inertia [kg m^2]
    b: float = 1e-4      # viscous friction [N m s / rad]
    v_max: float = 12.0  # supply limit [V]


def state_space(p: MotorParams = MotorParams()):
    """Return (A, B, Bd, C) with B for the voltage input and Bd for the load-torque input."""
    A = np.array([[-p.R / p.L, -p.Ke / p.L],
                  [p.Kt / p.J, -p.b / p.J]])
    B = np.array([[1.0 / p.L], [0.0]])
    Bd = np.array([[0.0], [-1.0 / p.J]])
    C = np.array([[0.0, 1.0]])
    return A, B, Bd, C


def transfer_function(p: MotorParams = MotorParams()) -> ct.TransferFunction:
    """w(s)/V(s) = Kt / ((J s + b)(L s + R) + Kt Ke)."""
    num = [p.Kt]
    den = np.polymul([p.J, p.b], [p.L, p.R]) + np.array([0.0, 0.0, p.Kt * p.Ke])
    return ct.tf(num, den)


def discretize(A, B, C, Ts, Bd=None):
    """Exact zero-order-hold discretisation. Bd (if given) is discretised with the same hold."""
    Bcat = B if Bd is None else np.hstack([B, Bd])
    Ad, Bcat_d, Cd, _, _ = cont2discrete((A, Bcat, C, np.zeros((C.shape[0], Bcat.shape[1]))), Ts, method="zoh")
    if Bd is None:
        return Ad, Bcat_d, Cd
    return Ad, Bcat_d[:, :1], Bcat_d[:, 1:], Cd
