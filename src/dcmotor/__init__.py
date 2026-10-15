"""DC motor control and estimation: PID, LQR with integral action, Kalman filtering."""
from .plant import MotorParams, state_space, transfer_function, discretize  # noqa: F401
