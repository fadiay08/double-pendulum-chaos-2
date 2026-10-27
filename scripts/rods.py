"""General double pendulum: covers point masses and real rods with one set of equations.

Any planar double pendulum (rigid arms, frictionless hinges) has
    T = 1/2 A w1^2 + 1/2 B w2^2 + C w1 w2 cos(th1 - th2)
    V = -D cos(th1) - F cos(th2)
and only the five numbers A, B, C, D, F depend on how the mass is distributed.

Point masses m1, m2 at the ends of massless rods L1, L2:
    A = (m1+m2) L1^2, B = m2 L2^2, C = m2 L1 L2, D = (m1+m2) g L1, F = m2 g L2
Uniform thin rods of mass m1, m2 (pivot at one end, the second rod hinged at the end of the first):
    A = (m1/3 + m2) L1^2, B = m2 L2^2 / 3, C = m2 L1 L2 / 2, D = (m1/2 + m2) g L1, F = m2 g L2 / 2
"""
import numpy as np


def coeffs(kind, m1, m2, L1, L2, g=9.81):
    if kind == "points":
        return dict(A=(m1 + m2) * L1 ** 2, B=m2 * L2 ** 2, C=m2 * L1 * L2,
                    D=(m1 + m2) * g * L1, F=m2 * g * L2)
    if kind == "rods":
        return dict(A=(m1 / 3 + m2) * L1 ** 2, B=m2 * L2 ** 2 / 3, C=m2 * L1 * L2 / 2,
                    D=(m1 / 2 + m2) * g * L1, F=m2 * g * L2 / 2)
    raise ValueError(kind)


def accel(th1, th2, w1, w2, A, B, C, D, F):
    """Solve the two Euler-Lagrange equations
        A a1 + C cos(d) a2 = -C sin(d) w2^2 - D sin(th1)
        C cos(d) a1 + B a2 =  C sin(d) w1^2 - F sin(th2)
    for the angular accelerations a1, a2 (d = th1 - th2)."""
    d = th1 - th2
    c, s = np.cos(d), np.sin(d)
    r1 = -C * s * w2 ** 2 - D * np.sin(th1)
    r2 = C * s * w1 ** 2 - F * np.sin(th2)
    det = A * B - (C * c) ** 2
    return (B * r1 - C * c * r2) / det, (A * r2 - C * c * r1) / det


def deriv(y, p):
    a1, a2 = accel(y[0], y[1], y[2], y[3], **p)
    return np.stack([y[2], y[3], a1, a2])


def rk4_step(y, dt, p):
    k1 = deriv(y, p)
    k2 = deriv(y + 0.5 * dt * k1, p)
    k3 = deriv(y + 0.5 * dt * k2, p)
    k4 = deriv(y + dt * k3, p)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def energy(y, A, B, C, D, F):
    th1, th2, w1, w2 = y
    return (0.5 * A * w1 ** 2 + 0.5 * B * w2 ** 2 + C * w1 * w2 * np.cos(th1 - th2)
            - D * np.cos(th1) - F * np.cos(th2))


def escale(p):
    """Energy unit for reporting drift: D + F."""
    return p["D"] + p["F"]


def lower_flip_possible(th1_deg, th2_deg, p):
    """Released from rest, can energy ever carry the lower arm over the top?
    Needs E >= V(0, 180 deg) = -D + F, i.e. D cos(th1) + F cos(th2) <= D - F."""
    K = p["D"] * np.cos(np.deg2rad(th1_deg)) + p["F"] * np.cos(np.deg2rad(th2_deg))
    return K <= p["D"] - p["F"] + 1e-12


def upper_flip_possible(th1_deg, th2_deg, p):
    """Needs E >= V(180 deg, 0) = D - F, i.e. D cos(th1) + F cos(th2) <= -D + F."""
    K = p["D"] * np.cos(np.deg2rad(th1_deg)) + p["F"] * np.cos(np.deg2rad(th2_deg))
    return K <= -p["D"] + p["F"] + 1e-12
