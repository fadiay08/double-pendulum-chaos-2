"""Re-run the original 30 angle pairs from simulation_sweep.py.

Original method (kept exactly): point masses, M1 = M2 = 1 kg, L1 = L2 = 1 m, g = 9.81,
released from rest, twin run with both angles +0.1 deg, "chaos" = the two tips end up more
than 0.1 m apart at some point within 50 s.

Compared here:
  1. forward Euler, dt = 0.001 s  (the original code)
  2. RK4,           dt = 0.001 s  (same chaos test, accurate integrator)
  3. RK4 Lyapunov exponent over 400 s (chaos test that doesn't depend on a window or threshold)

Usage: python3 reanalysis_30.py  -> results/reanalysis_30.npz and a printed table
"""
from pathlib import Path

import numpy as np

import dp

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

PAIRS = [(30, 30), (30, 60), (30, 90), (30, 105), (30, 120),
         (45, 30), (45, 60), (45, 90), (45, 105), (45, 120),
         (60, 30), (60, 45), (60, 60), (60, 90), (60, 105), (60, 120),
         (75, 15), (75, 60), (75, 75), (75, 90),
         (90, 0), (90, 30), (90, 45), (90, 60), (90, 90),
         (120, 30), (120, 45), (120, 60), (120, 90), (120, 120)]

# What the original write-up reports for each pair (True = chaotic), from its results section:
# th1 = 30: chaos only at th2 = 120; th1 = 45: not described pair by pair in the text;
# th1 = 60: chaos from th2 = 105 up, plus (60, 45); th1 = 75: chaos from th2 = 90;
# th1 = 90 or 120: chaos for every th2.
P = dp.PARAMS


def tips(y, p=P):
    x1 = p["L1"] * np.sin(y[0]); y1 = -p["L1"] * np.cos(y[0])
    return x1 + p["L2"] * np.sin(y[1]), y1 - p["L2"] * np.cos(y[1])


def separation_time(th1, th2, step, dt, T=50.0, delta_deg=0.1, thresh=0.1, p=P):
    """First time (s) the two tips are more than `thresh` metres apart; NaN if never within T."""
    th1, th2 = np.asarray(th1, float), np.asarray(th2, float)
    a = dp.initial_state(th1, th2)
    b = dp.initial_state(th1 + delta_deg, th2 + delta_deg)
    N = a.shape[1]
    Y = np.concatenate([a, b], axis=1)
    t_sep = np.full(N, np.nan)
    n = int(round(T / dt))
    with np.errstate(all="ignore"):
        for i in range(1, n + 1):
            Y = step(Y, dt, p)
            xa, ya = tips(Y[:, :N], p)
            xb, yb = tips(Y[:, N:], p)
            d = np.hypot(xa - xb, ya - yb)
            new = np.isnan(t_sep) & (d > thresh)
            t_sep[new] = i * dt
    return t_sep


if __name__ == "__main__":
    th1 = np.array([a for a, _ in PAIRS], float)
    th2 = np.array([b for _, b in PAIRS], float)
    t_euler = separation_time(th1, th2, dp.euler_step, 0.001)
    t_rk4 = separation_time(th1, th2, dp.rk4_step, 0.001)
    r = dp.lyapunov(dp.initial_state(th1, th2), T=400, dt=0.002, checkpoints=[50, 100, 200])
    lo, _ = dp.flip_thresholds()
    K = (P["m1"] + P["m2"]) * P["L1"] * np.cos(np.deg2rad(th1)) + P["m2"] * P["L2"] * np.cos(np.deg2rad(th2))
    can_flip = K <= lo + 1e-12
    np.savez(OUT / "reanalysis_30.npz", th1=th1, th2=th2, t_euler=t_euler, t_rk4=t_rk4,
             lam=r["lam"], lam_T50=r["lam_at"][50], lam_T100=r["lam_at"][100], lam_T200=r["lam_at"][200],
             flipL=r["flip_lower"], flipU=r["flip_upper"], drift=r["drift"], can_flip=can_flip)
    print(" th1  th2 | Euler: split at | RK4: split at | lambda(400 s) | lower arm can flip | flipped in RK4 run")
    for i, (a, b) in enumerate(PAIRS):
        fe = "never" if np.isnan(t_euler[i]) else f"{t_euler[i]:6.2f} s"
        fr = "never" if np.isnan(t_rk4[i]) else f"{t_rk4[i]:6.2f} s"
        fl = "-" if np.isnan(r["flip_lower"][i]) else f"{r['flip_lower'][i]:.1f} s"
        print(f"{a:4d} {b:4d} | {fe:>15s} | {fr:>13s} | {r['lam'][i]:8.3f} /s   | {str(bool(can_flip[i])):>18s} | {fl}")
    print("max RK4 energy drift:", np.nanmax(r["drift"]))
