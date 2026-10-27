"""The real pendulum: uniform rods, L1 = 24 cm, L2 = 20 cm, same material (mass ~ length).

Usage: python3 rods_analysis.py pairs | map
"""
import sys
import time
from pathlib import Path

import numpy as np

import dp
import rods

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

L1, L2 = 0.24, 0.20
P_ROD = rods.coeffs("rods", m1=L1, m2=L2, L1=L1, L2=L2)        # mass proportional to length
DT = 0.001


def rod_step(Y, dt, p):
    return rods.rk4_step(Y, dt, p)


def rod_lyapunov(y0, T, checkpoints=None):
    return dp.lyapunov(y0, T=T, dt=DT, step=rod_step, p=P_ROD, checkpoints=checkpoints,
                       energy_fn=lambda y: rods.energy(y, **P_ROD), escale=rods.escale(P_ROD))


def rod_split_time(th1, th2, delta_deg=1.0, thresh=0.1 * (L1 + L2), T=30.0):
    """Two releases 1 degree apart (both angles): first time their rod tips are more than
    10% of the full reach (4.4 cm) apart. NaN if not within T."""
    a = dp.initial_state(th1, th2)
    b = dp.initial_state(np.asarray(th1, float) + delta_deg, np.asarray(th2, float) + delta_deg)
    N = a.shape[1]
    Y = np.concatenate([a, b], axis=1)
    t_sep = np.full(N, np.nan)
    with np.errstate(all="ignore"):
        for i in range(1, int(round(T / DT)) + 1):
            Y = rods.rk4_step(Y, DT, P_ROD)
            xa = L1 * np.sin(Y[0, :N]) + L2 * np.sin(Y[1, :N])
            ya = -L1 * np.cos(Y[0, :N]) - L2 * np.cos(Y[1, :N])
            xb = L1 * np.sin(Y[0, N:]) + L2 * np.sin(Y[1, N:])
            yb = -L1 * np.cos(Y[0, N:]) - L2 * np.cos(Y[1, N:])
            new = np.isnan(t_sep) & (np.hypot(xa - xb, ya - yb) > thresh)
            t_sep[new] = i * DT
    return t_sep


# The nine releases filmed in the experiment and what the write-up says was seen
EXPERIMENT = [((30, 90), "not stated"), ((30, 120), "chaotic"), ((60, 90), "stable"),
              ((60, 120), "chaotic"), ((75, 15), "chaotic"), ((75, 75), "stable"),
              ((75, 120), "borderline"), ((90, 0), "chaotic"), ((90, 90), "chaotic")]


def run_pairs():
    pairs = [p for p, _ in EXPERIMENT]
    th1 = np.array([a for a, _ in pairs], float)
    th2 = np.array([b for _, b in pairs], float)
    t0 = time.time()
    r = rod_lyapunov(dp.initial_state(th1, th2), T=200, checkpoints=[50, 100])
    ts = rod_split_time(th1, th2)
    pm = dp.lyapunov(dp.initial_state(th1, th2), T=400, dt=0.002)   # the write-up's 1 m point-mass model
    can = rods.lower_flip_possible(th1, th2, P_ROD)
    np.savez(OUT / "rods_pairs.npz", th1=th1, th2=th2, lam_rod=r["lam"], lam_rod_T50=r["lam_at"][50],
             lam_rod_T100=r["lam_at"][100], split_rod=ts, lam_point=pm["lam"], can_flip_rod=can,
             flip_rod=r["flip_lower"], drift=r["drift"], observed=np.array([o for _, o in EXPERIMENT]))
    print(f"done in {time.time() - t0:.0f}s, max energy drift {np.nanmax(r['drift']):.1e}")
    print(" release   | seen in experiment | rods: lambda | rods: 1-deg releases split at | rods: lower arm can flip | 1 m point masses: lambda")
    for i, ((a, b), obs) in enumerate(EXPERIMENT):
        sp = "never (30 s)" if np.isnan(ts[i]) else f"{ts[i]:.2f} s"
        print(f" ({a:3d},{b:3d}) | {obs:>18s} | {r['lam'][i]:7.2f} /s  | {sp:>29s} | {str(bool(can[i])):>24s} | {pm['lam'][i]:.3f} /s")


def run_map():
    th1 = np.arange(0, 180.0001, 3.0)
    th2 = np.arange(-180, 180.0001, 3.0)
    A1, A2 = np.meshgrid(th1, th2, indexing="ij")
    t0 = time.time()
    r = rod_lyapunov(dp.initial_state(A1.ravel(), A2.ravel()), T=100, checkpoints=[50])
    np.savez(OUT / "rods_map.npz", th1=th1, th2=th2, lam=r["lam"].reshape(A1.shape),
             lam_T50=r["lam_at"][50].reshape(A1.shape), drift=r["drift"].reshape(A1.shape),
             flipL=r["flip_lower"].reshape(A1.shape))
    print(f"rods map done in {time.time() - t0:.0f}s, max energy drift {np.nanmax(r['drift']):.1e}")


if __name__ == "__main__":
    for job in sys.argv[1:]:
        dict(pairs=run_pairs, map=run_map)[job]()
