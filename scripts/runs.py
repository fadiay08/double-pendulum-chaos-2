"""Compute all results and save them to results/*.npz.

Usage: python3 runs.py sweep | euler | energy | map | dtcheck
"""
import sys
import time
from pathlib import Path

import numpy as np

import dp

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def two_configs(ang):
    """Release from rest with the lower arm hanging straight down (th2 = 0)
    and with both arms in a straight line (th2 = th1)."""
    return np.concatenate([dp.initial_state(ang, 0), dp.initial_state(ang, ang)], axis=1)


def run_sweep():
    ang = np.arange(0, 180.0001, 0.25)
    t0 = time.time()
    r = dp.lyapunov(two_configs(ang), T=400, dt=0.002, checkpoints=[50, 100, 200, 400])
    n = len(ang)
    np.savez(OUT / "sweep_rk4.npz", ang=ang,
             lam_down=r["lam"][:n], lam_line=r["lam"][n:],
             **{f"lam_down_T{t}": v[:n] for t, v in r["lam_at"].items()},
             **{f"lam_line_T{t}": v[n:] for t, v in r["lam_at"].items()},
             drift_down=r["drift"][:n], drift_line=r["drift"][n:],
             flipL_down=r["flip_lower"][:n], flipL_line=r["flip_lower"][n:],
             flipU_down=r["flip_upper"][:n], flipU_line=r["flip_upper"][n:])
    print(f"sweep done in {time.time() - t0:.0f}s, max energy drift {np.nanmax(r['drift']):.1e}")


def run_dtcheck():
    """Same sweep on a 1 degree grid with half the step size, for convergence."""
    ang = np.arange(0, 180.0001, 1.0)
    n = len(ang)
    t0 = time.time()
    a = dp.lyapunov(two_configs(ang), T=200, dt=0.002)
    b = dp.lyapunov(two_configs(ang), T=200, dt=0.001)
    np.savez(OUT / "dtcheck.npz", ang=ang, lam_002=a["lam"], lam_001=b["lam"], n=n)
    print(f"dtcheck done in {time.time() - t0:.0f}s")


def run_euler():
    ang = np.arange(0, 180.0001, 1.0)
    n = len(ang)
    y0 = two_configs(ang)
    dts = np.array([0.01, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0003, 0.0002, 0.0001])
    lam = np.zeros((len(dts), 2 * n))
    gain = np.zeros((len(dts), 2 * n))
    t0 = time.time()
    for i, dt in enumerate(dts):
        r = dp.lyapunov(y0, T=50, dt=dt, step=dp.euler_step)
        lam[i], gain[i] = r["lam"], r["drift"]
    ref = dp.lyapunov(y0, T=50, dt=0.002)
    np.savez(OUT / "euler.npz", ang=ang, dts=dts, lam=lam, gain=gain,
             lam_rk4=ref["lam"], n=n)
    print(f"euler done in {time.time() - t0:.0f}s")


def run_energy():
    y0 = dp.initial_state(90, 0)
    out = {}
    for name, step, dt in [("euler_0.01", dp.euler_step, 0.01),
                           ("euler_0.001", dp.euler_step, 0.001),
                           ("euler_0.0001", dp.euler_step, 0.0001),
                           ("rk4_0.002", dp.rk4_step, 0.002)]:
        ts, ys = dp.integrate(y0, 60.0, dt, step=step, every=int(round(0.1 / dt)))
        E = dp.energy(ys[:, :, 0].T, **dp.PARAMS)
        out[f"t_{name}"] = ts
        out[f"dE_{name}"] = (E - E[0]) / dp.energy_scale()
    np.savez(OUT / "energy.npz", **out)
    print("energy done")


def run_map():
    th1 = np.arange(0, 180.0001, 2.0)
    th2 = np.arange(-180, 180.0001, 2.0)
    A1, A2 = np.meshgrid(th1, th2, indexing="ij")
    y0 = dp.initial_state(A1.ravel(), A2.ravel())
    t0 = time.time()
    r = dp.lyapunov(y0, T=200, dt=0.004, checkpoints=[100, 200])
    np.savez(OUT / "map.npz", th1=th1, th2=th2,
             lam=r["lam"].reshape(A1.shape),
             lam_T100=r["lam_at"][100].reshape(A1.shape),
             drift=r["drift"].reshape(A1.shape),
             flipL=r["flip_lower"].reshape(A1.shape),
             flipU=r["flip_upper"].reshape(A1.shape))
    print(f"map done in {time.time() - t0:.0f}s, max energy drift {np.nanmax(r['drift']):.1e}")


if __name__ == "__main__":
    jobs = dict(sweep=run_sweep, dtcheck=run_dtcheck, euler=run_euler,
                energy=run_energy, map=run_map)
    for name in sys.argv[1:]:
        jobs[name]()
