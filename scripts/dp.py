"""Double pendulum: equations of motion, integrators, energy, Lyapunov exponents.

Model: two point masses m1, m2 on massless rigid rods L1, L2, no friction.
Angles th1, th2 are measured from the downward vertical.
State arrays have shape (4, N): rows are th1, th2, w1, w2, so N initial
conditions are integrated at once (vectorised with NumPy).
"""
import numpy as np

G = 9.81
PARAMS = dict(m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=G)


def accel(th1, th2, w1, w2, m1, m2, L1, L2, g):
    """Angular accelerations from the Euler-Lagrange equations (solved for a1, a2)."""
    d = th1 - th2
    sd, cd = np.sin(d), np.cos(d)
    den = 2 * m1 + m2 - m2 * np.cos(2 * d)
    a1 = (-g * (2 * m1 + m2) * np.sin(th1)
          - m2 * g * np.sin(th1 - 2 * th2)
          - 2 * sd * m2 * (w2 ** 2 * L2 + w1 ** 2 * L1 * cd)) / (L1 * den)
    a2 = (2 * sd * (w1 ** 2 * L1 * (m1 + m2)
                    + g * (m1 + m2) * np.cos(th1)
                    + w2 ** 2 * L2 * m2 * cd)) / (L2 * den)
    return a1, a2


def deriv(y, p):
    a1, a2 = accel(y[0], y[1], y[2], y[3], **p)
    return np.stack([y[2], y[3], a1, a2])


def euler_step(y, dt, p):
    """Forward (explicit) Euler: one derivative evaluation per step."""
    return y + dt * deriv(y, p)


def rk4_step(y, dt, p):
    """Classic 4th-order Runge-Kutta."""
    k1 = deriv(y, p)
    k2 = deriv(y + 0.5 * dt * k1, p)
    k3 = deriv(y + 0.5 * dt * k2, p)
    k4 = deriv(y + dt * k3, p)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def energy(y, m1, m2, L1, L2, g):
    th1, th2, w1, w2 = y
    T = (0.5 * (m1 + m2) * L1 ** 2 * w1 ** 2 + 0.5 * m2 * L2 ** 2 * w2 ** 2
         + m2 * L1 * L2 * w1 * w2 * np.cos(th1 - th2))
    V = -(m1 + m2) * g * L1 * np.cos(th1) - m2 * g * L2 * np.cos(th2)
    return T + V


def initial_state(th1_deg, th2_deg):
    """Released from rest at the given angles (degrees). Returns shape (4, N)."""
    th1 = np.deg2rad(np.atleast_1d(np.asarray(th1_deg, dtype=float)))
    th2 = np.deg2rad(np.atleast_1d(np.asarray(th2_deg, dtype=float)))
    th1, th2 = np.broadcast_arrays(th1, th2)
    z = np.zeros_like(th1)
    return np.stack([th1, th2, z, z])


def integrate(y0, T, dt, step=rk4_step, p=PARAMS, every=1):
    """Integrate and return times and states (states shape: (n_saved, 4, N))."""
    n = int(round(T / dt))
    y = y0.copy()
    ts, ys = [0.0], [y.copy()]
    with np.errstate(all="ignore"):
        for i in range(1, n + 1):
            y = step(y, dt, p)
            if i % every == 0:
                ts.append(i * dt)
                ys.append(y.copy())
    return np.array(ts), np.array(ys)


def energy_scale(p=PARAMS):
    """Energy unit used to report drift: m1*g*L1 + m2*g*(L1+L2), which is the
    energy difference between hanging straight down and the reference level."""
    return p["m1"] * p["g"] * p["L1"] + p["m2"] * p["g"] * (p["L1"] + p["L2"])


def flip_thresholds(p=PARAMS):
    """Energy conditions for flips, for a release from rest at (th1, th2).

    With K = (m1+m2)*L1*cos(th1) + m2*L2*cos(th2):
      lower arm can go over the top only if K <= (m1+m2)*L1 - m2*L2
      upper arm can go over the top only if K <= -(m1+m2)*L1 + m2*L2
    Returns the two right-hand sides.
    """
    a = (p["m1"] + p["m2"]) * p["L1"]
    b = p["m2"] * p["L2"]
    return a - b, -a + b


def lyapunov(y0, T=200.0, dt=0.002, step=rk4_step, p=PARAMS,
             d0=1e-8, renorm_dt=0.05, seed=0, checkpoints=None,
             energy_fn=None, escale=None):
    """Largest Lyapunov exponent (1/s) for each initial condition, Benettin method.

    A twin trajectory starts a distance d0 away in phase space. Every renorm_dt
    seconds we measure how far it has drifted, add log(d/d0) to a running sum,
    and pull it back to distance d0 along the same direction. The exponent is
    the average growth rate: sum / elapsed time.

    Returns a dict with:
      lam        exponent at time T, shape (N,)
      lam_at     {t: exponent at checkpoint time t} for each t in checkpoints
      drift      |E(T) - E(0)| / energy_scale for the reference trajectory
      flip_lower time (s) of the lower arm's first flip over the top, NaN if never
      flip_upper same for the upper arm
    """
    rng = np.random.default_rng(seed)
    N = y0.shape[1]
    u = rng.normal(size=y0.shape)
    u /= np.linalg.norm(u, axis=0)
    Y = np.concatenate([y0, y0 + d0 * u], axis=1)  # reference | twin
    k = max(1, int(round(renorm_dt / dt)))
    block = k * dt
    n_blocks = int(round(T / block))
    checkpoints = sorted(checkpoints or [])
    cp_blocks = {int(round(t / block)): t for t in checkpoints}
    lam_at = {}
    log_sum = np.zeros(N)
    flip_lower = np.full(N, np.nan)
    flip_upper = np.full(N, np.nan)
    if energy_fn is None:                      # default: point-mass model with parameters p
        energy_fn = lambda y: energy(y, **p)
        escale = energy_scale(p)
    E0 = energy_fn(y0)
    with np.errstate(all="ignore"):
        for b in range(1, n_blocks + 1):
            for _ in range(k):
                Y = step(Y, dt, p)
            ref, twin = Y[:, :N], Y[:, N:]
            diff = twin - ref
            d = np.linalg.norm(diff, axis=0)
            log_sum += np.log(d / d0)
            Y[:, N:] = ref + diff * (d0 / d)
            t_now = b * block
            newly = np.isnan(flip_lower) & (np.abs(ref[1]) > np.pi)
            flip_lower[newly] = t_now
            newly = np.isnan(flip_upper) & (np.abs(ref[0]) > np.pi)
            flip_upper[newly] = t_now
            if b in cp_blocks:
                lam_at[cp_blocks[b]] = log_sum / t_now
        E1 = energy_fn(Y[:, :N])
    lam = log_sum / (n_blocks * block)
    lam[~np.isfinite(lam)] = np.nan
    for v in lam_at.values():
        v[~np.isfinite(v)] = np.nan
    drift = np.abs(E1 - E0) / escale
    return dict(lam=lam, lam_at=lam_at, drift=drift,
                flip_lower=flip_lower, flip_upper=flip_upper)
