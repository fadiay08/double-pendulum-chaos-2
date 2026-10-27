"""The filmed releases of the physical pendulum (Tracker data from the original project).

Each of the nine starting positions was released twice and filmed (runs A and B). Run B was filmed
with a different camera and most of those videos are mirrored, which load() undoes the same way the
original plotting scripts did. The videos are 30 or 60 frames per second, and Tracker gives the
position (m) of the tip of the lower rod relative to the pivot for about 5 s.

Two mix-ups in the original data are corrected here (see SOURCES):
  * the B data stored under (90°, 0°) is an exact copy of the B data for (75°, 120°): dropped;
  * the B data stored under (90°, 90°) starts with the rods at a right angle, the shape of a
    (90°, 0°) release, and follows run A of (90°, 0°) to within 5 cm until the rods first fold:
    used as run B of (90°, 0°). (90°, 90°) therefore has a single run, and there are 17 runs.

The video calibration is off by 9-20% (the tip reaches 0.48-0.53 m from the pivot instead of
0.44 m), so the analysis only uses quantities that do not depend on it:
  * the tip angle: direction of the tip seen from the pivot, from the downward vertical;
  * the distance ratio: tip distance divided by its largest value in the run (99th percentile).
    Close to 1 when the rods are in line, below 0.5 when they fold over each other (more than
    about 120 degrees between the rods). Each drop below 0.5 counts as a fold.
Runs (and the model) are lined up at the moment the tip first passes under the pivot and compared
over the next 4.5 s.

The rod model is run without friction and with dry friction in the middle hinge: a torque of fixed
size tau = FRICTION * F (F = m2 g L2 / 2) opposing the rotation of the lower rod relative to the
upper one. FRICTION = 2.5% was chosen by comparing with the films (friction_scan()), not measured.

Data: data/tracker_data_original.csv, the Tracker numbers exactly as they appear in the original
experiment scripts, with the original labels.

Usage: python3 experiment.py   (prints Table 3 of the paper, writes results/experiment.npz,
                                 experiment_releases.csv and tracked_runs.csv)
"""
import csv
from pathlib import Path

import numpy as np

import rods

HERE = Path(__file__).parent
OUT = HERE / "results"
DATA = HERE / "data" / "tracker_data_original.csv"
if not DATA.exists():                        # scripts/ folder of the repo: data/ is one level up
    DATA = HERE.parent / "data" / "tracker_data_original.csv"

# How the original plotting scripts drew run B (camera B's image is mirrored): factors for (x, y)
MIRROR_B = {(30, 90): (-1, 1), (30, 120): (-1, 1), (60, 90): (1, -1), (60, 120): (1, 1), (75, 15): (-1, 1),
            (75, 75): (-1, 1), (75, 120): (-1, 1), (90, 0): (-1, 1), (90, 90): (-1, 1)}


def read_tracker_data(path=DATA):
    """The Tracker data exactly as stored in the original experiment scripts (labels as they were),
    as {"A_30_90": array of (t, x, y) rows, ...}."""
    raw = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            key = f"{r['run']}_{r['theta1_label_deg']}_{r['theta2_label_deg']}"
            raw.setdefault(key, []).append((float(r["t_s"]), float(r["x_m"]), float(r["y_m"])))
    return {k: np.array(v) for k, v in raw.items()}


RAW = read_tracker_data()

L1, L2 = 0.24, 0.20
REACH = L1 + L2
P_ROD = rods.coeffs("rods", m1=L1, m2=L2, L1=L1, L2=L2)
FRICTION = 0.025          # dry friction torque in the middle hinge, as a fraction of F = m2 g L2 / 2
TAU_MID = FRICTION * P_ROD["F"]
EPS = 0.02                # rad/s, smoothing of the sign of the relative rotation in the friction torque
DT = 0.001                # s, RK4 step for all model runs compared with the films
WINDOW = 4.5              # s after the first crossing used to compare runs
SEP_DEG = 45.0            # tip angles more than this apart = the runs have separated

RELEASES = [(30, 90), (30, 120), (60, 90), (60, 120), (75, 15), (75, 75), (75, 120), (90, 0), (90, 90)]
SEEN = {(30, 90): "not classified", (30, 120): "chaotic", (60, 90): "stable", (60, 120): "chaotic",
        (75, 15): "chaotic", (75, 75): "stable", (75, 120): "borderline", (90, 0): "chaotic",
        (90, 90): "chaotic"}
SOURCES = {
    (30, 90): {"A": "A_30_90", "B": "B_30_90"},
    (30, 120): {"A": "A_30_120", "B": "B_30_120"},
    (60, 90): {"A": "A_60_90", "B": "B_60_90"},
    (60, 120): {"A": "A_60_120", "B": "B_60_120"},
    (75, 15): {"A": "A_75_15", "B": "B_75_15"},
    (75, 75): {"A": "A_75_75", "B": "B_75_75"},
    (75, 120): {"A": "A_75_120", "B": "B_75_120"},
    (90, 0): {"A": "A_90_0", "B": "B_90_90"},       # see the module docstring
    (90, 90): {"A": "A_90_90"},
}


def friction_ratio():
    """TAU_MID as a fraction of F = m2*g*L2/2, the largest gravity torque on the lower rod
    about the middle hinge (only this ratio matters, the rod masses were not measured)."""
    return TAU_MID / P_ROD["F"]


def friction_scan(fractions=(0.0, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)):
    """How the choice of friction changes the comparison: starts that fold (of 9) per release and
    the tip-angle RMS error for the runs whose rods never fold. The films do not pin it down well."""
    R = runs()
    regular = [r for r in R if r["release"] in [(30, 90), (30, 120), (60, 90), (60, 120), (75, 75), (75, 120)]]
    film = {rel: np.mean([r["folds"] > 0 for r in R if r["release"] == rel]) for rel in RELEASES}
    out = []
    for f in fractions:
        tau = f * P_ROD["F"]
        m, near = model_runs(tau), nearby_folds(tau)
        err = np.array([tracking_error(m[r["release"]], r) for r in regular])
        score = sum(abs(near[rel] / len(NEARBY) - film[rel]) for rel in RELEASES)
        out.append((f, near, err[:, 0].mean(), err[:, 1].mean(), score))
    return out


# ------------------------------------------------------------------ measured runs
def load(key):
    """Time, x, y of one run: camera B's mirror image undone as in the original plots, and the
    release turned to the right (x > 0 at the start) so all runs share one convention."""
    D = RAW[key].astype(float).copy()
    if key.startswith("B"):
        a, b = key[2:].split("_")
        sx, sy = MIRROR_B[(int(a), int(b))]
        D[:, 1] *= sx
        D[:, 2] *= sy
    t, x, y = D.T
    if x[0] < 0:
        x = -x
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep] - t[0], x[keep], y[keep]


def tip_angle(x, y):
    """Direction of the tip seen from the pivot, from the downward vertical (degrees)."""
    return np.degrees(np.unwrap(np.arctan2(x, -y), axis=0))


def distance_ratio(x, y, ref=None):
    """Tip distance from the pivot divided by `ref` (default: the full reach of the rods)."""
    r = np.hypot(x, y)
    return r / (REACH if ref is None else ref)


def first_crossing(t, phi):
    """Time of the first downward zero crossing of the tip angle (tip passing under the pivot)."""
    i = np.where((phi[:-1] > 0) & (phi[1:] <= 0))[0][0]
    return t[i] + (t[i + 1] - t[i]) * phi[i] / (phi[i] - phi[i + 1])


def fold_events(t, rho, t_from, t_to):
    """Number of times the rods fold over each other (tip closer than half the reach to the pivot)."""
    f = rho[(t > t_from) & (t < t_to)] < 0.5
    return int(np.sum(f[1:] & ~f[:-1]) + (1 if len(f) and f[0] else 0))


def compare_runs(tA, phiA, tB, phiB, window=WINDOW, threshold=SEP_DEG):
    """Align two runs at their first crossings. Returns the common time axis, the tip-angle
    difference, its maximum, and the first time it exceeds `threshold` (NaN if never)."""
    t0A, t0B = first_crossing(tA, phiA), first_crossing(tB, phiB)
    tt = np.arange(0, window, 0.005)
    tt = tt[(tt + t0A <= tA[-1]) & (tt + t0B <= tB[-1])]
    diff = np.abs(np.interp(tt + t0A, tA, phiA) - np.interp(tt + t0B, tB, phiB))
    hit = np.where(diff > threshold)[0]
    return tt, diff, diff.max(), (tt[hit[0]] if len(hit) else np.nan)


def runs():
    out = []
    for rel in RELEASES:
        for name, key in SOURCES[rel].items():
            t, x, y = load(key)
            phi = tip_angle(x, y)
            ref = np.percentile(np.hypot(x, y), 99)
            rho = distance_ratio(x, y, ref)
            tc = first_crossing(t, phi)
            out.append(dict(release=rel, run=name, key=key, t=t, x=x, y=y, phi=phi, rho=rho, t_cross=tc,
                            scale=ref / REACH, folds=fold_events(t, rho, tc, tc + WINDOW)))
    return out


# ------------------------------------------------------------------ rod model, with optional friction
def simulate(th1_deg, th2_deg, T=6.0, dt=DT, p=P_ROD, tau_mid=0.0, eps=EPS):
    """Rod model released from rest (RK4), vectorized over arrays of starting angles.
    tau_mid: dry friction in the middle hinge, torque tau_mid*tanh((w2 - w1)/eps) opposing the
    relative rotation of the rods. Returns t, tip x, tip y with shape (steps + 1, N)."""
    A, B, C, D, F = p["A"], p["B"], p["C"], p["D"], p["F"]
    th1 = np.radians(np.atleast_1d(np.asarray(th1_deg, float)))
    th2 = np.radians(np.atleast_1d(np.asarray(th2_deg, float)))
    y = np.stack([th1, th2, np.zeros_like(th1), np.zeros_like(th1)])

    def deriv(y):
        a1, a2, w1, w2 = y
        d = a1 - a2
        cs, sn = np.cos(d), np.sin(d)
        fr = tau_mid * np.tanh((w2 - w1) / eps)
        r1 = -C * sn * w2 ** 2 - D * np.sin(a1) + fr
        r2 = C * sn * w1 ** 2 - F * np.sin(a2) - fr
        det = A * B - (C * cs) ** 2
        return np.stack([w1, w2, (B * r1 - C * cs * r2) / det, (A * r2 - C * cs * r1) / det])

    n = int(round(T / dt))
    ang = np.empty((n + 1, 2, th1.size))
    ang[0] = y[:2]
    for i in range(n):
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ang[i + 1] = y[:2]
    t = np.arange(n + 1) * dt
    a1, a2 = ang[:, 0], ang[:, 1]
    return t, L1 * np.sin(a1) + L2 * np.sin(a2), -L1 * np.cos(a1) - L2 * np.cos(a2)


def model_runs(tau_mid, dt=DT, T=6.0):
    """The nine releases in the model: tip angle, distance ratio, first crossing, folds."""
    th1 = [a for a, _ in RELEASES]
    th2 = [b for _, b in RELEASES]
    t, x, y = simulate(th1, th2, T=T, dt=dt, tau_mid=tau_mid)
    out = {}
    for j, rel in enumerate(RELEASES):
        phi = tip_angle(x[:, j], y[:, j])
        rho = distance_ratio(x[:, j], y[:, j])
        tc = first_crossing(t, phi)
        out[rel] = dict(t=t, phi=phi, rho=rho, t_cross=tc, folds=fold_events(t, rho, tc, tc + WINDOW))
    return out


NEARBY = [(a, b) for a in (-3, 0, 3) for b in (-3, 0, 3)]   # the nominal start and 8 neighbours


def nearby_folds(tau_mid, dt=DT):
    """For each release: in how many of the 9 starts within 3 degrees (NEARBY) the model's rods
    fold within the comparison window. The real starting angles are only known to a few degrees."""
    th1 = np.array([a + p for a, _ in RELEASES for p, _ in NEARBY], float)
    th2 = np.array([b + q for _, b in RELEASES for _, q in NEARBY], float)
    t, x, y = simulate(th1, th2, T=6.0, dt=dt, tau_mid=tau_mid)
    folded = np.zeros(len(th1), bool)
    for j in range(len(th1)):
        phi, rho = tip_angle(x[:, j], y[:, j]), distance_ratio(x[:, j], y[:, j])
        tc = first_crossing(t, phi)
        folded[j] = fold_events(t, rho, tc, tc + WINDOW) > 0
    return dict(zip(RELEASES, folded.reshape(len(RELEASES), len(NEARBY)).sum(axis=1).tolist()))


def twin_gap(tau_mid, delta=1.0, dt=DT):
    """Largest tip-angle difference (deg) within the comparison window between two model releases
    `delta` degrees apart in both angles, aligned like the filmed runs: what the films could show."""
    th1 = np.array([a for a, _ in RELEASES], float)
    th2 = np.array([b for _, b in RELEASES], float)
    t, x, y = simulate(np.r_[th1, th1 + delta], np.r_[th2, th2 + delta], T=6.0, dt=dt, tau_mid=tau_mid)
    n = len(RELEASES)
    out = {}
    for j, rel in enumerate(RELEASES):
        _, _, mx, ts = compare_runs(t, tip_angle(x[:, j], y[:, j]), t, tip_angle(x[:, j + n], y[:, j + n]))
        out[rel] = (float(mx), float(ts))
    return out


def spread(t, rho, t_cross, t_from=1.0):
    """Standard deviation of the distance ratio from t_from s after the first crossing to the end of
    the window: how much the angle between the rods keeps changing."""
    m = (t > t_cross + t_from) & (t < t_cross + WINDOW)
    return float(rho[m].std())


def tracking_error(m, r, T=4.0, early=1.5):
    """RMS difference (deg) between model and measured tip angle, both aligned at their first
    crossing, over T s and over the first `early` s."""
    tt = np.arange(0, T, 0.01)
    tt = tt[tt + r["t_cross"] <= r["t"][-1]]
    dif = np.interp(tt + m["t_cross"], m["t"], m["phi"]) - np.interp(tt + r["t_cross"], r["t"], r["phi"])
    return np.sqrt(np.mean(dif ** 2)), np.sqrt(np.mean(dif[tt < early] ** 2))


LABELS = {"none": "rods never fold", "fold": "rods fold", "mixed": "one run folds"}


def analysis():
    R = runs()
    free, fric = model_runs(0.0), model_runs(TAU_MID)
    near_free, near_fric = nearby_folds(0.0), nearby_folds(TAU_MID)
    twin_free, twin_fric = twin_gap(0.0), twin_gap(TAU_MID)
    if (OUT / "rods_pairs.npz").exists():          # written by rods_analysis.py pairs
        pz = np.load(OUT / "rods_pairs.npz")
        lam = {(int(a), int(b)): float(v) for a, b, v in zip(pz["th1"], pz["th2"], pz["lam_rod"])}
    else:
        print("results/rods_pairs.npz not found (run: python3 rods_analysis.py pairs); lambda left empty")
        lam = {rel: np.nan for rel in RELEASES}
    regular, rows = [], []
    for rel in RELEASES:
        rr = [r for r in R if r["release"] == rel]
        row = dict(release=rel, seen=SEEN[rel], lam_rod=lam[rel], folds={r["run"]: r["folds"] for r in rr},
                   spread={r["run"]: spread(r["t"], r["rho"], r["t_cross"]) for r in rr},
                   folds_free=free[rel]["folds"], folds_fric=fric[rel]["folds"],
                   near_free=near_free[rel], near_fric=near_fric[rel],
                   spread_free=spread(free[rel]["t"], free[rel]["rho"], free[rel]["t_cross"]),
                   spread_fric=spread(fric[rel]["t"], fric[rel]["rho"], fric[rel]["t_cross"]),
                   twin_free=twin_free[rel][0], twin_fric=twin_fric[rel][0])
        if len(rr) == 2:
            _, _, mx, tsep = compare_runs(rr[0]["t"], rr[0]["phi"], rr[1]["t"], rr[1]["phi"])
            gap = np.hypot(rr[0]["x"][0] / rr[0]["scale"] - rr[1]["x"][0] / rr[1]["scale"],
                           rr[0]["y"][0] / rr[0]["scale"] - rr[1]["y"][0] / rr[1]["scale"])
            row.update(max_diff=float(mx), t_sep=float(tsep), start_gap_cm=float(100 * gap))
        else:
            row.update(max_diff=np.nan, t_sep=np.nan, start_gap_cm=np.nan)
        n_fold = sum(r["folds"] > 0 for r in rr)
        row["tracked"] = "none" if n_fold == 0 else ("fold" if n_fold == len(rr) else "mixed")
        if row["tracked"] == "none":
            regular.extend(rr)
        rows.append(row)
    err_free = np.array([tracking_error(free[r["release"]], r) for r in regular])
    err_fric = np.array([tracking_error(fric[r["release"]], r) for r in regular])
    return R, rows, free, fric, err_free, err_fric


def fmt_sep(w):
    if "B" not in w["folds"]:
        return "-"
    return "never" if np.isnan(w["t_sep"]) else f"{w['t_sep']:.1f}"


def main():
    R, rows, free, fric, err_free, err_fric = analysis()
    print(f"dry friction in the middle hinge: tau = {TAU_MID} (= {100 * friction_ratio():.1f}% of m2 g L2/2)")
    print(f"comparison window: {WINDOW} s after the tip first passes under the pivot\n")
    print("TABLE 3")
    print(f"{'release':10s} {'first analysis':15s} {'largest gap':>11s} {'45 deg apart':>12s} {'folds A/B':>9s} "
          f"{'model starts that fold (of 9)':>30s} {'lambda':>7s}   films")
    for w in rows:
        f = w["folds"]
        folds = f"{f['A']}/{f['B']}" if "B" in f else f"{f['A']}/-"
        diff = "-" if np.isnan(w["max_diff"]) else f"{w['max_diff']:.0f}"
        print(f"{str(w['release']):10s} {w['seen']:15s} {diff:>11s} {fmt_sep(w):>12s} {folds:>9s} "
              f"{w['near_free']:>17d} / {w['near_fric']:<10d} {w['lam_rod']:7.2f}   {LABELS[w['tracked']]}")
    print("\nsupporting numbers")
    print(f"{'release':10s} {'start gap':>9s} {'spread films':>14s} {'spread model':>16s} {'folds model (nominal)':>22s} "
          f"{'twins 1 deg apart: largest gap':>31s}")
    for w in rows:
        gap = "-" if np.isnan(w["start_gap_cm"]) else f"{w['start_gap_cm']:.1f} cm"
        sp = " ".join(f"{v:.3f}" for v in w["spread"].values())
        print(f"{str(w['release']):10s} {gap:>9s} {sp:>14s} {w['spread_free']:7.3f} / {w['spread_fric']:.3f} "
              f"{w['folds_free']:>12d} / {w['folds_fric']:<8d} {w['twin_free']:>14.0f} / {w['twin_fric']:.0f} deg")
    print(f"\ntip-angle RMS error, {len(err_free)} runs of the releases whose rods never fold:")
    print(f"  no friction:   {err_free[:, 0].mean():.1f} deg over 4 s, {err_free[:, 1].mean():.1f} deg over the first 1.5 s")
    print(f"  with friction: {err_fric[:, 0].mean():.1f} deg over 4 s, {err_fric[:, 1].mean():.1f} deg over the first 1.5 s")
    print("\nper run: video scale (tracked reach / 44 cm), folds, first pass under the pivot (s)")
    for r in R:
        print(f"  {str(r['release']):10s} {r['run']}  scale {r['scale']:.3f}  folds {r['folds']}  first pass {r['t_cross']:.2f}")

    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "experiment.npz", release=np.array([w["release"] for w in rows]),
             tracked=np.array([w["tracked"] for w in rows]), max_diff=np.array([w["max_diff"] for w in rows]),
             t_sep=np.array([w["t_sep"] for w in rows]), err_free=err_free, err_fric=err_fric, tau=TAU_MID)
    with open(HERE / "experiment_releases.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["theta1_deg", "theta2_deg", "first_analysis_by_eye", "films", "start_gap_tip_cm",
                     "largest_tip_angle_gap_between_runs_deg", "runs_45deg_apart_after_s", "folds_run_A",
                     "folds_run_B", "model_starts_that_fold_of_9_no_friction", "model_starts_that_fold_of_9_friction",
                     "lambda_frictionless_rods_per_s", "model_twins_1deg_largest_gap_no_friction_deg",
                     "model_twins_1deg_largest_gap_friction_deg"])
        for w in rows:
            f = w["folds"]
            wr.writerow([w["release"][0], w["release"][1], w["seen"], LABELS[w["tracked"]],
                         "" if np.isnan(w["start_gap_cm"]) else f"{w['start_gap_cm']:.1f}",
                         "" if np.isnan(w["max_diff"]) else f"{w['max_diff']:.1f}",
                         "" if fmt_sep(w) == "-" else fmt_sep(w), f["A"], f.get("B", ""),
                         w["near_free"], w["near_fric"], f"{w['lam_rod']:.2f}",
                         f"{w['twin_free']:.1f}", f"{w['twin_fric']:.1f}"])
    with open(HERE / "tracked_runs.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["theta1_deg", "theta2_deg", "run", "t_s", "x_m", "y_m"])
        for r in R:
            for ti, xi, yi in zip(r["t"], r["x"], r["y"]):
                wr.writerow([r["release"][0], r["release"][1], r["run"], f"{ti:.3f}", f"{xi:.5f}", f"{yi:.5f}"])


if __name__ == "__main__":
    main()
