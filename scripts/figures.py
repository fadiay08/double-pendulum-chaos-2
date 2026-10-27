"""Make the paper figures from results/*.npz. Usage: python3 figures.py [energy sweep euler map pairs rods films]
(PAPER=1 writes the versions without titles used in the paper to figures/paper/)"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import dp

RES = Path(__file__).parent / "results"
import os
PAPER = os.environ.get("PAPER") == "1"
FIG = Path(__file__).parent / ("figures/paper" if PAPER else "figures")
FIG.mkdir(parents=True, exist_ok=True)

# Colors (validated: categorical slots 1-2, blue ordinal ramp for step sizes)
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 9.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

THRESH = 0.1   # 1/s: lambda above this over a 400 s run counts as chaotic
STRONG = 0.3   # 1/s: "strong chaos" threshold for the short 50 s Euler comparison


def onset(ang, lam, thr):
    """First chaotic angle, and the angle above which every start is chaotic."""
    chaotic = lam > thr
    first = ang[np.argmax(chaotic)] if chaotic.any() else np.nan
    regular_idx = np.where(~chaotic)[0]
    always = ang[regular_idx[-1] + 1] if len(regular_idx) and regular_idx[-1] + 1 < len(ang) else np.nan
    return first, always


def K_value(th1_deg, th2_deg, p=dp.PARAMS):
    """(m1+m2)*L1*cos(th1) + m2*L2*cos(th2): compared against dp.flip_thresholds()."""
    return ((p["m1"] + p["m2"]) * p["L1"] * np.cos(np.deg2rad(th1_deg))
            + p["m2"] * p["L2"] * np.cos(np.deg2rad(th2_deg)))


def flip_angle(th2_of_th1):
    """Smallest th1 (deg) at which energy first allows the lower arm to flip."""
    a = np.linspace(0, 180, 180001)
    ok = K_value(a, th2_of_th1(a)) <= dp.flip_thresholds()[0] + 1e-12
    return a[np.argmax(ok)] if ok.any() else np.nan


def suptitle(fig, title, sub):
    if PAPER:
        return
    fig.text(0.0, 1.0, title, ha="left", va="bottom", fontsize=11.5, fontweight="bold",
             color=INK, transform=fig.transFigure)
    fig.text(0.0, 0.985, sub, ha="left", va="top", fontsize=8.8, color=INK2,
             transform=fig.transFigure)


def fig_energy():
    d = np.load(RES / "energy.npz")
    runs = [("euler_0.01", "Euler, Δt = 0.01 s", RAMP[3]),
            ("euler_0.001", "Euler, Δt = 0.001 s", RAMP[7]),
            ("euler_0.0001", "Euler, Δt = 0.0001 s", RAMP[11]),
            ("rk4_0.002", "RK4, Δt = 0.002 s", ORANGE)]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for key, label, color in runs:
        t = d["t_" + key][1:]
        e = np.maximum.accumulate(np.abs(d["dE_" + key][1:]))  # largest error so far
        ax.plot(t, e, color=color, label=label)
        ax.annotate(label.split(", ")[1] if key.startswith("euler") else "RK4",
                    (t[-1], e[-1]), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(1e-11, 30)
    ax.set_xlim(0, 60)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("largest energy error so far / E_ref")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.47), ncol=2)
    suptitle(fig, "Forward Euler keeps adding energy, RK4 does not",
             "Release from rest at θ1 = 90°, θ2 = 0. Error = |E(t) − E(0)|, "
             "E_ref = m1·g·L1 + m2·g·(L1 + L2) ≈ 29.4 J.")
    fig.subplots_adjust(top=0.86)
    fig.savefig(FIG / "fig1_energy_drift.png")
    plt.close(fig)


def fig_sweep():
    d = np.load(RES / "sweep_rk4.npz")
    ang = d["ang"]
    panels = [("lam_down", "Lower arm starts hanging straight down (θ2 = 0)",
               flip_angle(lambda a: 0 * a)),
              ("lam_line", "Both arms start in a straight line (θ2 = θ1)",
               flip_angle(lambda a: a))]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.6), sharex=True)
    summary = {}
    for ax, (key, title, flip_deg) in zip(axes, panels):
        lam = d[key]
        ax.plot(ang, lam, color=BLUE, lw=1.2)
        ax.set_yscale("log")
        ax.set_ylim(0.003, 5)
        ax.axhline(THRESH, color=MUTED, lw=0.9, ls=(0, (4, 3)))
        ax.text(2, THRESH * 1.12, "chaos threshold, λ = 0.1 /s", fontsize=8, color=MUTED, va="bottom")
        ax.axvline(flip_deg, color=INK2, lw=0.9)
        ax.text(flip_deg - 1.5, 2.6, f"lower arm can flip\nfrom {flip_deg:.1f}° up",
                fontsize=8, color=INK2, ha="right", va="top")
        first, always = onset(ang, lam, THRESH)
        summary[key] = (first, always, flip_deg)
        marks = ([(first, f"chaotic from {first:g}°", (0.60, 0.30))] if first == always else
                 [(first, f"first chaotic start\n{first:g}°", (0.22, 0.66)),
                  (always, f"chaotic for every start\nfrom {always:g}°", (0.60, 0.30))])
        for x, txt, pos in marks:
            yv = np.interp(x, ang, lam)
            ax.plot([x], [yv], "o", ms=6, color=INK, mec=SURF, mew=1.5, zorder=5)
            ax.annotate(txt, (x, yv), xytext=pos, textcoords="axes fraction", fontsize=8, color=INK,
                        va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=3, shrinkB=5))
        ax.set_title(("(a) " if key == "lam_down" else "(b) ") + title if PAPER else title, fontsize=9.5, pad=4)
        ax.set_ylabel("Lyapunov exponent λ (1/s)")
        ax.set_yticks([0.01, 0.1, 1])
        ax.set_yticklabels(["0.01", "0.1", "1"])
    axes[-1].set_xlabel("starting angle of the upper arm, θ1 (degrees)")
    axes[-1].set_xlim(0, 180)
    axes[-1].set_xticks(range(0, 181, 30))
    suptitle(fig, "Where chaos starts, measured with RK4",
             "Largest Lyapunov exponent over a 400 s run, every 0.25°. "
             "Equal masses and lengths, released from rest.")
    fig.subplots_adjust(top=0.89, hspace=0.28)
    fig.savefig(FIG / "fig2_lyapunov_vs_angle.png")
    plt.close(fig)
    return summary


def fig_euler():
    d = np.load(RES / "euler.npz")
    ang, dts, lam, ref, n = d["ang"], d["dts"], d["lam"], d["lam_rk4"], int(d["n"])
    cfgs = [(slice(0, n), "θ2 = 0 (lower arm hanging down)", BLUE),
            (slice(n, 2 * n), "θ2 = θ1 (arms in a line)", ORANGE)]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    out = {}
    for sl, label, color in cfgs:
        on = np.array([onset(ang, lam[i, sl], STRONG)[0] for i in range(len(dts))])
        rk = onset(ang, ref[sl], STRONG)[0]
        out[label] = (dts, on, rk)
        ax.axhline(rk, color=color, lw=0.9, ls=(0, (4, 3)), alpha=0.9)
        ax.plot(dts, on, "-o", color=color, ms=6, mec=SURF, mew=1.5, label=label)
        ax.annotate(f"RK4: {rk:g}°", (1.0, rk), xycoords=("axes fraction", "data"),
                    xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2, annotation_clip=False)
    ax.set_xscale("log")
    ax.set_xlim(8e-5, 1.25e-2)
    ax.set_xticks([1e-4, 1e-3, 1e-2])
    ax.set_xticklabels(["0.0001", "0.001", "0.01"])
    ax.set_ylim(-5, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Euler step size Δt (s)")
    ax.set_ylabel("apparent chaos onset (degrees)")
    ax.legend(loc="lower left")
    suptitle(fig, "With Euler, the chaos angle you find depends on your step size",
             "First starting angle with λ > 0.3 /s over a 50 s run. Dashed lines: same analysis with RK4.")
    fig.subplots_adjust(top=0.86)
    fig.savefig(FIG / "fig3_euler_step_size.png")
    plt.close(fig)
    return out


def fig_map():
    d = np.load(RES / "map.npz")
    th1, th2, lam = d["th1"], d["th2"], d["lam"]          # th1 >= 0 half
    # mirror (th1, th2) -> (-th1, -th2) to fill the whole square
    neg = lam[:0:-1, ::-1]
    full = np.concatenate([neg, lam], axis=0)             # rows: th1 from -180 to 180
    th1_full = np.concatenate([-th1[:0:-1], th1])
    cmap = LinearSegmentedColormap.from_list("blue_seq", RAMP)
    fig = plt.figure(figsize=(6.2, 6.1))
    side = 0.70                                    # square plot: same size in inches both ways
    ax = fig.add_axes([0.11, 0.20, side, side * 6.2 / 6.1])
    cax = fig.add_axes([0.11 + side + 0.03, 0.26, 0.028, side * 6.2 / 6.1 - 0.12])
    cax.grid(False)
    im = ax.imshow(full.T, origin="lower", cmap=cmap, vmin=0, vmax=1.5,
                   extent=[th1_full[0] - 1, th1_full[-1] + 1, th2[0] - 1, th2[-1] + 1],
                   interpolation="nearest", aspect="equal")
    ax.grid(False)
    lo, up = dp.flip_thresholds()
    T1, T2 = np.meshgrid(np.linspace(-180, 180, 721), np.linspace(-180, 180, 721), indexing="ij")
    K = K_value(T1, T2)
    ax.contour(T1, T2, K, levels=[lo], colors=[INK], linewidths=1.0)
    ax.contour(T1, T2, K, levels=[up], colors=[INK], linewidths=1.0, linestyles=[(0, (1.5, 2))])
    ax.set_xticks(range(-180, 181, 90))
    ax.set_yticks(range(-180, 181, 90))
    ax.set_xlabel("θ1, upper arm (degrees)")
    ax.set_ylabel("θ2, lower arm (degrees)")
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label("Lyapunov exponent λ (1/s)", color=INK2)
    cb.outline.set_edgecolor(AXIS)
    cb.ax.tick_params(color=AXIS, labelcolor=INK2)
    handles = [Line2D([], [], color=INK, lw=1.0, label="outside: energy lets the lower arm flip"),
               Line2D([], [], color=INK, lw=1.0, ls=(0, (1.5, 2)),
                      label="outside: energy lets the upper arm flip")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    suptitle(fig, "Chaos map for every pair of starting angles",
             "Released from rest. Dark = chaotic, light = regular. 200 s RK4 runs on a 2° grid.")
    fig.savefig(FIG / "fig4_chaos_map.png")
    plt.close(fig)




# ---------------------------------------------------------------------------
# The original 30 pairs and the real rod pendulum
# ---------------------------------------------------------------------------
import matplotlib.patheffects as pe

import rods

# What the original write-up says about each of the 30 simulated pairs
PAPER_SIM = {
    (30, 30): "stable", (30, 60): "stable", (30, 90): "stable", (30, 105): "stable", (30, 120): "chaotic",
    (45, 30): "stable", (45, 60): "stable", (45, 90): "stable", (45, 105): "not stated", (45, 120): "not stated",
    (60, 30): "stable", (60, 45): "chaotic", (60, 60): "stable", (60, 90): "stable", (60, 105): "chaotic",
    (60, 120): "chaotic", (75, 15): "stable", (75, 60): "stable", (75, 75): "stable", (75, 90): "chaotic",
    (90, 0): "chaotic", (90, 30): "chaotic", (90, 45): "chaotic", (90, 60): "chaotic", (90, 90): "chaotic",
    (120, 30): "chaotic", (120, 45): "chaotic", (120, 60): "chaotic", (120, 90): "chaotic", (120, 120): "chaotic",
}
REG, WEAK = 0.03, 0.10      # lambda (1/s) over a 400 s run: regular < 0.03 <= weakly chaotic <= 0.10 < chaotic


def verdict(lam, reg=REG, weak=WEAK):
    return "regular" if lam < reg else ("weakly chaotic" if lam <= weak else "chaotic")


HALO = [pe.withStroke(linewidth=2.6, foreground=SURF)]
MARK = {"chaotic": dict(marker="o", mfc=INK, mec=SURF, ms=7.5, mew=1.4),
        "stable": dict(marker="o", mfc=SURF, mec=INK, ms=7.5, mew=1.4),
        "borderline": dict(marker="D", mfc=MUTED, mec=SURF, ms=6.5, mew=1.2),
        "not stated": dict(marker="X", mfc=SURF, mec=INK, ms=7.5, mew=1.0)}


def _map_axes(fig, lam, th1, th2, extent, vmax, cbar_label):
    cmap = LinearSegmentedColormap.from_list("blue_seq", RAMP)
    side = 0.70
    ax = fig.add_axes([0.11, 0.20, side, side * fig.get_figwidth() / fig.get_figheight()])
    cax = fig.add_axes([0.11 + side + 0.03, 0.26, 0.028, side * fig.get_figwidth() / fig.get_figheight() - 0.12])
    cax.grid(False)
    h1, h2 = (th1[1] - th1[0]) / 2, (th2[1] - th2[0]) / 2
    im = ax.imshow(lam.T, origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                   extent=[th1[0] - h1, th1[-1] + h1, th2[0] - h2, th2[-1] + h2],
                   interpolation="nearest", aspect="auto")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.grid(False)
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label(cbar_label, color=INK2)
    cb.outline.set_edgecolor(AXIS)
    cb.ax.tick_params(color=AXIS, labelcolor=INK2)
    ax.set_xlabel("θ1, upper arm (degrees)")
    ax.set_ylabel("θ2, lower arm (degrees)")
    return ax


def fig_pairs_on_map():
    d = np.load(RES / "map.npz")
    r = np.load(RES / "reanalysis_30.npz")
    fig = plt.figure(figsize=(6.4, 6.6))
    ax = _map_axes(fig, d["lam"], d["th1"], d["th2"], (0, 136, -8, 136), 1.5, "Lyapunov exponent λ (1/s)")
    T1, T2 = np.meshgrid(np.linspace(0, 140, 561), np.linspace(-10, 140, 601), indexing="ij")
    ax.contour(T1, T2, K_value(T1, T2), levels=[dp.flip_thresholds()[0]], colors=[INK], linewidths=1.0)
    changed = []
    for a, b, lam in zip(r["th1"], r["th2"], r["lam"]):
        key = (int(a), int(b))
        said = PAPER_SIM[key]
        now = verdict(lam)
        ax.plot([a], [b], ls="none", zorder=5, **MARK[said])
        if said != "not stated" and ((said == "stable") != (now == "regular")):
            changed.append((key, said, now))
    offsets = {(60, 45): (14, -18), (120, 90): (-30, 22), (60, 90): (12, 16), (30, 105): (-14, 20),
               (30, 90): (-14, -24), (75, 15): (14, -18)}
    for (a, b), said, now in changed:
        dx, dy = offsets.get((a, b), (10, 10))
        ax.annotate(f"RK4: {now}" if PAPER else f"now {now}", (a, b), xytext=(dx, dy), textcoords="offset points", fontsize=7.8,
                    color=INK, path_effects=HALO, zorder=6, va="center",
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkA=1, shrinkB=5))
    handles = [Line2D([], [], ls="none", label="first analysis: chaotic" if PAPER else "write-up said chaotic", **MARK["chaotic"]),
               Line2D([], [], ls="none", label="first analysis: stable" if PAPER else "write-up said stable", **MARK["stable"]),
               Line2D([], [], ls="none", label="not classified" if PAPER else "not stated", **MARK["not stated"]),
               Line2D([], [], color=INK, lw=1.0, label="right of the line: lower arm can flip")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, columnspacing=1.2)
    suptitle(fig, "Your 30 simulated starts, checked with RK4",
             "Background: RK4 chaos map (dark = chaotic). Markers: the original Euler-based verdicts.")
    fig.savefig(FIG / "fig5_original_pairs_checked.png")
    plt.close(fig)
    return changed


FILM_MARK = {"none": dict(marker="o", mfc=SURF, mec=INK, ms=7.5, mew=1.4),
             "fold": dict(marker="o", mfc=INK, mec=INK, ms=7.5, mew=1.4),
             "mixed": dict(marker="o", fillstyle="left", mfc=INK, mfcalt=SURF, mec=INK, ms=7.5, mew=1.4)}
FILM_LABEL = {"none": "films: rods never fold", "fold": "films: rods fold",
              "mixed": "films: one of two runs folds"}


def fig_rods_map():
    d = np.load(RES / "rods_map.npz")
    e = np.load(RES / "experiment.npz")          # written by experiment.py
    p = rods.coeffs("rods", m1=0.24, m2=0.20, L1=0.24, L2=0.20)
    th1, th2, lam = d["th1"], d["th2"], d["lam"]
    full = np.concatenate([lam[:0:-1, ::-1], lam], axis=0)
    th1_full = np.concatenate([-th1[:0:-1], th1])
    fig = plt.figure(figsize=(6.2, 6.1))
    ax = _map_axes(fig, full, th1_full, th2, (-180, 180, -180, 180), 3.0, "Lyapunov exponent λ (1/s)")
    T1, T2 = np.meshgrid(np.linspace(-180, 180, 721), np.linspace(-180, 180, 721), indexing="ij")
    K = p["D"] * np.cos(np.deg2rad(T1)) + p["F"] * np.cos(np.deg2rad(T2))
    ax.contour(T1, T2, K, levels=[p["D"] - p["F"]], colors=[INK], linewidths=1.0)
    for (a, b), kind in zip(e["release"], e["tracked"]):
        ax.plot([a], [b], ls="none", zorder=5, **FILM_MARK[str(kind)])
    ax.set_xticks(range(-180, 181, 90)); ax.set_yticks(range(-180, 181, 90))
    handles = [Line2D([], [], ls="none", label=FILM_LABEL[k], **FILM_MARK[k]) for k in ("none", "fold", "mixed")]
    handles.append(Line2D([], [], color=INK, lw=1.0, label="outside: energy lets the lower arm flip"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, columnspacing=1.2)
    suptitle(fig, "Chaos map for your real pendulum (uniform rods, 24 cm and 20 cm)",
             "Background: frictionless rods, 100 s RK4 runs on a 3° grid. Markers: what the films show.")
    fig.savefig(FIG / "fig6_rods_map.png")
    plt.close(fig)


def fig_films():
    """Two filmed releases against the rod model with and without friction in the middle hinge."""
    import experiment as X
    R = X.runs()
    free, fric = X.model_runs(0.0), X.model_runs(X.TAU_MID)
    rels = [(75, 120), (90, 0)]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4), sharex=True)
    styles = [("run A (film)", dict(color=INK, lw=1.3)), ("run B (film)", dict(color=MUTED, lw=1.3)),
              ("model, no friction", dict(color=ORANGE, lw=1.2, ls=(0, (4, 2.5)))),
              (f"model, friction {100 * X.friction_ratio():.1f}% of F", dict(color=BLUE, lw=1.2))]
    letters = iter("abcd")
    for row, rel in zip(axes, rels):
        runs = [r for r in R if r["release"] == rel]
        curves = [(r["t"] - r["t_cross"], r["phi"], r["rho"]) for r in runs]
        curves += [(m["t"] - m["t_cross"], m["phi"], m["rho"]) for m in (free[rel], fric[rel])]
        for (tt, phi, rho), (label, st) in zip(curves, styles):
            row[0].plot(tt, phi, label=label, **st)
            row[1].plot(tt, rho, **st)
        name = f"({rel[0]}°, {rel[1]}°)"
        row[0].set_title(f"({next(letters)}) {name}: tip angle", fontsize=9.5, pad=4)
        row[1].set_title(f"({next(letters)}) {name}: distance ratio", fontsize=9.5, pad=4)
        row[0].set_ylabel("tip angle (degrees)")
        row[0].set_ylim(-125, 125); row[0].set_yticks(range(-90, 91, 45))
        row[1].set_ylabel("distance ratio")
        row[1].set_ylim(0, 1.08); row[1].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        row[1].axhline(0.5, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=0)
    axes[0, 1].text(-0.3, 0.47, "folded below 0.5", ha="left", va="top", fontsize=7.8, color=INK2,
                    path_effects=HALO, zorder=6)
    fig.supxlabel("time after the tip first passes under the pivot (s)", fontsize=9, color=INK2, y=0.01)
    for ax in axes.ravel():
        ax.set_xlim(-0.35, 4.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.005 if PAPER else 0.955),
               fontsize=8, handlelength=2.2, columnspacing=1.1)
    suptitle(fig, "Two filmed releases against the rod model",
             "Films and model aligned at the tip's first pass under the pivot.")
    fig.subplots_adjust(top=0.90 if PAPER else 0.84, bottom=0.09, hspace=0.32, wspace=0.28)
    fig.savefig(FIG / "fig7_films_vs_model.png")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["energy", "sweep", "euler", "map", "pairs", "rods", "films"]
    if "energy" in which:
        fig_energy()
    if "sweep" in which:
        print(fig_sweep())
    if "euler" in which:
        for k, (dts, on, rk) in fig_euler().items():
            print(k, dict(zip(dts.tolist(), on.tolist())), "RK4", rk)
    if "map" in which:
        fig_map()
    if "pairs" in which:
        print(fig_pairs_on_map())
    if "rods" in which:
        fig_rods_map()
    if "films" in which:
        fig_films()
