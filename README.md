# Double pendulum chaos: RK4, Lyapunov exponent and the filmed releases

Code and data for the paper *Release Angles and the Onset of Chaos in a Double Pendulum* (Fadi Ayoub). It redoes my Ma'alah project with Nahed Mansour (the original version is in the repository `double-pendulum-chaos`).

## What's here

- `double_pendulum_chaos.ipynb`: the whole analysis in one notebook, already run, so it opens with every output. It runs in Colab as is (Runtime → Run all, about 35 minutes; set `FAST = True` for a rough version in about 6 minutes). In Colab it downloads `data/tracker_data_original.csv` from this repository. That link assumes the default branch is called `main`; if it isn't, change `URL` in section 9 or upload the file in Colab's Files panel.
- `data/tracker_data_original.csv`: the Tracker data for the 18 filmed runs (position of the lower rod's tip relative to the pivot), exactly as in the original experiment scripts and with the original labels. Columns: starting angles as labelled, run (A or B), time (s), x and y (m).
- `figures/`
  - `fig1_energy_drift.png`: forward Euler keeps adding energy, RK4 doesn't.
  - `fig2_lyapunov_vs_angle.png`: Lyapunov exponent vs the upper arm's starting angle, for θ2 = 0 and θ2 = θ1.
  - `fig3_euler_step_size.png`: the chaos angle Euler finds depends on its step size.
  - `fig4_chaos_map.png`: chaos map over every pair of starting angles (1 kg masses, 1 m arms).
  - `fig5_original_pairs_checked.png`: the 30 original pairs on the RK4 chaos map, with the first analysis's verdicts.
  - `fig6_rods_map.png`: chaos map for the real pendulum (frictionless uniform rods, 24 cm and 20 cm), with what the films show for the 9 filmed releases.
  - `fig7_films_vs_model.png`: two filmed releases next to the rod model with and without friction in the middle hinge.
- `original_pairs_redone.csv`: the 30 original pairs: first-analysis verdict, original Euler test, the same test with RK4, Lyapunov exponent, new verdict.
- `experiment_releases.csv`: Table 3 of the paper, the 9 filmed releases: what the films show and what the rod model predicts.
- `lyapunov_sweep.csv`: λ every 0.25° for θ2 = 0 and θ2 = θ1.
- `scripts/`: the plain Python version used for the paper. `runs.py`, `reanalysis_30.py` and `rods_analysis.py` do the simulations, `experiment.py` analyzes the films (it prints Table 3), and `figures.py` draws the figures. `dp.py` is the point-mass model and `rods.py` the general one that also covers rods. Run `rods_analysis.py pairs` before `experiment.py` to get the λ column.

## Labels used

- Lyapunov exponent λ from 400 s RK4 runs (1 m model): regular below 0.03 /s, weakly chaotic 0.03 to 0.1 /s, chaotic above 0.1 /s. For the faster rod pendulum: regular below 0.1 /s, chaotic above 0.3 /s.
- Films: the tip angle (direction of the tip seen from the pivot) and the distance ratio (tip distance divided by its largest value in the run) don't depend on the video scale. A fold is a drop of the distance ratio below 0.5. Runs are lined up at the tip's first pass under the pivot and compared over 4.5 s.
- Friction model: dry friction in the middle hinge, 2.5% of F = m2·g·L2/2. This value was chosen by comparing with the films, not measured; anything from 1% to 5% fits the films better than no friction.

## Checks done

- Equations of motion match an independent computer-algebra derivation (point masses and rods) to about 1e-14.
- RK4 energy error stays under 0.01% of the energy scale in every frictionless run. The two chaos maps are the worst at about 7e-5; the sweeps and the 30 pairs stay under 4e-6.
- Halving the RK4 step gives the same chaotic/regular label for every 1° start.
- A second method (SciPy's DOP853 solver with the tangent equations) gives the same verdict for every start tested, including (60°, 90°), (120°, 90°), (30°, 90°), (75°, 15°), (60°, 45°) and (45°, 105°).
- The film comparison gives the same numbers with RK4 steps of 0.001 s and 0.0005 s.

## Fixes to the original files (all handled in the code)

- Two experiment data sets were mixed up. The B data stored under (90°, 0°) is an exact copy of the B data for (75°, 120°), so it is dropped. The B data stored under (90°, 90°) starts with the rods at a right angle and follows run A of (90°, 0°), so it is used as run B of (90°, 0°). That leaves 17 runs, and (90°, 90°) has only one.
- Most run B videos are mirrored (a different camera). The code flips them back the same way the original plotting scripts did.
- The video scale is 9% to 20% too large (the tracked tip reaches 48 to 53 cm from the pivot, but the rods are 44 cm long together), so only measures that don't depend on the scale are used.
- The (30°, 90°) experiment script plots `x_a` and `y_a` without defining them (the line `t_a, x_a, y_a = data_a[:, 0], data_a[:, 1], data_a[:, 2]` is missing), so it stops with a NameError before drawing anything.
- The write-up says the twin releases differ by 0.01 rad in one place, but the code uses 0.1° (0.0017 rad).
