# World-model adaptation robustness study: code

This code regenerates every number, table and figure in the paper
"How Robust Is World-Model Adaptation? A Preregistered Simulation Study of Data Budget,
Data Quality and Physics Mismatch" (registration: https://osf.io/9kz2j).

Everything is simulated: there is no hardware and no external dataset.

## Requirements

Python 3.13. Install the pinned versions:

    pip install -r requirements.txt

## Quick start

    ./run_all.sh

This runs every step below in order and writes the results to `out/` and the figures to `fig/`.
On 2 CPU cores the full run took about 75 minutes. Runtime depends on the hardware.

## Execution order and outputs

| Step | Command | Writes | Paper section |
|---|---|---|---|
| 1 | `python experiments.py base` | `out/base.pkl` (DR base ensemble), `out/source_sample.npy` (20k source transitions for EDA) | Method |
| 2 | `python experiments.py DR` | `out/DR.csv` (DR, true simulator, and the exact and history-constrained latency-shift references) | Results A |
| 3 | `python experiments.py E1` | `out/E1.csv` budget sweep, 2 regimes x 4 mismatch types x 2 sizes x 7 budgets x 10 seeds | Results A |
| 4 | `python experiments.py E2` | `out/E2.csv` missingness: 3 mechanisms x 3 rates x 7 strategies | Results B |
| 5 | `python experiments.py E2b` | `out/E2b.csv` post-review: lag-aware imputer and the size-matched clean control | Results B |
| 6 | `python experiments.py E3` | `out/E3.csv` spikes x treatment (including treatment on clean data), and Gaussian noise | Results C |
| 7 | `python experiments.py E6` | `out/E6.csv` budget x mechanism with MICE | Results B |
| 8 | `python experiments.py E7` | `out/E7.csv` robustness by mismatch type | Results D |
| 9 | `python experiments.py E8` | `out/E8.csv` MPC task success, mass +60% and latency +4 | Results E |
| 10 | `python experiments.py base 1`, `python experiments.py base 2` | `out/base1.pkl`, `out/base2.pkl` (independently pretrained bases) | Results F |
| 11 | `TEST_SEED=54321 OUT_TAG=_b0 python experiments.py E9` (and the same with `BASE_FILE=base1.pkl` / `base2.pkl`, tags `_b1`, `_b2`) | `out/E9_b*.csv` key findings with each base and a second held-out test set | Results F |
| 12 | `python extras.py` | `out/extras.json`, `fig/eda.pdf`, `fig/shap.pdf` (EDA, missingness diagnostics, SHAP/ICE, PrefixSpan, fault detection) | EDA, Results G |
| 13 | `python analysis.py` | `out/stats.json`, `out/E1_tests.csv`, `out/E1_summary.csv`, `out/E2_ratios.csv`, `out/E3_ratios.csv`, `fig/scaling.pdf`, `fig/missing.pdf`, `fig/outliers.pdf`, `fig/cross.pdf`, `fig/mpc.pdf` | all |

Steps 3 to 11 need `out/base.pkl` from step 1. Step 13 needs every CSV.

## Files

- `sim.py`: two-link arm simulator, exploration policy, features and transitions.
- `models.py`: DR base ensemble, residual model, known-axis single-parameter system identification, true-simulator references, evaluation.
- `faults.py`: data collection under a strict budget, fault injection, spike treatments, imputers.
- `mpc.py`: random-shooting MPC reaching task.
- `experiments.py`: experiment grid and runner (joblib, `NJOBS` processes, default 2).
- `extras.py`: EDA, diagnostics and interpretation analyses.
- `analysis.py`: statistics and figures.

## Sources of randomness

- Base pretraining: one seed per base (`base` = 0; the multi-base check also uses bases 1 and 2).
- Held-out test set: 40 clean trajectories from `TEST_SEED` (main 12345; the multi-base check uses 54321).
- Per adaptation seed (0 to 9): the adaptation trajectories, the fault realization and the residual-network initialization.

The confidence intervals in the main results are over adaptation seeds, so they are conditional on base 0 and test set 12345.

## Warnings

Warnings are not silenced globally. Residual-network convergence is recorded in the
`res_converged` column. IterativeImputer convergence warnings are counted in the `imp_warnings` column.

## Changes after review

See `CHANGES.md`.
