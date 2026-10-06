# Which Confidence Interval for Which Metric?

Companion code, results and figures for

> Ameera Jaradat, Ahmad Nasayreh, Ayah Bashkami, Ahmad Ibrahim Alshdaifat and Hasan Gharaibeh,
> **"Which Confidence Interval for Which Metric? A Coverage Study for Medical AI Classification,"** CAISAIS 2026.

The study measures the empirical coverage of nine confidence-interval constructions (Wald, Wilson, Agresti–Coull, Clopper–Pearson, Jeffreys, percentile bootstrap, BCa bootstrap, bootstrap-SE normal approximation and DeLong) for accuracy, F1, MCC, balanced accuracy, AUROC and paired model differences, and applies them to two clinical test sets (Breast Cancer Wisconsin, PneumoniaMNIST).

## Repository layout

```
bootci.py                          vectorised bootstrap and closed-form interval library (used by everything below)
notebooks/
  bootstrap_ci_medical_ai.ipynb    main experiments, executed; outputs are those reported in the paper
scripts/
  sensitivity_check.py             sensitivity settings S3 (prevalence 0.10) and S4 (bimodal positives)
  make_fig_coverage.py             draws Fig. 1 of the paper from results/table9_coverage_complex.csv
tests/
  test_bootci.py                   verification of every metric kernel and interval against SciPy / scikit-learn
results/                           all tables as CSV, plus results_all.json (configuration, environment, every table)
figures/
  fig_coverage.pdf|png             Fig. 1 of the paper
  notebook_run/                    figures written by the notebook run
```

## Where each result in the paper comes from

| Paper | Produced by | Result file |
|---|---|---|
| Table IV (simulation settings, population values) | notebook Section 16 (moderate, high); `scripts/sensitivity_check.py` (S3, S4) | `table9_coverage_complex.csv`, `table12_sensitivity.csv` |
| Table V (coverage for accuracy) | notebook Section 12 | `table7_coverage.csv` |
| Table VI (coverage for F1, MCC, balanced accuracy, AUROC) | notebook Section 16, Part A | `table9_coverage_complex.csv` |
| Fig. 1 | `scripts/make_fig_coverage.py` | `figures/fig_coverage.pdf` |
| Table VII (paired differences) | notebook Section 16, Part B | `table10_paired_coverage.csv` |
| Table VIII (CNN vs. pixel logistic regression) | notebook Sections 10 and 16, Part C | `table5_imaging_metrics.csv`, `table11_paired_complex.csv` |
| Sensitivity analysis (Sec. V-B) | `scripts/sensitivity_check.py` | `table12_sensitivity.csv` |
| Boundary collapse and ceiling effect (Sec. V-D) | notebook Sections 3, 6 and 10 | `table1_tabular_metrics.csv`, `table4_sweep_tabular.csv`, `table5_imaging_metrics.csv` |
| Computational cost (Sec. IV) | notebook Section 13 | `table8_benchmark.csv` |
| Holm-adjusted p-values (Sec. VI) | notebook Section 16, Part C (printed output) | — |

The notebook numbers its own tables 1–12; these are not the table numbers of the paper.

## Reproducing the results

**Main experiments.** Open `notebooks/bootstrap_ci_medical_ai.ipynb` in Google Colab and choose `Runtime → Run all`. A CPU runtime is sufficient; the full run takes about 20–25 minutes. Set `CONFIG["FAST_MODE"] = True` in Section 0 for a short smoke test.

**Scripts and tests** (from the repository root, Python 3.12 recommended):

```bash
pip install -r requirements.txt
python tests/test_bootci.py            # all checks must pass
python scripts/sensitivity_check.py    # about 6 min on one CPU core
python scripts/make_fig_coverage.py
```

### Software environment

| Run | Python | NumPy | SciPy | scikit-learn | pandas | PyTorch | Hardware |
|---|---|---|---|---|---|---|---|
| Notebook (main experiments) | 3.12.13 | 2.0.2 | 1.16.3 | 1.6.1 | 2.2.2 | 2.11.0 (CPU build) | Google Colab CPU runtime, no GPU |
| S3 and S4 sensitivity settings | 3.13.16 | 2.5.3 | 1.18.1 | 1.9.1 | 3.0.5 | — | Linux, one CPU core |

### Random seeds and determinism

| Component | Seed |
|---|---|
| Train/test split, models, real-data bootstrap | `CONFIG["SEED"] = 42` |
| Accuracy coverage study (Section 12) | `int(1000 * p) + n` for each cell |
| Coverage study for other metrics (Section 16, Part A) | `20240 + n + int(1000 * mu)` |
| Paired-difference study (Section 16, Part B) | `20240 + 7 * n + int(100 * mu_a)` |
| S3 and S4 (`sensitivity_check.py`) | `20240 + 7919 * (setting index + 3) + n` |

All random numbers come from `numpy.random.Generator` (PCG64) with explicit seeds, so the tabular experiment and the simulations reproduce exactly under the versions above. CNN training on a CPU can differ in the last digits across hardware and PyTorch versions; the numbers in the paper are the saved outputs of the notebook.

## Data

No data are redistributed here. Breast Cancer Wisconsin (Diagnostic) is loaded with `sklearn.datasets.load_breast_cancer`, and PneumoniaMNIST is downloaded by the `medmnist` package on first use. The licences of the original sources apply.

## Note on the saved notebook

After execution, narrative text in the notebook (markdown cells, code comments, a docstring and two printed status messages) was edited for clarity. Code logic and every numerical output are exactly as executed. Colab-internal `DeprecationWarning` messages and file-download widgets were removed from the saved outputs.

## Licence

Code is released under the MIT licence (see `LICENSE`). If you use it, please cite the paper above (see `CITATION.cff`).
