"""Verification suite for bootci.py.  Run from the repository root:

    python tests/test_bootci.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
import numpy as np
from scipy import stats, optimize
from sklearn.metrics import (roc_auc_score, f1_score, matthews_corrcoef,
                             balanced_accuracy_score, recall_score,
                             precision_score, accuracy_score)
import bootci as bc

rng = np.random.default_rng(0)
FAIL = []

def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (" | " + extra if extra else ""))
    if not cond:
        FAIL.append(name)

# ---------------------------------------------------------------- 1. midranks
a = rng.integers(0, 5, size=(7, 11)).astype(float)      # heavy ties
ref = np.stack([stats.rankdata(row, method="average") for row in a])
check("midranks vs scipy.rankdata (ties)", np.allclose(bc._midranks_2d(a), ref))
b = rng.normal(size=(5, 9))
ref2 = np.stack([stats.rankdata(r, method="average") for r in b])
check("midranks vs scipy.rankdata (no ties)", np.allclose(bc._midranks_2d(b), ref2))

# ---------------------------------------------------------------- 2. metrics
for trial in range(5):
    n = rng.integers(30, 120)
    yt = rng.integers(0, 2, n)
    ys = np.round(rng.random(n), 2)            # rounding -> genuine ties
    yp = (ys > 0.5).astype(int)
    if yt.sum() in (0, n):
        continue
    YT, YP, YS = yt[None, :], yp[None, :], ys[None, :]
    ok = (
        np.isclose(bc._accuracy(YT, YP, YS)[0], accuracy_score(yt, yp)) and
        np.isclose(bc._auroc(YT, YP, YS)[0], roc_auc_score(yt, ys)) and
        np.isclose(bc._f1(YT, YP, YS)[0], f1_score(yt, yp, zero_division=0)) and
        np.isclose(bc._mcc(YT, YP, YS)[0], matthews_corrcoef(yt, yp), atol=1e-12) and
        np.isclose(bc._balanced_accuracy(YT, YP, YS)[0], balanced_accuracy_score(yt, yp)) and
        np.isclose(bc._sensitivity(YT, YP, YS)[0], recall_score(yt, yp, zero_division=0)) and
        np.isclose(bc._ppv(YT, YP, YS)[0], precision_score(yt, yp, zero_division=0))
    )
    check(f"metric kernels vs sklearn (trial {trial}, n={n})", ok)

# AUROC on an actual bootstrap resample (duplicated rows => massive ties)
n = 60
yt = rng.integers(0, 2, n); ys = rng.random(n); yp = (ys > .5).astype(int)
idx = rng.integers(0, n, size=(20, n))
mine = bc._auroc(yt[idx], yp[idx], ys[idx])
ref3 = np.array([roc_auc_score(yt[i], ys[i]) if 0 < yt[i].sum() < n else np.nan for i in idx])
m = np.isfinite(ref3)
check("bootstrap AUROC vs sklearn on resamples", np.allclose(mine[m], ref3[m]))

# ---------------------------------------------------------------- 3. binomial CIs
def wilson_numeric(k, n, alpha=0.05):
    ph, z = k / n, stats.norm.ppf(1 - alpha / 2)
    f = lambda p: (ph - p) ** 2 - z ** 2 * p * (1 - p) / n
    e = 1e-12
    lo = optimize.brentq(f, e, max(ph - e, e)) if k > 0 else 0.0
    hi = optimize.brentq(f, min(ph + e, 1 - e), 1 - e) if k < n else 1.0
    return lo, hi

for (k, n) in [(45, 50), (2, 20), (95, 100), (500, 624), (0, 30), (30, 30)]:
    w = bc.binomial_ci(k, n, method="wilson")
    wn = wilson_numeric(k, n)
    check(f"Wilson closed-form vs score-test root ({k}/{n})",
          np.allclose(w, wn, atol=1e-8), f"{w} vs {wn}")
    cp = bc.binomial_ci(k, n, method="clopper_pearson")
    # Clopper-Pearson must be at least as wide as Wilson and contain p_hat
    check(f"Clopper-Pearson sane ({k}/{n})",
          cp[0] <= k / n <= cp[1] and cp[0] <= w[0] + 1e-9 and cp[1] + 1e-9 >= w[1], str(cp))

lo, hi = bc.binomial_ci(50, 50, method="wald")
check("Wald degenerates at p_hat = 1 (zero width)", np.isclose(lo, 1.0) and np.isclose(hi, 1.0))
lo, hi = bc.binomial_ci(29, 30, method="wald")
check("Wald overshoots [0,1] at p_hat -> 1", hi > 1.0, f"upper={hi:.4f}")

# ---------------------------------------------------------------- 4. DeLong
n = 400
yt = rng.integers(0, 2, n)
ys = rng.normal(yt * 1.1, 1.0)
d = bc.delong_auc_ci(yt, ys)
check("DeLong AUC == sklearn AUC", np.isclose(d["auc"], roc_auc_score(yt, ys)))
bs = bc.bootstrap_ci(yt, (ys > 0).astype(int), ys, metric="auroc", B=4000,
                     method="percentile", seed=1)
check("DeLong SE ~ bootstrap SE (within 10%)",
      abs(d["se"] - bs.se_boot) / bs.se_boot < 0.10,
      f"delong={d['se']:.5f} boot={bs.se_boot:.5f}")

# ---------------------------------------------------------------- 5. bootstrap_ci
r1 = bc.bootstrap_ci(yt, (ys > 0).astype(int), ys, metric="accuracy", B=3000, seed=7)
r2 = bc.bootstrap_ci(yt, (ys > 0).astype(int), ys, metric="accuracy", B=3000, seed=7)
check("bootstrap is reproducible under a fixed seed",
      (r1.ci_low, r1.ci_high) == (r2.ci_low, r2.ci_high))
check("point estimate matches sklearn accuracy",
      np.isclose(r1.estimate, accuracy_score(yt, (ys > 0).astype(int))))
check("BCa z0 and acceleration are finite",
      np.isfinite(r1.z0) and np.isfinite(r1.accel), f"z0={r1.z0:.4f} a={r1.accel:.5f}")
w = bc.binomial_ci(int((yt == (ys > 0).astype(int)).sum()), n, method="wilson")
check("BCa accuracy CI close to Wilson at n=400",
      abs(r1.ci_low - w[0]) < 0.02 and abs(r1.ci_high - w[1]) < 0.02,
      f"bca={r1.ci_low:.4f},{r1.ci_high:.4f} wilson={w[0]:.4f},{w[1]:.4f}")
check("bootstrap SE ~ analytic binomial SE",
      abs(r1.se_boot - np.sqrt(r1.estimate * (1 - r1.estimate) / n)) < 0.004,
      f"{r1.se_boot:.5f}")

# stratified resampling must preserve class counts
strata = yt
gen = bc._iter_index_chunks(n, 50, np.random.default_rng(0), strata=strata, chunk_size=50)
idx = next(gen)
check("stratified bootstrap preserves prevalence",
      bool(np.all(yt[idx].sum(1) == yt.sum())))

# degenerate-replicate guard
tiny_yt = np.array([1, 0, 0, 0, 0, 0, 0, 0])
tiny_ys = np.array([.9, .1, .2, .3, .4, .5, .6, .7])
res = bc.bootstrap_ci(tiny_yt, (tiny_ys > .5).astype(int), tiny_ys,
                      metric="auroc", B=500, method="percentile", seed=3)
check("degenerate AUROC replicates are counted, not crashed",
      res.n_valid < 500 and np.isfinite(res.ci_low), f"n_valid={res.n_valid}")

# ---------------------------------------------------------------- 6. paired test
pa = (ys > 0).astype(int)
pb = pa.copy()
flip = rng.choice(n, 40, replace=False)
pb[flip] = 1 - pb[flip]
pd_ = bc.paired_bootstrap_diff(yt, pa, pb, metric="accuracy", B=6000, seed=11)
mc = bc.mcnemar_exact(yt, pa, pb)
check("paired delta matches direct computation",
      np.isclose(pd_["delta"], accuracy_score(yt, pa) - accuracy_score(yt, pb)))
check("paired bootstrap ASL agrees with exact McNemar (same decision)",
      (pd_["asl_p"] < .05) == (mc["p_value"] < .05),
      f"asl={pd_['asl_p']:.4g} mcnemar={mc['p_value']:.4g}")
# paired CI must be narrower than the naive difference of independent SEs
ra = bc.bootstrap_ci(yt, pa, metric="accuracy", B=6000, method="percentile", seed=11)
rb = bc.bootstrap_ci(yt, pb, metric="accuracy", B=6000, method="percentile", seed=12)
naive = np.sqrt(ra.se_boot**2 + rb.se_boot**2) * 2 * 1.96
check("paired CI narrower than unpaired (correlation exploited)",
      (pd_["ci_high"] - pd_["ci_low"]) < naive,
      f"paired={pd_['ci_high']-pd_['ci_low']:.4f} unpaired={naive:.4f}")

# ---------------------------------------------------------------- 7. coverage
cov = bc.coverage_simulation(p_true=0.95, n=50, n_mc=1500, B=1500, seed=5)
for k, v in cov.items():
    print(f"    {k:18s} coverage={v['coverage']:.3f}  width={v['mean_width']:.4f}")
check("Wald under-covers at p=0.95, n=50", cov["wald"]["coverage"] < 0.93)
check("Clopper-Pearson is conservative (>= 0.95)", cov["clopper_pearson"]["coverage"] >= 0.95)
check("Wilson coverage near nominal", 0.92 <= cov["wilson"]["coverage"] <= 0.99)

cov2 = bc.coverage_simulation(p_true=0.85, n=500, n_mc=1200, B=1200, seed=6)
check("all methods ~nominal at n=500, p=0.85",
      all(0.93 <= v["coverage"] <= 0.97 for v in cov2.values()),
      str({k: round(v["coverage"], 3) for k, v in cov2.items()}))

print("\n" + ("ALL TESTS PASSED" if not FAIL else f"FAILURES: {FAIL}"))
