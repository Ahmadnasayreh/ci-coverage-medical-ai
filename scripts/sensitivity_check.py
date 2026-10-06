"""
sensitivity_check.py — sensitivity settings S3 and S4 (Table IV of the paper).

Extends the binormal coverage study (Table VI of the paper) with two settings
that change one ingredient at a time:

  S3  class imbalance   : prevalence 0.10 (instead of 0.40), same separation
                          and threshold as the moderate setting.
  S4  non-binormal ROC  : positives drawn from a two-component mixture
                          0.7 N(2.5,1) + 0.3 N(0,1) (30 % "hard" positives that
                          look like negatives), prevalence 0.40, threshold 1.25.

Population values stay available in closed form, because AUROC and TPR are
linear in the positive-class distribution:
    AUROC = sum_k w_k * Phi(mu_k / sqrt(1 + s_k^2))
    TPR   = sum_k w_k * (1 - Phi((t - mu_k) / s_k)),   FPR = 1 - Phi(t).

Same interval code (bootci.py), same M = 1000, B = 999, same n grid as the
main study.  Cohorts with fewer than 2 positives or 2 negatives are
skipped and counted (M_used).  Runtime ~6 min on one CPU core.

Run from the repository root:   python scripts/sensitivity_check.py
Output:                         results/table12_sensitivity.csv
Complexity: O(M * B * n log n) per setting, dominated by the AUROC ranks.
"""
import sys, time, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]          # repository root
sys.path.insert(0, str(ROOT))
import bootci as bc  # noqa: E402

SEED, M, B, ALPHA, CHUNK = 20240, 1000, 999, 0.05, 1000
N_GRID = [50, 100, 200, 500]
METRICS = ["accuracy", "f1", "mcc", "balanced_accuracy", "auroc"]
ARMS = ["percentile", "bca", "normal"]
Z = sps.norm.ppf(1 - ALPHA / 2)
Phi = sps.norm.cdf

SETTINGS = {
    # name: (prevalence, [(weight, mu, sd), ...] for positives, threshold)
    "S3 imbalanced (pi=0.10)": (0.10, [(1.0, 1.5, 1.0)], 0.75),
    "S4 bimodal positives":    (0.40, [(0.7, 2.5, 1.0), (0.3, 0.0, 1.0)], 1.25),
}


def truth(pi, comps, t):
    tpr = sum(w * (1 - Phi((t - m) / s)) for w, m, s in comps)
    fpr = 1 - Phi(t)
    tp, fn = pi * tpr, pi * (1 - tpr)
    fp, tn = (1 - pi) * fpr, (1 - pi) * (1 - fpr)
    return {
        "accuracy": tp + tn,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "mcc": (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
        "balanced_accuracy": 0.5 * (tpr + 1 - fpr),
        "auroc": sum(w * Phi(m / np.sqrt(1 + s ** 2)) for w, m, s in comps),
    }


def draw(n, pi, comps, t, rng):
    y = rng.binomial(1, pi, n)
    w = np.array([c[0] for c in comps])
    k = rng.choice(len(comps), size=n, p=w)
    mu = np.array([c[1] for c in comps])[k]
    sd = np.array([c[2] for c in comps])[k]
    s = np.where(y == 1, rng.normal(mu, sd), rng.normal(0.0, 1.0, n))
    return y, s, (s >= t).astype(np.int64)


def replicates(y, s, p, rng):
    out = {m: np.empty(B) for m in METRICS}
    pos = 0
    for idx in bc._iter_index_chunks(y.size, B, rng, None, CHUNK):
        b = idx.shape[0]
        for m in METRICS:
            out[m][pos:pos + b] = bc.METRICS[m](y[idx], p[idx], s[idx])
        pos += b
    return out


def jackknife(y, s, p):
    n = y.size
    loo = np.stack([np.delete(np.arange(n), i) for i in range(n)])
    return {m: bc.METRICS[m](y[loo], p[loo], s[loo]) for m in METRICS}


def verify_truth():
    rng = np.random.default_rng(0)
    for name, (pi, comps, t) in SETTINGS.items():
        tr = truth(pi, comps, t)
        y, s, p = draw(400_000, pi, comps, t, rng)
        emp = {m: float(bc.METRICS[m](y[None], p[None], s[None])[0]) for m in METRICS}
        err = max(abs(tr[m] - emp[m]) for m in METRICS)
        print(f"{name}: max |closed form - 400k MC| = {err:.4f}")
        assert err < 0.005


def run():
    rows = []
    for si, (name, (pi, comps, t)) in enumerate(SETTINGS.items()):
        tr = truth(pi, comps, t)
        print(f"\n{name}: " + ", ".join(f"{m}={tr[m]:.4f}" for m in METRICS))
        for n in N_GRID:
            rng = np.random.default_rng(SEED + 7919 * (si + 3) + n)
            hit = {(m, a): 0 for m in METRICS for a in ARMS}
            wid = {(m, a): 0.0 for m in METRICS for a in ARMS}
            dl_hit, used, skipped, t0 = 0, 0, 0, time.perf_counter()
            npos = []
            for _ in range(M):
                y, s, p = draw(n, pi, comps, t, rng)
                if y.sum() < 2 or (1 - y).sum() < 2:
                    skipped += 1
                    continue
                used += 1
                npos.append(int(y.sum()))
                star, jack = replicates(y, s, p, rng), jackknife(y, s, p)
                for m in METRICS:
                    th = float(bc.METRICS[m](y[None], p[None], s[None])[0])
                    st = star[m][np.isfinite(star[m])]
                    if not np.isfinite(th) or st.size < 10:
                        continue                      # metric undefined: counts as a miss
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        lo_p, hi_p = np.percentile(st, [2.5, 97.5])
                        lo_b, hi_b, _, _ = bc._bca_ci(th, st, jack[m], ALPHA)
                    se = st.std(ddof=1)
                    for a, (lo, hi) in {"percentile": (lo_p, hi_p), "bca": (lo_b, hi_b),
                                        "normal": (th - Z * se, th + Z * se)}.items():
                        if np.isfinite(lo):
                            hit[(m, a)] += int(lo <= tr[m] <= hi)
                            wid[(m, a)] += hi - lo
                dl = bc.delong_auc_ci(y, s, alpha=ALPHA)
                dl_hit += int(dl["ci_low"] <= tr["auroc"] <= dl["ci_high"])
            for m in METRICS:
                for a in ARMS:
                    c = hit[(m, a)] / used
                    rows.append(dict(setting=name, n=n, metric=m, method=a, true_value=tr[m],
                                     coverage=c, mc_se=np.sqrt(c * (1 - c) / used),
                                     mean_width=wid[(m, a)] / used, M_used=used,
                                     skipped=skipped, median_pos=float(np.median(npos))))
            c = dl_hit / used
            rows.append(dict(setting=name, n=n, metric="auroc", method="delong",
                             true_value=tr["auroc"], coverage=c, mc_se=np.sqrt(c * (1 - c) / used),
                             mean_width=np.nan, M_used=used, skipped=skipped,
                             median_pos=float(np.median(npos))))
            print(f"  n={n:3d}  used={used} skipped={skipped} median n+={np.median(npos):.0f} "
                  f"({time.perf_counter()-t0:.0f}s)  " +
                  "  ".join(f"{m[:4]}: pct {hit[(m,'percentile')]/used:.3f} bca {hit[(m,'bca')]/used:.3f}"
                            for m in ["f1", "mcc", "auroc"]) + f"  delong {dl_hit/used:.3f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    verify_truth()
    df = run()
    out = ROOT / "results" / "table12_sensitivity.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")
