"""
bootci.py
=========
Vectorised non-parametric bootstrap confidence intervals (CIs) for the
evaluation of medical machine-learning models.

Design goals
------------
1. *Correctness*  : every metric is computed with an explicit, tie-aware,
                    fully vectorised NumPy implementation that is unit-tested
                    against scikit-learn reference implementations.
2. *Scalability*  : B = 10,000 resamples are generated in memory-bounded
                    chunks, so the peak RAM footprint is O(chunk * n) and not
                    O(B * n).
3. *Reproducibility*: every stochastic routine takes an explicit `seed`, and
                    `numpy.random.Generator` (PCG64) is used rather than the
                    deprecated global legacy RandomState.
4. *Auditability* : classical closed-form intervals (Wald, Wilson,
                    Agresti-Coull, Clopper-Pearson, Jeffreys) and the
                    asymptotic DeLong AUC interval are included so that the
                    bootstrap can be benchmarked instead of merely asserted.

Complexity (n = test-set size, B = number of resamples)
-------------------------------------------------------
  index generation ............ time O(B n)          space O(chunk * n)
  threshold metrics (ACC, F1..) time O(B n)          space O(chunk * n)
  AUROC (tie-aware midranks) .. time O(B n log n)    space O(chunk * n)
  BCa jackknife ............... time O(n^2) [O(n) evaluations of an O(n) metric]
  DeLong AUC variance ......... time O(n log n)      space O(n)

Authors : A. Jaradat, A. Nasayreh, A. Bashkami, A. I. Alshdaifat, H. Gharaibeh
          (companion code for the CAISAIS 2026 paper).
Licence : MIT.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

__all__ = [
    "set_global_seed",
    "BootResult",
    "bootstrap_ci",
    "bootstrap_many",
    "paired_bootstrap_diff",
    "mcnemar_exact",
    "binomial_ci",
    "rate_ci_closed_form",
    "delong_auc_ci",
    "coverage_simulation",
    "format_ci",
    "METRICS",
]

# --------------------------------------------------------------------------- #
# 0. Reproducibility
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Seed every RNG that can influence the reported numbers.

    Torch is imported lazily so that the module remains usable in a
    NumPy-only environment.
    """
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # only affects legacy global calls; Generators are explicit
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # warn_only=True: a few cuDNN kernels have no deterministic variant.
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# 1. Tie-aware vectorised midranks  (needed for a correct bootstrap AUROC)
# --------------------------------------------------------------------------- #
def _midranks_2d(a: np.ndarray) -> np.ndarray:
    """Average ("mid") ranks of every row of `a`, ties resolved by averaging.

    Equivalent to ``scipy.stats.rankdata(a, method='average', axis=1)`` but
    written explicitly because ties are *guaranteed* under resampling with
    replacement (a duplicated observation produces an exactly duplicated
    score) and the tie correction is what makes the Mann-Whitney identity
    AUC = (R+ - n+(n+ +1)/2) / (n+ n-) exact.

    Complexity: O(B n log n) time, O(B n) space.
    """
    B, n = a.shape
    order = np.argsort(a, axis=1, kind="stable")
    s = np.take_along_axis(a, order, axis=1)

    pos = np.broadcast_to(np.arange(n), (B, n))

    starts = np.ones((B, n), dtype=bool)
    starts[:, 1:] = s[:, 1:] != s[:, :-1]
    ends = np.ones((B, n), dtype=bool)
    ends[:, :-1] = s[:, 1:] != s[:, :-1]

    first = np.maximum.accumulate(np.where(starts, pos, 0), axis=1)
    last = np.minimum.accumulate(np.where(ends, pos, n - 1)[:, ::-1], axis=1)[:, ::-1]

    ranks_sorted = (first + last) / 2.0 + 1.0  # 1-based average ranks
    ranks = np.empty((B, n), dtype=np.float64)
    np.put_along_axis(ranks, order, ranks_sorted, axis=1)
    return ranks


# --------------------------------------------------------------------------- #
# 2. Vectorised metric kernels
#    Every kernel receives the *resampled* matrices
#       yt : (B, n) int   true labels in {0, 1}
#       yp : (B, n) int   hard predictions in {0, 1}
#       ys : (B, n) float positive-class scores
#    and returns a (B,) vector.  Degenerate resamples yield np.nan.
# --------------------------------------------------------------------------- #
def _confusion(yt: np.ndarray, yp: np.ndarray):
    tp = np.sum((yt == 1) & (yp == 1), axis=1).astype(np.float64)
    tn = np.sum((yt == 0) & (yp == 0), axis=1).astype(np.float64)
    fp = np.sum((yt == 0) & (yp == 1), axis=1).astype(np.float64)
    fn = np.sum((yt == 1) & (yp == 0), axis=1).astype(np.float64)
    return tp, tn, fp, fn


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _accuracy(yt, yp, ys):
    return np.mean(yt == yp, axis=1).astype(np.float64)


def _sensitivity(yt, yp, ys):          # recall / TPR
    tp, tn, fp, fn = _confusion(yt, yp)
    return _safe_div(tp, tp + fn)


def _specificity(yt, yp, ys):          # TNR
    tp, tn, fp, fn = _confusion(yt, yp)
    return _safe_div(tn, tn + fp)


def _ppv(yt, yp, ys):                  # precision
    tp, tn, fp, fn = _confusion(yt, yp)
    return _safe_div(tp, tp + fp)


def _npv(yt, yp, ys):
    tp, tn, fp, fn = _confusion(yt, yp)
    return _safe_div(tn, tn + fn)


def _f1(yt, yp, ys):
    tp, tn, fp, fn = _confusion(yt, yp)
    return _safe_div(2 * tp, 2 * tp + fp + fn)


def _balanced_accuracy(yt, yp, ys):
    return 0.5 * (_sensitivity(yt, yp, ys) + _specificity(yt, yp, ys))


def _mcc(yt, yp, ys):
    tp, tn, fp, fn = _confusion(yt, yp)
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return _safe_div(num, den)


def _auroc(yt, yp, ys):
    """Tie-corrected AUROC via the Mann-Whitney U identity."""
    ranks = _midranks_2d(ys)
    n_pos = np.sum(yt == 1, axis=1).astype(np.float64)
    n_neg = np.sum(yt == 0, axis=1).astype(np.float64)
    r_pos = np.sum(ranks * (yt == 1), axis=1)
    num = r_pos - n_pos * (n_pos + 1.0) / 2.0
    return _safe_div(num, n_pos * n_neg)


METRICS: Dict[str, Callable] = {
    "accuracy": _accuracy,
    "sensitivity": _sensitivity,
    "specificity": _specificity,
    "ppv": _ppv,
    "npv": _npv,
    "f1": _f1,
    "balanced_accuracy": _balanced_accuracy,
    "mcc": _mcc,
    "auroc": _auroc,
}

NEEDS_SCORES = {"auroc"}


# --------------------------------------------------------------------------- #
# 3. Resampling engine
# --------------------------------------------------------------------------- #
def _iter_index_chunks(
    n: int,
    B: int,
    rng: np.random.Generator,
    strata: Optional[np.ndarray] = None,
    chunk_size: int = 2000,
) -> Iterable[np.ndarray]:
    """Yield (chunk, n) matrices of resampling indices.

    strata is None -> i.i.d. bootstrap: indices ~ Uniform{0..n-1}.
    strata given   -> stratified bootstrap: resample *within* each class, so
                      the class prevalence of every replicate equals that of
                      the observed test set.  This removes the extra variance
                      caused by fluctuating prevalence and prevents degenerate
                      single-class replicates when the minority class is tiny.
                      NOTE: it conditions on the observed prevalence, hence it
                      answers "what if we re-drew patients within each arm"
                      rather than "what if we re-drew the cohort".  Report the
                      scheme explicitly in the manuscript.
    """
    if strata is not None:
        groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    done = 0
    while done < B:
        b = int(min(chunk_size, B - done))
        if strata is None:
            idx = rng.integers(0, n, size=(b, n), dtype=np.int64)
        else:
            idx = np.empty((b, n), dtype=np.int64)
            col = 0
            for gi in groups:
                k = gi.size
                idx[:, col : col + k] = gi[rng.integers(0, k, size=(b, k))]
                col += k
        yield idx
        done += b


def _replicates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    metric: str,
    B: int,
    rng: np.random.Generator,
    stratified: bool,
    chunk_size: int,
) -> np.ndarray:
    fn = METRICS[metric]
    n = y_true.size
    strata = y_true if stratified else None
    out = np.empty(B, dtype=np.float64)
    pos = 0
    ys_src = y_score if y_score is not None else np.zeros(n)
    for idx in _iter_index_chunks(n, B, rng, strata, chunk_size):
        b = idx.shape[0]
        out[pos : pos + b] = fn(y_true[idx], y_pred[idx], ys_src[idx])
        pos += b
    return out


# --------------------------------------------------------------------------- #
# 4. Interval constructors
# --------------------------------------------------------------------------- #
def _percentile_ci(theta_star: np.ndarray, alpha: float) -> Tuple[float, float]:
    lo, hi = np.nanpercentile(theta_star, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _basic_ci(theta_hat: float, theta_star: np.ndarray, alpha: float) -> Tuple[float, float]:
    lo, hi = _percentile_ci(theta_star, alpha)
    return float(2 * theta_hat - hi), float(2 * theta_hat - lo)


def _bca_ci(
    theta_hat: float,
    theta_star: np.ndarray,
    jack: np.ndarray,
    alpha: float,
    mid_p: bool = False,
) -> Tuple[float, float, float, float]:
    """Bias-corrected and accelerated interval (Efron, 1987).

    Returns (lo, hi, z0, a).

    `mid_p=False` reproduces the textbook definition
        z0 = Phi^-1( #{theta* < theta_hat} / B ).
    For discrete metrics on small test sets the bootstrap distribution is
    lattice-valued and a large point mass sits exactly at theta_hat; the
    mid-p variant  ( #{<} + 0.5 #{=} ) / B  is then markedly better behaved.
    Both are exposed; the manuscript must state which one was used.
    """
    ts = theta_star[np.isfinite(theta_star)]
    B = ts.size
    if B == 0:
        return np.nan, np.nan, np.nan, np.nan

    n_less = float(np.sum(ts < theta_hat))
    if mid_p:
        n_less += 0.5 * float(np.sum(ts == theta_hat))
    prop = n_less / B
    prop = min(max(prop, 0.5 / B), 1.0 - 0.5 / B)  # keep z0 finite
    z0 = float(stats.norm.ppf(prop))

    jk = jack[np.isfinite(jack)]
    jbar = jk.mean()
    d = jbar - jk
    denom = 6.0 * (np.sum(d**2) ** 1.5)
    a = float(np.sum(d**3) / denom) if denom > 0 else 0.0

    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    def _adj(z):
        val = z0 + (z0 + z) / (1.0 - a * (z0 + z))
        return float(stats.norm.cdf(val))

    a1, a2 = _adj(z_lo), _adj(z_hi)
    if not np.isfinite(a1) or not np.isfinite(a2) or a1 >= a2:
        warnings.warn("BCa adjustment degenerate; falling back to percentile.")
        lo, hi = _percentile_ci(ts, alpha)
        return lo, hi, z0, a
    lo, hi = np.percentile(ts, [100 * a1, 100 * a2])
    return float(lo), float(hi), z0, a


def _jackknife(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    metric: str,
    max_n: int = 5000,
) -> np.ndarray:
    """Leave-one-out replicates required by the BCa acceleration constant."""
    n = y_true.size
    if n > max_n:
        warnings.warn(f"n={n} > max_n={max_n}: jackknife skipped, BCa unavailable.")
        return np.full(1, np.nan)
    fn = METRICS[metric]
    base = np.arange(n)
    # (n, n-1) leave-one-out index matrix; 8 bytes * n * (n-1) — fine for n <= 5000.
    loo = np.stack([np.delete(base, i) for i in range(n)])
    ys_src = y_score if y_score is not None else np.zeros(n)
    return fn(y_true[loo], y_pred[loo], ys_src[loo])


# --------------------------------------------------------------------------- #
# 5. Public API
# --------------------------------------------------------------------------- #
@dataclass
class BootResult:
    metric: str
    estimate: float
    ci_low: float
    ci_high: float
    method: str
    alpha: float
    B: int
    n: int
    se_boot: float
    bias_boot: float
    n_valid: int
    stratified: bool
    boundary_degenerate: bool = False
    z0: Optional[float] = None
    accel: Optional[float] = None
    replicates: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low

    @property
    def moe(self) -> float:
        """Half-width, i.e. the margin of error in +/- notation."""
        return 0.5 * (self.ci_high - self.ci_low)

    def as_row(self) -> Dict:
        return {
            "metric": self.metric,
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "width": self.width,
            "moe": self.moe,
            "se_boot": self.se_boot,
            "bias_boot": self.bias_boot,
            "method": self.method,
            "B": self.B,
            "n": self.n,
            "n_valid": self.n_valid,
            "stratified": self.stratified,
            "boundary_degenerate": self.boundary_degenerate,
            "z0": self.z0,
            "accel": self.accel,
        }

    def __str__(self) -> str:
        return format_ci(self.estimate, self.ci_low, self.ci_high)


def format_ci(est: float, lo: float, hi: float, digits: int = 4) -> str:
    return f"{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    y_score: Optional[Sequence] = None,
    metric: str = "accuracy",
    B: int = 10_000,
    alpha: float = 0.05,
    method: str = "bca",           # {"percentile", "basic", "bca"}
    seed: int = 42,
    stratified: bool = False,
    chunk_size: int = 2000,
    mid_p: bool = False,
    return_replicates: bool = True,
) -> BootResult:
    """Non-parametric bootstrap CI for a single metric on a fixed test set.

    The resampling unit is the *patient/image*, i.e. the model is held fixed
    and only the finite-test-set sampling error is quantified.  This is the
    quantity a clinician needs when asking "would this accuracy hold on the
    next cohort drawn from the same population?"  It does *not* capture
    training-set variability (see `Discussion` in the manuscript).
    """
    y_true = np.asarray(y_true).ravel().astype(np.int64)
    y_pred = np.asarray(y_pred).ravel().astype(np.int64)
    y_score = None if y_score is None else np.asarray(y_score, dtype=np.float64).ravel()

    if metric not in METRICS:
        raise KeyError(f"unknown metric '{metric}'; available: {sorted(METRICS)}")
    if metric in NEEDS_SCORES and y_score is None:
        raise ValueError(f"metric '{metric}' requires y_score")
    if not (y_true.size == y_pred.size and (y_score is None or y_score.size == y_true.size)):
        raise ValueError("y_true, y_pred and y_score must have identical length")

    n = y_true.size
    rng = np.random.default_rng(seed)
    fn = METRICS[metric]
    theta_hat = float(fn(y_true[None, :], y_pred[None, :],
                         (y_score if y_score is not None else np.zeros(n))[None, :])[0])

    ts = _replicates(y_true, y_pred, y_score, metric, B, rng, stratified, chunk_size)
    valid = np.isfinite(ts)
    n_valid = int(valid.sum())
    if n_valid < 0.9 * B:
        warnings.warn(
            f"{B - n_valid}/{B} degenerate replicates for metric '{metric}'. "
            "Consider stratified=True (small minority class)."
        )

    # --- Failure mode: if the observed statistic sits exactly on the boundary
    # of the parameter space (e.g. specificity = 1.000 because no negative was
    # misclassified), every bootstrap replicate is identical and the interval
    # collapses to zero width.  The bootstrap is then NOT valid: the empirical
    # distribution carries no information about how far below 1 the true value
    # may lie.  Report `rate_ci_closed_form(..., method='clopper_pearson')`
    # instead (see Section "Boundary failure" of the manuscript).
    boundary = bool(np.nanstd(ts) < 1e-12) or theta_hat in (0.0, 1.0)
    if boundary:
        warnings.warn(
            f"metric '{metric}' = {theta_hat:.4f} lies on the boundary of the "
            "parameter space: the non-parametric bootstrap degenerates "
            "(zero-width interval). Use rate_ci_closed_form() / Clopper-Pearson."
        )

    z0 = accel = None
    if method == "percentile":
        lo, hi = _percentile_ci(ts, alpha)
    elif method == "basic":
        lo, hi = _basic_ci(theta_hat, ts, alpha)
    elif method == "bca":
        jack = _jackknife(y_true, y_pred, y_score, metric)
        if np.all(np.isnan(jack)):
            lo, hi = _percentile_ci(ts, alpha)
            method = "percentile(bca-fallback)"
        else:
            lo, hi, z0, accel = _bca_ci(theta_hat, ts, jack, alpha, mid_p=mid_p)
    else:
        raise KeyError("method must be one of {'percentile', 'basic', 'bca'}")

    return BootResult(
        metric=metric,
        estimate=theta_hat,
        ci_low=lo,
        ci_high=hi,
        method=method,
        alpha=alpha,
        B=B,
        n=n,
        se_boot=float(np.nanstd(ts, ddof=1)),
        bias_boot=float(np.nanmean(ts) - theta_hat),
        n_valid=n_valid,
        stratified=stratified,
        boundary_degenerate=boundary,
        z0=z0,
        accel=accel,
        replicates=ts if return_replicates else None,
    )


def bootstrap_many(
    y_true, y_pred, y_score=None, metrics: Sequence[str] = ("accuracy",), **kw
) -> Dict[str, BootResult]:
    """Convenience wrapper: identical seed => identical resamples across metrics,
    which keeps the reported intervals mutually consistent."""
    return {m: bootstrap_ci(y_true, y_pred, y_score, metric=m, **kw) for m in metrics}


# --------------------------------------------------------------------------- #
# 6. Paired comparison of two models on the SAME test set
# --------------------------------------------------------------------------- #
def paired_bootstrap_diff(
    y_true: Sequence,
    pred_a: Sequence,
    pred_b: Sequence,
    score_a: Optional[Sequence] = None,
    score_b: Optional[Sequence] = None,
    metric: str = "accuracy",
    B: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
    stratified: bool = False,
    chunk_size: int = 2000,
) -> Dict:
    """CI and achieved significance level (ASL) for Delta = metric(A) - metric(B).

    The *same* resampled indices are applied to both models, so the strong
    positive correlation between two models evaluated on one cohort is
    preserved.  Ignoring the pairing (comparing two independent CIs and
    checking for overlap) is a well-known conservative error: non-overlapping
    intervals imply significance, but overlapping intervals do NOT imply
    non-significance.
    """
    y_true = np.asarray(y_true).ravel().astype(np.int64)
    pred_a = np.asarray(pred_a).ravel().astype(np.int64)
    pred_b = np.asarray(pred_b).ravel().astype(np.int64)
    n = y_true.size
    zeros = np.zeros(n)
    sa = zeros if score_a is None else np.asarray(score_a, float).ravel()
    sb = zeros if score_b is None else np.asarray(score_b, float).ravel()
    if metric in NEEDS_SCORES and (score_a is None or score_b is None):
        raise ValueError(f"metric '{metric}' requires score_a and score_b")

    fn = METRICS[metric]
    theta_a = float(fn(y_true[None, :], pred_a[None, :], sa[None, :])[0])
    theta_b = float(fn(y_true[None, :], pred_b[None, :], sb[None, :])[0])
    delta_hat = theta_a - theta_b

    rng = np.random.default_rng(seed)
    strata = y_true if stratified else None
    d = np.empty(B, dtype=np.float64)
    pos = 0
    for idx in _iter_index_chunks(n, B, rng, strata, chunk_size):
        b = idx.shape[0]
        yt = y_true[idx]
        d[pos : pos + b] = fn(yt, pred_a[idx], sa[idx]) - fn(yt, pred_b[idx], sb[idx])
        pos += b

    dv = d[np.isfinite(d)]
    lo, hi = np.percentile(dv, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided ASL under H0: Delta = 0, obtained by centring the bootstrap
    # distribution (Efron & Tibshirani, 1993, ch. 16).
    asl = float(np.mean(np.abs(dv - delta_hat) >= abs(delta_hat)))
    return {
        "metric": metric,
        "theta_a": theta_a,
        "theta_b": theta_b,
        "delta": delta_hat,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "asl_p": asl,
        "significant_at_alpha": bool(lo > 0 or hi < 0),
        "B": B,
        "n": n,
        "replicates": dv,
    }


def mcnemar_exact(y_true, pred_a, pred_b) -> Dict:
    """Exact (binomial) McNemar test — closed-form cross-check for the paired
    bootstrap on *accuracy*.  Discordant pairs only."""
    y_true = np.asarray(y_true).ravel()
    a_ok = np.asarray(pred_a).ravel() == y_true
    b_ok = np.asarray(pred_b).ravel() == y_true
    n01 = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
    n10 = int(np.sum(~a_ok & b_ok))   # A wrong, B right
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    p = stats.binomtest(n01, n01 + n10, 0.5, alternative="two-sided").pvalue
    return {"n01": n01, "n10": n10, "p_value": float(p)}


# --------------------------------------------------------------------------- #
# 7. Closed-form binomial intervals (benchmarks for the bootstrap)
# --------------------------------------------------------------------------- #
def binomial_ci(k: int, n: int, alpha: float = 0.05, method: str = "wilson") -> Tuple[float, float]:
    """Interval for a binomial proportion.

    method in {"wald", "wilson", "wilson_cc", "agresti_coull",
               "clopper_pearson", "jeffreys"}.
    Wald is included *only* as the negative control criticised in the paper:
    its coverage collapses when n is small or p approaches 0/1, and it can
    return limits outside [0, 1] (Brown, Cai & DasGupta, 2001).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    z = stats.norm.ppf(1 - alpha / 2)

    if method == "wald":
        se = np.sqrt(p * (1 - p) / n)
        return p - z * se, p + z * se           # deliberately NOT clipped
    if method in ("wilson", "wilson_cc"):
        cc = 0.5 / n if method == "wilson_cc" else 0.0
        d = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / d
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2) + cc / n) / d
        return max(0.0, centre - half), min(1.0, centre + half)
    if method == "agresti_coull":
        n_t = n + z**2
        p_t = (k + z**2 / 2) / n_t
        half = z * np.sqrt(p_t * (1 - p_t) / n_t)
        return max(0.0, p_t - half), min(1.0, p_t + half)
    if method == "clopper_pearson":
        lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
        hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
        return float(lo), float(hi)
    if method == "jeffreys":
        lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k + 0.5, n - k + 0.5)
        hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)
        return float(lo), float(hi)
    raise KeyError(f"unknown method '{method}'")


RATE_DENOMINATOR = {
    "accuracy": "all",
    "sensitivity": "true_pos",
    "specificity": "true_neg",
    "ppv": "pred_pos",
    "npv": "pred_neg",
}


def rate_ci_closed_form(
    y_true, y_pred, metric: str = "accuracy", alpha: float = 0.05,
    method: str = "clopper_pearson"
) -> Dict:
    """Closed-form binomial interval for any metric that is a simple rate.

    Each of accuracy / sensitivity / specificity / PPV / NPV is a proportion
    k/m over an explicitly defined denominator, so an exact or score interval
    applies directly and — unlike the non-parametric bootstrap — remains valid
    when the observed rate is exactly 0 or 1.

    Caveat: for PPV and NPV the denominator (number of predicted positives /
    negatives) is itself random; conditioning on it is the standard, mildly
    anti-conservative convention and must be stated in the manuscript.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    if metric not in RATE_DENOMINATOR:
        raise KeyError(f"'{metric}' is not a simple rate; use bootstrap_ci().")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    table = {
        "accuracy": (tp + tn, tp + tn + fp + fn),
        "sensitivity": (tp, tp + fn),
        "specificity": (tn, tn + fp),
        "ppv": (tp, tp + fp),
        "npv": (tn, tn + fn),
    }
    k, m = table[metric]
    if m == 0:
        return {"metric": metric, "k": k, "n": m, "estimate": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "method": method}
    lo, hi = binomial_ci(k, m, alpha=alpha, method=method)
    return {"metric": metric, "k": k, "n": m, "estimate": k / m,
            "ci_low": float(lo), "ci_high": float(hi), "method": method}


# --------------------------------------------------------------------------- #
# 8. DeLong asymptotic AUC interval (parametric benchmark)
# --------------------------------------------------------------------------- #
def _midrank_1d(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def delong_auc_ci(y_true, y_score, alpha: float = 0.05) -> Dict:
    """AUC, its DeLong variance and the corresponding normal interval.

    Implementation follows the O(n log n) formulation of Sun & Xu (2014) for
    the single-classifier case, using the structural components (placement
    values) of DeLong, DeLong & Clarke-Pearson (1988).
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = pos.size, neg.size
    if m == 0 or n == 0:
        return {"auc": np.nan, "var": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    tz = _midrank_1d(np.concatenate([pos, neg]))
    tx, ty = _midrank_1d(pos), _midrank_1d(neg)
    auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)

    v01 = (tz[:m] - tx) / n            # placement values for positives
    v10 = 1.0 - (tz[m:] - ty) / m      # placement values for negatives
    s01 = np.var(v01, ddof=1) if m > 1 else 0.0
    s10 = np.var(v10, ddof=1) if n > 1 else 0.0
    var = s01 / m + s10 / n
    se = float(np.sqrt(max(var, 0.0)))
    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "auc": float(auc),
        "var": float(var),
        "se": se,
        "ci_low": float(auc - z * se),
        "ci_high": float(auc + z * se),
    }


# --------------------------------------------------------------------------- #
# 9. Monte-Carlo coverage study
# --------------------------------------------------------------------------- #
def coverage_simulation(
    p_true: float,
    n: int,
    n_mc: int = 2000,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    methods: Sequence[str] = ("wald", "wilson", "clopper_pearson",
                              "boot_percentile", "boot_bca"),
) -> Dict[str, Dict[str, float]]:
    """Empirical coverage of nominal (1-alpha) intervals for a *known* accuracy.

    Exact shortcut used for the bootstrap arms
    ------------------------------------------
    For the mean of a binary vector containing k ones out of n, resampling n
    observations with replacement yields a replicate mean k*/n with
    k* ~ Binomial(n, k/n) *exactly*.  Drawing the binomial directly is
    therefore not an approximation of the bootstrap: it is the bootstrap,
    which reduces the cost of the study from O(n_mc B n) to O(n_mc B).

    The leave-one-out (jackknife) values of a binary mean take only two
    distinct values, so the BCa acceleration is likewise available in closed
    form.
    """
    rng = np.random.default_rng(seed)
    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    hits = {m: 0 for m in methods}
    widths = {m: 0.0 for m in methods}

    ks = rng.binomial(n, p_true, size=n_mc)
    for k in ks:
        p_hat = k / n
        for m in methods:
            if m.startswith("boot_"):
                star = rng.binomial(n, p_hat, size=B) / n
                if m == "boot_percentile":
                    lo, hi = np.percentile(star, [100 * alpha / 2, 100 * (1 - alpha / 2)])
                else:  # boot_bca / boot_bca_midp
                    prop = np.mean(star < p_hat)
                    if m.endswith("midp"):
                        prop += 0.5 * np.mean(star == p_hat)
                    prop = min(max(prop, 0.5 / B), 1 - 0.5 / B)
                    z0 = stats.norm.ppf(prop)
                    # jackknife of a binary mean: k ones -> two distinct LOO values
                    jack = np.concatenate([
                        np.full(k, (k - 1) / (n - 1)),
                        np.full(n - k, k / (n - 1)),
                    ]) if n > 1 else np.array([p_hat])
                    d = jack.mean() - jack
                    den = 6.0 * (np.sum(d**2) ** 1.5)
                    a = float(np.sum(d**3) / den) if den > 0 else 0.0
                    a1 = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
                    a2 = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
                    if not np.isfinite(a1) or not np.isfinite(a2) or a1 >= a2:
                        lo, hi = np.percentile(star, [100 * alpha / 2, 100 * (1 - alpha / 2)])
                    else:
                        lo, hi = np.percentile(star, [100 * a1, 100 * a2])
            else:
                lo, hi = binomial_ci(int(k), n, alpha, method=m)
            hits[m] += int(lo <= p_true <= hi)
            widths[m] += hi - lo

    return {
        m: {
            "coverage": hits[m] / n_mc,
            "mean_width": widths[m] / n_mc,
            "mc_se": float(np.sqrt((hits[m] / n_mc) * (1 - hits[m] / n_mc) / n_mc)),
        }
        for m in methods
    }
