"""Scene-level paired statistics (thesis §3.4.3).

The source scene is the independent unit.  Paired differences are tested with a
two-sided sign-flip permutation test (Monte-Carlo, fixed RNG, add-one p-value),
summarised with Cohen's d_z and a scene-cluster bootstrap CI; secondary
families use Holm correction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "PairedResult",
    "bootstrap_ci",
    "cohens_dz",
    "holm",
    "one_sided_t_bound",
    "paired_comparison",
    "sign_flip_p",
    "stars",
]


def sign_flip_p(diffs: npt.ArrayLike, n_flips: int = 20_000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation p-value with add-one correction."""
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        return math.nan
    rng = np.random.default_rng(seed)
    obs = abs(float(d.mean()))
    perm = np.abs((rng.choice([-1.0, 1.0], size=(n_flips, d.size)) * d).mean(axis=1))
    return float((int((perm >= obs - 1e-12).sum()) + 1) / (n_flips + 1))


def bootstrap_ci(
    values: npt.ArrayLike, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile CI of the mean under (scene-cluster) resampling."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def cohens_dz(diffs: npt.ArrayLike) -> float:
    d = np.asarray(diffs, dtype=np.float64)
    if d.size < 2:
        return math.nan
    sd = float(d.std(ddof=1))
    if sd <= 1e-12:
        return math.inf if d.mean() != 0 else math.nan
    return float(d.mean() / sd)


def one_sided_t_bound(values: npt.ArrayLike, side: str = "lower", level: float = 0.95) -> float:
    """One-sided t confidence bound on the mean (non-inferiority guardrails)."""
    from scipy import stats

    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        return math.nan
    se = float(x.std(ddof=1)) / math.sqrt(x.size)
    t = float(stats.t.ppf(level, x.size - 1))
    return float(x.mean()) - t * se if side == "lower" else float(x.mean()) + t * se


def holm(pvals: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, adj = len(items), 0.0, {}
    for rank, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adj[k] = running
    return adj


def stars(p: float) -> str:
    if not math.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


@dataclass(frozen=True)
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    median_diff: float
    ci95: tuple[float, float]
    p: float
    dz: float
    wins: int  # a < b  (useful when lower is better)
    ties: int
    losses: int

    def to_dict(self) -> dict:
        return asdict(self)


def paired_comparison(
    a: Mapping[str, float | None],
    b: Mapping[str, float | None],
    n_flips: int = 20_000,
    n_boot: int = 10_000,
    seed: int = 0,
) -> PairedResult:
    """Paired analysis of ``a - b`` over the scenes both mappings share with finite values."""

    def ok(x: float | None) -> bool:
        return x is not None and math.isfinite(x)

    keys = sorted(k for k in set(a) & set(b) if ok(a[k]) and ok(b[k]))
    va = np.asarray([a[k] for k in keys], dtype=np.float64)
    vb = np.asarray([b[k] for k in keys], dtype=np.float64)
    d = va - vb
    if d.size == 0:
        nan = math.nan
        return PairedResult(0, nan, nan, nan, nan, (nan, nan), nan, nan, 0, 0, 0)
    return PairedResult(
        n=int(d.size),
        mean_a=float(va.mean()),
        mean_b=float(vb.mean()),
        mean_diff=float(d.mean()),
        median_diff=float(np.median(d)),
        ci95=bootstrap_ci(d, n_boot, seed),
        p=sign_flip_p(d, n_flips, seed),
        dz=cohens_dz(d),
        wins=int((d < 0).sum()),
        ties=int((d == 0).sum()),
        losses=int((d > 0).sum()),
    )
