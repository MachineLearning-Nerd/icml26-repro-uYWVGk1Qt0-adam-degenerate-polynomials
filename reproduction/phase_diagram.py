"""Phase-diagram and three-regime analysis for Claims 5-6.

Reproduces Theorem 6.1 (Section 6) and the empirical phase diagram (Figure 3 / Figure 7)
of arXiv:2603.09581 on the scalar degenerate polynomial family ``L(x) = x^k / k``.

All dynamics use the paper's uncorrected, epsilon-free Adam recurrence
(``x_{t+1} = x_t - eta * m_t / sqrt(v_t)``), initialized at ``x0=1, m0=g0, v0=g0^2``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def stable_bound(k: int, beta2: float) -> float:
    r"""Eq. (8): primary stability boundary  beta1 = beta2^{k/(2(k-2))}."""
    return beta2 ** (k / (2 * (k - 2)))


def exist_bound(k: int, beta2: float) -> float:
    r"""Existence boundary  beta1 = beta2^{(k-1)/(2(k-2))}  (Theorem 4.1 part i)."""
    return beta2 ** ((k - 1) / (2 * (k - 2)))


def lower_corner_bound(k: int, beta2: float) -> float:
    r"""Eq. (9): lower-left corner stability constraint on beta1.

    beta1 > [(k-2) beta2^{-1/(2(k-2))} - k] / [(k - (k-2) beta2^{1/(2(k-2))}) beta2^{-k/(2(k-2))}]
    """
    e = 1 / (2 * (k - 2))
    num = (k - 2) * beta2 ** (-e) - k
    den = (k - (k - 2) * beta2 ** e) * beta2 ** (-k / (2 * (k - 2)))
    return num / den


def theoretical_regime(k: int, beta1: float, beta2: float) -> str:
    """Classify (beta1, beta2) per Theorem 6.1 / Section 6.1.

    Precondition: beta2 > ((k-2)/k)^{2(k-2)}  (always satisfied for practical beta2).
    Returns one of: 'I_stable', 'II_spike', 'III_signgd'.
    """
    eb = exist_bound(k, beta2)
    sb = stable_bound(k, beta2)
    if beta1 > eb:
        return "III_signgd"
    elif beta1 > sb:
        return "II_spike"
    else:
        lb = lower_corner_bound(k, beta2)
        if beta1 < lb:
            return "II_spike"
        return "I_stable"


def signgd_loss(eta: float, k: int) -> float:
    r"""L(eta/2) = (eta/2)^k / k — the SignGD oscillation loss level."""
    return (eta / 2) ** k / k


@dataclass
class TrajectoryResult:
    steps: np.ndarray
    x: np.ndarray
    loss: np.ndarray
    m: np.ndarray
    v: np.ndarray
    g_sq: np.ndarray


def adam_trajectory(
    k: int, beta1: float, beta2: float, eta: float,
    x0: float = 1.0, steps: int = 100_000, sample: int = 100,
) -> TrajectoryResult:
    """Single Adam trajectory in float64.

    Samples every ``sample`` steps (plus the first 10 steps) for diagnostics.
    """
    x = float(x0)
    g0 = x0 ** (k - 1)
    m, v = g0, g0 * g0

    ts, xs, losses, ms, vs, gsqs = [], [], [], [], [], []
    for t in range(1, steps + 1):
        g = x ** (k - 1)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        sv = math.sqrt(v)
        if sv < 1e-300:
            x = 0.0
        else:
            x = x - eta * m / sv
        if t <= 10 or t % sample == 0:
            ts.append(t)
            xs.append(abs(x))
            losses.append(abs(x) ** k / k)
            ms.append(m)
            vs.append(v)
            gsqs.append(g * g)

    arr = lambda lst: np.asarray(lst, dtype=np.float64)
    return TrajectoryResult(arr(ts), arr(xs), arr(losses), arr(ms), arr(vs), arr(gsqs))


def adam_grid(
    k: int, beta1_vals: np.ndarray, beta2_vals: np.ndarray,
    eta: float, x0: float = 1.0, steps: int = 100_000,
    record_trajectories: bool = False, traj_sample: int = 200,
) -> dict:
    """Vectorised Adam sweep over a (beta1, beta2) grid.

    Returns a dict with keys:
        beta1_grid, beta2_grid : 2-D meshgrid arrays (ij indexing)
        min_loss, final_loss    : 2-D arrays of min_t L(x_t) and L(x_T)
        coupling_ratio_max      : 2-D array of max_t (v_t / g_t^2)
        trajectories            : dict of {flat_index: TrajectoryResult} (optional)
    """
    B1, B2 = np.meshgrid(beta1_vals, beta2_vals, indexing="ij")
    b1 = B1.ravel()
    b2 = B2.ravel()
    n = b1.size

    x = np.full(n, float(x0))
    g = x ** (k - 1)
    m = g.copy()
    v = g * g

    loss = np.abs(x) ** k / k
    min_loss = loss.copy()
    coupling_ratio_max = np.ones(n)

    traj_store: dict[int, TrajectoryResult] = {}
    traj_indices: set[int] = set()
    if record_trajectories:
        traj_indices = set(range(n))

    traj_t, traj_x, traj_loss = {i: [] for i in traj_indices}, {i: [] for i in traj_indices}, {i: [] for i in traj_indices}
    traj_m, traj_v, traj_gsqs = {i: [] for i in traj_indices}, {i: [] for i in traj_indices}, {i: [] for i in traj_indices}

    for t in range(1, steps + 1):
        g = x ** (k - 1)
        gsq = g * g
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * gsq
        sv = np.sqrt(v)
        step = np.where(sv > 1e-300, eta * m / np.where(sv > 1e-300, sv, 1.0), 0.0)
        x = x - step
        loss = np.abs(x) ** k / k
        min_loss = np.minimum(min_loss, loss)
        ratio = np.where(gsq > 1e-320, v / np.where(gsq > 1e-320, gsq, 1.0), 1.0)
        coupling_ratio_max = np.maximum(coupling_ratio_max, ratio)

        if traj_indices and (t <= 10 or t % traj_sample == 0):
            for i in traj_indices:
                traj_t[i].append(t)
                traj_x[i].append(abs(x[i]))
                traj_loss[i].append(loss[i])
                traj_m[i].append(m[i])
                traj_v[i].append(v[i])
                traj_gsqs[i].append(gsq[i])

    final_loss = np.abs(x) ** k / k

    result = {
        "beta1_grid": B1,
        "beta2_grid": B2,
        "min_loss": min_loss.reshape(B1.shape),
        "final_loss": final_loss.reshape(B1.shape),
        "coupling_ratio_max": coupling_ratio_max.reshape(B1.shape),
    }
    if record_trajectories:
        for i in traj_indices:
            arr = lambda lst: np.asarray(lst, dtype=np.float64)
            traj_store[i] = TrajectoryResult(
                arr(traj_t[i]), arr(traj_x[i]), arr(traj_loss[i]),
                arr(traj_m[i]), arr(traj_v[i]), arr(traj_gsqs[i]),
            )
        result["trajectories"] = traj_store
    return result


def classify_empirical(min_loss: float, final_loss: float, signgd_thr: float) -> str:
    """Classify a single grid cell from empirical (min_loss, final_loss).

    Uses L(eta/2) as the SignGD reference level:
      - I_stable   : final_loss well below the SignGD level (converged).
      - II_spike   : min_loss well below but final_loss at/above the SignGD level.
      - III_signgd : min_loss never drops well below the SignGD level.
    """
    low = signgd_thr * 1e-3
    if final_loss < low:
        return "I_stable"
    if min_loss < low:
        return "II_spike"
    return "III_signgd"


def grid_alignment(k: int, beta1_vals: np.ndarray, beta2_vals: np.ndarray,
                   grid: dict, eta: float) -> dict:
    """Compute theoretical-vs-empirical alignment metrics for the phase diagram.

    Returns per-cell theoretical regime, empirical regime, and summary
    classification-accuracy statistics.
    """
    thr = signgd_loss(eta, k)
    B1, B2 = grid["beta1_grid"], grid["beta2_grid"]
    min_l, fin_l = grid["min_loss"], grid["final_loss"]

    theo = np.empty(B1.shape, dtype=object)
    emp = np.empty(B1.shape, dtype=object)
    for i in range(B1.shape[0]):
        for j in range(B1.shape[1]):
            t = theoretical_regime(k, B1[i, j], B2[i, j])
            e = classify_empirical(min_l[i, j], fin_l[i, j], thr)
            theo[i, j] = t
            emp[i, j] = e

    match = (theo == emp)
    accuracy = match.mean()

    per_regime = {}
    for regime in ("I_stable", "II_spike", "III_signgd"):
        mask = theo == regime
        n = mask.sum()
        per_regime[regime] = {
            "n_cells": int(n),
            "empirical_match_frac": float(match[mask].mean()) if n > 0 else float("nan"),
        }

    return {
        "theoretical": theo,
        "empirical": emp,
        "match": match,
        "accuracy": float(accuracy),
        "per_regime": per_regime,
        "signgd_threshold": thr,
    }
