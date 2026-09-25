"""Scoring for the game-theory benchmark.

Every metric is reported against an explicit normative reference, so a number
can be read as "how far from the game-theoretic answer" rather than as a bare
score. Invariance metrics are the strictest: their reference is exactly zero,
because the manipulations they measure are ones that provably cannot change
what a payoff maximiser should do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 20260922
READOUTS = ["p_coop_choice", "p_coop_noul", "p_coop_score"]


def bootstrap_ci(values, statistic=np.mean, draws=10000, alpha=0.05, seed=RNG_SEED):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(values))
    if len(values) == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    stats = statistic(values[idx], axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def accuracy_table(frame: pd.DataFrame, by) -> pd.DataFrame:
    """Share of trials whose picked action is the unambiguous best response."""
    scored = frame[frame.best_response.notna()].copy()
    scored["correct"] = scored.picked_cooperate == (scored.best_response == "cooperate")
    scored["p_best"] = np.where(scored.best_response == "cooperate",
                                scored.p_coop_choice, 1 - scored.p_coop_choice)
    rows = []
    for key, group in scored.groupby(by, dropna=False):
        point, lo, hi = bootstrap_ci(group.p_best.to_numpy())
        keys = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by if isinstance(by, list) else [by], keys)),
                     "n": len(group),
                     "best_response": group.best_response.iloc[0]
                     if group.best_response.nunique() == 1 else "mixed",
                     "accuracy": float(group.correct.mean()),
                     "p_best_action": point, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def invariance_table(frame: pd.DataFrame, factor: str, within) -> pd.DataFrame:
    """Spread of P(cooperate) across a factor that must not matter.

    The reference value is zero: these manipulations are positive affine
    transforms of the payoffs, or pure rewordings, so an expected-utility
    maximiser's answer is provably unchanged by them.
    """
    rows = []
    for key, group in frame.groupby(within, dropna=False):
        cell = group.groupby(factor)["p_coop_choice"].mean()
        if len(cell) < 2:
            continue
        keys = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(within if isinstance(within, list) else [within], keys)),
                     "levels": len(cell),
                     "min": float(cell.min()), "max": float(cell.max()),
                     "range": float(cell.max() - cell.min()),
                     "sd": float(cell.std(ddof=0)),
                     "argmin": str(cell.idxmin()), "argmax": str(cell.idxmax())})
    return pd.DataFrame(rows)


def logistic_threshold(deltas, probabilities):
    """Continuation probability at which cooperation overtakes defection.

    Ordinary least squares on the logit of the observed means, which is closed
    form and adequate here because the sweep is dense and the means are well
    inside (0, 1) after clipping.
    """
    x = np.asarray(deltas, dtype=float)
    y = np.clip(np.asarray(probabilities, dtype=float), 1e-3, 1 - 1e-3)
    logit = np.log(y / (1 - y))
    slope, intercept = np.polyfit(x, logit, 1)
    if abs(slope) < 1e-9:
        return float("nan"), float(slope)
    return float(-intercept / slope), float(slope)


def threshold_table(frame: pd.DataFrame, draws: int = 2000,
                    seed: int = RNG_SEED) -> pd.DataFrame:
    """Where JEV switches to cooperation, against where theory says it pays."""
    rng = np.random.default_rng(seed)
    rows = []
    for (payoff_set, opponent), group in frame.groupby(["payoff_set", "opponent"]):
        means = group.groupby("delta")["p_coop_choice"].mean()
        crossing, slope = logistic_threshold(means.index.to_numpy(), means.to_numpy())
        # Resample whole trials within each delta to get an interval.
        samples = []
        by_delta = {d: g["p_coop_choice"].to_numpy() for d, g in group.groupby("delta")}
        for _ in range(draws):
            resampled = [rng.choice(v, size=len(v), replace=True).mean()
                         for v in by_delta.values()]
            value, _ = logistic_threshold(list(by_delta), resampled)
            if np.isfinite(value):
                samples.append(value)
        lo, hi = (np.quantile(samples, [0.025, 0.975]) if samples
                  else (float("nan"), float("nan")))
        rows.append({"payoff_set": payoff_set, "opponent": opponent,
                     "theoretical_threshold": float(group.grim_threshold.iloc[0]),
                     "observed_crossing": crossing, "ci_low": float(lo),
                     "ci_high": float(hi), "slope": slope, "n": len(group)})
    return pd.DataFrame(rows)


def coherence_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Agreement between the three read-outs of the same decision.

    All three come back in a single response, so any disagreement is internal
    to one call rather than run-to-run noise.
    """
    rows = []
    pairs = [("p_coop_choice", "p_coop_noul"), ("p_coop_choice", "p_coop_score"),
             ("p_coop_noul", "p_coop_score")]
    for left, right in pairs:
        sub = frame[[left, right]].dropna()
        both = ((sub[left] > 0.5) == (sub[right] > 0.5)).mean()
        rows.append({"pair": f"{left.replace('p_coop_', '')} vs "
                             f"{right.replace('p_coop_', '')}",
                     "n": len(sub),
                     "pearson_r": float(sub[left].corr(sub[right])),
                     "mean_abs_gap": float((sub[left] - sub[right]).abs().mean()),
                     "same_side_of_half": float(both)})
    return pd.DataFrame(rows)


def reliability(frame: pd.DataFrame, cell_keys) -> pd.DataFrame:
    """Within-cell spread across identical repeated requests."""
    rows = []
    for readout in READOUTS:
        spreads = frame.groupby(cell_keys, dropna=False)[readout].std(ddof=1)
        spreads = spreads[np.isfinite(spreads)]
        point, lo, hi = bootstrap_ci(spreads.to_numpy())
        rows.append({"readout": readout.replace("p_coop_", ""),
                     "cells": int(len(spreads)), "mean_within_cell_sd": point,
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def position_bias(frame: pd.DataFrame, within) -> pd.DataFrame:
    """Signed effect of the presentation counterbalance on P(cooperate).

    Cell keys are filled before grouping: several of them are only defined for
    one block, and pandas silently drops rows with a missing group key, which
    would quietly empty the table.
    """
    work = frame.copy()
    work[within] = work[within].fillna("na").astype(str)
    wide = work.pivot_table(index=within, columns="coop_first",
                            values="p_coop_choice", aggfunc="mean")
    wide = wide.dropna()
    if wide.empty or True not in wide or False not in wide:
        return pd.DataFrame()
    gaps = (wide[True] - wide[False]).to_numpy()
    point, lo, hi = bootstrap_ci(gaps)
    abs_point, abs_lo, abs_hi = bootstrap_ci(np.abs(gaps))
    return pd.DataFrame([{"cells": len(gaps), "mean_signed_gap": point,
                          "ci_low": lo, "ci_high": hi,
                          "mean_absolute_gap": abs_point,
                          "abs_ci_low": abs_lo, "abs_ci_high": abs_hi}])
