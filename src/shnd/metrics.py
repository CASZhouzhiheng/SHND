from __future__ import annotations

import math

import numpy as np


DEFAULT_COVERAGES = (10, 30, 50, 70, 75, 80, 85, 90, 95, 100)


def risk_at_coverages(predictions, coverages=DEFAULT_COVERAGES):
    ordered = sorted(predictions, key=lambda row: row["score"], reverse=True)
    if not ordered:
        return {coverage: float("nan") for coverage in coverages}
    risks = {}
    for coverage in coverages:
        if not 0 < coverage <= 100:
            raise ValueError("coverage must lie in (0,100]")
        count = max(1, math.ceil(len(ordered) * coverage / 100.0))
        accepted = ordered[:count]
        risks[coverage] = 100.0 * np.mean([row["label"] != row["prediction"] for row in accepted])
    return {coverage: float(risk) for coverage, risk in risks.items()}


def selection_score(risks, metric: str = "lowcov") -> float:
    if metric == "risk95":
        return float(risks[95])
    if metric == "mean_risk":
        return float(np.mean(list(risks.values())))
    if metric == "lowcov":
        low = np.mean([risks[10], risks[30], risks[50]])
        return float(0.7 * low + 0.3 * np.mean(list(risks.values())))
    raise ValueError(f"unknown selection metric: {metric}")



def coverage_details(predictions, coverages=DEFAULT_COVERAGES):
    risks = risk_at_coverages(predictions, coverages)
    n = len(predictions)
    return [{"coverage_requested_pct": c,
             "coverage_realized_pct": 100*max(1, math.ceil(n*c/100))/n if n else float("nan"),
             "accepted_count": max(1, math.ceil(n*c/100)) if n else 0,
             "total_count": n, "risk_pct": risks[c]} for c in coverages]
