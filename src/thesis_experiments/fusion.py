"""Decision-level references operating on the same frozen detector opinions."""
from __future__ import annotations

import numpy as np


def sensor_opinions(rows, sensor):
    belief, uncertainty = [], []
    for row in rows:
        value = row[sensor]["candidate"]
        belief.append([0., 0.] if value is None else value["beliefs"])
        uncertainty.append(1. if value is None else value["semantic_uncertainty"])
    return np.array(belief, dtype=np.float64), np.array(uncertainty, dtype=np.float64)


def abf(br, ur, bt, ut):
    denom = ur + ut
    small = denom < 1e-8
    b = (br * ut[:, None] + bt * ur[:, None]) / np.maximum(denom[:, None], 1e-8)
    u = 2 * ur * ut / np.maximum(denom, 1e-8)
    b[small], u[small] = (br[small] + bt[small]) / 2, 0
    return b, u


def predict_reference(rows, method, *, prevalence=None, exponent=1.):
    br, ur = sensor_opinions(rows, "rgb")
    bt, ut = sensor_opinions(rows, "thermal")
    if method in ("rgb", "thermal"):
        b, u = (br, ur) if method == "rgb" else (bt, ut)
    elif method == "naive":
        b = (br + bt) / 2
        u = np.clip(1 - b.sum(axis=1), 0, 1)
    elif method == "abf":
        b, u = abf(br, ur, bt, ut)
    elif method == "dbf":
        if exponent <= 0:
            raise ValueError("DBF exponent must be positive")
        c = np.clip(br.sum(axis=1) * bt.sum(axis=1) - (br * bt).sum(axis=1), 0, 1)
        eta = np.maximum(1 - c ** exponent, 0) ** (1 / exponent)
        br, bt = br * eta[:, None], bt * eta[:, None]
        b, u = abf(br, np.clip(1 - br.sum(axis=1), 0, 1), bt, np.clip(1 - bt.sum(axis=1), 0, 1))
    elif method in ("always_present", "prevalence"):
        if method == "prevalence" and (prevalence is None or not 0 <= prevalence <= 1):
            raise ValueError("training prevalence required")
        return np.full(len(rows), 1. if method == "always_present" else prevalence), None
    else:
        raise ValueError(f"unknown fusion reference: {method}")
    return np.clip(b[:, 0] + u / 2, 0, 1), u


def disagreement(rows):
    br, ur = sensor_opinions(rows, "rgb")
    bt, ut = sensor_opinions(rows, "thermal")
    pr, pt = br + ur[:, None] / 2, bt + ut[:, None] / 2
    return {"belief_difference": np.abs(br[:, 0] - bt[:, 0]),
            "degree_of_conflict": np.abs(pr - pt).sum(axis=1) / 2 * (1 - ur) * (1 - ut),
            "inherited_conflict": np.clip(br.sum(axis=1) * bt.sum(axis=1) - (br * bt).sum(axis=1), 0, 1),
            "rgb_missing": np.array([r["rgb"]["candidate"] is None for r in rows]),
            "thermal_missing": np.array([r["thermal"]["candidate"] is None for r in rows])}
