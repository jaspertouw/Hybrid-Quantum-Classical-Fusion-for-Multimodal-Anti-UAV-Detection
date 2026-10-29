"""
rescore_attribute_metrics.py — Attribute-based robustness table for a finished run
==================================================================================
Post-hoc per-attribute evaluation on a run's already-saved per-frame logs,
mirroring rescore_belief_metrics.py but pooling frames PER ATTRIBUTE subset.

Anti-UAV-RGBT assigns each sequence one or more condition attributes in
``label_new/<split>.json``:

    TC  thermal crossover      FM  fast motion         SV  scale variation
    LI  low illumination       LR  low resolution      OV  out of view
    OC  occlusion              TC-EASY / TC-MID / TC-HARD  difficulty tiers

Attributes are SEQUENCE-level and MULTI-label: subsets overlap (the same
frames appear under every attribute their sequence carries), so rows are
slices of the same data, not a partition.

Per attribute we report:
    n_seq, n_frames        subset size (judge stability before trusting a row)
    Acc                    EDTC paper Acc: per-sequence
                           mean(IoU on present, empty-pred on absent)
                           - 0.2*(failures/T)^0.3, averaged over the subset's
                           sequences (matches summary.json's 'acc')
    mIoU                   mean IoU on present frames
    ECE                    calibration of conf_prob vs correctness
    AUROC                  correct-vs-incorrect ranking by conf_prob
    AUSE                   sparsification of IoU by conf_prob (belief frames)
    mean u (all/present)   does vacuity inflate under this condition?
    Spearman rho(u, IoU)   is u still informative? (present belief frames;
                           negative = high uncertainty <-> low IoU, i.e. good)

All confidence/correctness conventions are identical to
rescore_belief_metrics.py (same _per_sequence_arrays).

Usage
-----
Single run:

    python -m src.edtc_fusion.rescore_attribute_metrics \\
        --results_dir  .../test_results_detector_only_dbf_peak \\
        --dataset_root /home/ssuttorp/Anti-UAV-RGBT/test

Multi-seed (pass every seed's results dir of the SAME variant + split; the
no-suffix dir is seed 0, per print_metrics_tables.py convention):

    python -m src.edtc_fusion.rescore_attribute_metrics \\
        --results_dir .../test_results_detector_only_dbf_peak \\
                      .../test_results_detector_only_dbf_peak_seed1 \\
                      .../test_results_detector_only_dbf_peak_seed2 \\
        --dataset_root /home/ssuttorp/Anti-UAV-RGBT/test

Each run gets its own ``summary_attribute_metrics.json``; with >1 dirs a
mean +/- std aggregate over seeds is also written to the FIRST dir as
``summary_attribute_metrics_seedavg.json`` (override with --out).

The attribute file defaults to ``<dataset_root>/../label_new/<split>.json``
(override with --attributes_json).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EDTC_ROOT = _REPO_ROOT / "resources" / "EDTC"
for _p in (str(_REPO_ROOT), str(_EDTC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.utils.eval_metrics import (  # noqa: E402
    compute_ece,
    compute_auroc,
    compute_sparsification,
)

from src.edtc_fusion.run_edtc_fusion import (  # noqa: E402
    load_paired_sequence,
    discover_sequences,
)
from src.edtc_fusion.rescore_belief_metrics import (  # noqa: E402
    _load_preds,
    _load_log,
    _detect_run_type,
    _per_sequence_arrays,
)

# Canonical reporting order (difficulty tiers last).
ATTRIBUTE_ORDER = ["TC", "FM", "SV", "LI", "LR", "OV", "OC",
                   "TC-EASY", "TC-MID", "TC-HARD"]

ATTRIBUTE_NAMES = {
    "TC": "thermal crossover",   "FM": "fast motion",
    "SV": "scale variation",     "LI": "low illumination",
    "LR": "low resolution",      "OV": "out of view",
    "OC": "occlusion",
    "TC-EASY": "TC easy tier",   "TC-MID": "TC mid tier",
    "TC-HARD": "TC hard tier",
}

# Numeric metrics aggregated across seeds (counts are checked, not averaged).
AGG_METRICS = ["acc_edtc", "accuracy_threshold", "mean_iou_present",
               "ece_conf_prob", "auroc_conf_prob", "ause_conf_prob",
               "ause_conf_sigma", "mean_u_all", "mean_u_present",
               "spearman_u_iou"]


# ---------------------------------------------------------------------------
# Spearman rank correlation (no scipy dependency)
# ---------------------------------------------------------------------------

def _rankdata(x):
    """Average ranks (ties share the mean rank), as in scipy.stats.rankdata."""
    x = np.asarray(x, dtype=float)
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(x))
    sorted_x = x[sorter]
    obs = np.r_[True, sorted_x[1:] != sorted_x[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _spearman(a, b):
    """Spearman rho between two 1-D arrays; NaN if degenerate."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or len(a) != len(b):
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:        # constant -> undefined
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# Pooled metrics over a list of per-sequence array dicts
# ---------------------------------------------------------------------------

def _nan_to_none(v):
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)


def _pool_metrics(seq_arrays):
    """Compute the full metric row over the pooled frames of ``seq_arrays``."""
    cat = lambda key, dt=float: (np.concatenate([a[key] for a in seq_arrays])
                                 if seq_arrays else np.array([], dtype=dt))

    correctness = cat("correctness", int)
    cp_all      = cat("conf_prob")
    cm_all      = cat("conf_1mu")
    ious_pres   = cat("ious_present")
    ious_belief = cat("ious_belief")
    cp_pres     = cat("cp_present")
    cm_pres     = cat("cm_present")
    cs_pres     = cat("cs_present")
    present     = cat("present_flags", bool)

    # EDTC paper Acc: per-sequence reduction (the 0.2*(failures/T)^0.3 penalty
    # is nonlinear, so it must NOT be pooled), then unweighted mean over the
    # subset's sequences — matching evaluate_antiuav_performance.py and the
    # 'acc' field the runners write to summary.json.
    sa_vals = [a["state_accuracy"] for a in seq_arrays
               if not np.isnan(a["state_accuracy"])]
    acc_edtc = float(np.mean(sa_vals)) if sa_vals else float("nan")

    has_belief = len(correctness) > 0

    def _try(fn, *args):
        try:
            return fn(*args)
        except Exception:
            return float("nan")

    ece   = _try(compute_ece,   cp_all, correctness) if has_belief else float("nan")
    auroc = _try(compute_auroc, cp_all, correctness) if has_belief else float("nan")
    ause = float("nan")
    if len(ious_belief) > 1:
        ause = _try(lambda i, c: compute_sparsification(i, c)["ause"],
                    ious_belief, cp_pres)
    ause_sigma = float("nan")
    if len(ious_pres) > 1:
        ause_sigma = _try(lambda i, c: compute_sparsification(i, c)["ause"],
                          ious_pres, cs_pres)

    # u = 1 - conf_1mu  (conf_1mu is logged for every scored belief frame).
    mean_u_all  = float(np.mean(1.0 - cm_all))  if len(cm_all)  else float("nan")
    mean_u_pres = float(np.mean(1.0 - cm_pres)) if len(cm_pres) else float("nan")
    # Spearman between per-frame u and IoU on present belief frames.
    # cm_pres = 1 - u, so rho(u, IoU) = -rho(1-u, IoU).  Negative = good
    # (higher uncertainty <-> lower IoU).
    spearman_u_iou = -_spearman(cm_pres, ious_belief)

    return {
        "n_seq":            len(seq_arrays),
        "n_frames_scored":  int(len(correctness)),
        "n_frames_present": int(len(ious_pres)),
        "n_present_gt":     int(np.sum(present)) if len(present) else 0,
        "n_absent_gt":      int(np.sum(~present)) if len(present) else 0,
        "acc_edtc":         _nan_to_none(acc_edtc),
        "accuracy_threshold": _nan_to_none(correctness.mean() if has_belief else float("nan")),
        "mean_iou_present": _nan_to_none(ious_pres.mean() if len(ious_pres) else float("nan")),
        "ece_conf_prob":    _nan_to_none(ece),
        "auroc_conf_prob":  _nan_to_none(auroc),
        "ause_conf_prob":   _nan_to_none(ause),
        "ause_conf_sigma":  _nan_to_none(ause_sigma),
        "mean_u_all":       _nan_to_none(mean_u_all),
        "mean_u_present":   _nan_to_none(mean_u_pres),
        "spearman_u_iou":   _nan_to_none(spearman_u_iou),
    }


# ---------------------------------------------------------------------------
# Score one results dir (one seed run)
# ---------------------------------------------------------------------------

def _score_run(results_dir, dataset_root, seq_attrs, args, verbose):
    """Per-attribute rows for one run.  Returns (run_type, rows, attrs_seen)
    and writes ``summary_attribute_metrics.json`` into ``results_dir``."""
    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.isfile(summary_path):
        raise SystemExit(f"missing {summary_path}")
    with open(summary_path) as f:
        run_type = _detect_run_type(json.load(f))

    sequences = discover_sequences(dataset_root)

    per_seq = {}
    missing_attr = []
    for v_idx, seq in enumerate(sequences):
        bbox_path = os.path.join(results_dir, seq + ".txt")
        log_path  = os.path.join(results_dir, seq + "_log.csv")
        seq_dir   = os.path.join(dataset_root, seq)
        if not (os.path.isfile(bbox_path) and os.path.isfile(log_path)):
            continue
        frames = load_paired_sequence(seq_dir)
        if not frames:
            continue
        a = _per_sequence_arrays(_load_preds(bbox_path), _load_log(log_path),
                                 frames, run_type,
                                 args.iou_threshold, args.cls_threshold)
        if len(a["correctness"]) == 0 and len(a["ious_present"]) == 0:
            continue
        per_seq[seq] = a
        if seq not in seq_attrs:
            missing_attr.append(seq)
        if verbose:
            print(f"[{v_idx + 1:3d}/{len(sequences)}] {seq:<40s} "
                  f"attrs={','.join(seq_attrs.get(seq, [])) or '-'}")

    if not per_seq:
        raise SystemExit(f"No sequences with usable data in {results_dir}")
    if missing_attr:
        print(f"WARNING: {len(missing_attr)} scored sequence(s) in "
              f"{os.path.basename(results_dir)} have no attribute entry: "
              f"{missing_attr}")

    attrs_seen = sorted({a for s in per_seq for a in seq_attrs.get(s, [])},
                        key=lambda a: (ATTRIBUTE_ORDER.index(a)
                                       if a in ATTRIBUTE_ORDER else 99, a))

    rows = {"ALL": _pool_metrics(list(per_seq.values()))}
    for attr in attrs_seen:
        members = [per_seq[s] for s in per_seq if attr in seq_attrs.get(s, [])]
        rows[attr] = _pool_metrics(members)
        rows[attr]["unstable"] = rows[attr]["n_seq"] < args.min_seq

    out = {
        "run_type":        run_type,
        "results_dir":     results_dir,
        "dataset_root":    dataset_root,
        "iou_threshold":   args.iou_threshold,
        "cls_threshold":   args.cls_threshold,
        "min_seq":         args.min_seq,
        "note": ("Attributes are sequence-level and multi-label: subsets "
                 "overlap and do not partition the split.  spearman_u_iou is "
                 "rho(u_fused, IoU) on present belief frames — negative means "
                 "high uncertainty tracks low IoU (informative).  Rows with "
                 "unstable=true have fewer than min_seq sequences."),
        "attribute_names": {a: ATTRIBUTE_NAMES.get(a, a) for a in attrs_seen},
        "per_attribute":   rows,
    }
    out_path = os.path.join(results_dir, "summary_attribute_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Per-run summary -> {out_path}")

    return run_type, rows, attrs_seen


# ---------------------------------------------------------------------------
# Console tables
# ---------------------------------------------------------------------------

_TABLE_COLS = [("acc_edtc", "Acc"), ("mean_iou_present", "mIoU"),
               ("ece_conf_prob", "ECE"), ("auroc_conf_prob", "AUROC"),
               ("ause_conf_prob", "AUSE"), ("mean_u_all", "u_all"),
               ("mean_u_present", "u_pres"), ("spearman_u_iou", "rho(u,IoU)")]


def _print_single_table(title, rows, order, min_seq):
    w_head, w_num = 8, 11
    print()
    print(f"== {title} ==")
    hdr = ["attr".rjust(w_head), "nseq".rjust(5), "frames".rjust(7)]
    hdr += [h.rjust(w_num) for _, h in _TABLE_COLS]
    print("".join(hdr))
    print("-" * (w_head + 12 + w_num * len(_TABLE_COLS)))
    for name in order:
        r = rows[name]
        cells = [name.rjust(w_head), f"{r['n_seq']:5d}",
                 f"{r['n_frames_scored']:7d}"]
        for key, _ in _TABLE_COLS:
            v = r.get(key)
            cells.append(("-" if v is None else f"{v:.4f}").rjust(w_num))
        print("".join(cells) + ("  *" if r.get("unstable") else ""))
    print(f"* fewer than {min_seq} sequences — treat as unstable")


def _print_agg_table(title, agg, order, n_runs, min_seq):
    w_head, w_num = 8, 16
    print()
    print(f"== {title}  (mean +/- std over {n_runs} seeds) ==")
    hdr = ["attr".rjust(w_head), "nseq".rjust(5), "frames".rjust(8)]
    hdr += [h.rjust(w_num) for _, h in _TABLE_COLS]
    print("".join(hdr))
    print("-" * (w_head + 13 + w_num * len(_TABLE_COLS)))
    for name in order:
        r = agg[name]
        cells = [name.rjust(w_head), f"{r['n_seq']:5d}",
                 f"{r['n_frames_scored']:8.0f}"]
        for key, _ in _TABLE_COLS:
            m, s = r.get(key + "_mean"), r.get(key + "_std")
            cell = "-" if m is None else (f"{m:.4f}" if s is None
                                          else f"{m:.4f}+-{s:.4f}")
            cells.append(cell.rjust(w_num))
        print("".join(cells) + ("  *" if r.get("unstable") else ""))
    print(f"* fewer than {min_seq} sequences — treat as unstable")


# ---------------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------------

def aggregate_seed_rows(all_rows):
    """Mean +/- sample std (ddof=1) over per-seed pooled rows.

    ``all_rows`` is a list of per-run ``rows`` dicts from _score_run.
    Returns (order, agg) where order is ["ALL", <attrs...>] and agg maps each
    name to {<metric>_mean, <metric>_std, <metric>_n, n_seq, ...}.
    """
    attrs_union = []
    for rows in all_rows:
        for a in rows:
            if a != "ALL" and a not in attrs_union:
                attrs_union.append(a)
    order = ["ALL"] + sorted(attrs_union,
                             key=lambda a: (ATTRIBUTE_ORDER.index(a)
                                            if a in ATTRIBUTE_ORDER else 99, a))
    agg = {}
    for name in order:
        rs = [rows[name] for rows in all_rows if name in rows]
        entry = {
            "n_runs":           len(rs),
            "n_seq":            rs[0]["n_seq"],
            "n_frames_scored":  float(np.mean([r["n_frames_scored"] for r in rs])),
            "unstable":         any(r.get("unstable") for r in rs),
        }
        if len({r["n_seq"] for r in rs}) > 1:
            print(f"WARNING: n_seq differs across seeds for '{name}': "
                  f"{[r['n_seq'] for r in rs]}")
        for key in AGG_METRICS:
            vals = [r[key] for r in rs if r.get(key) is not None]
            if vals:
                entry[key + "_mean"] = float(np.mean(vals))
                # Sample std over seeds (ddof=1); None when only one value.
                entry[key + "_std"] = (float(np.std(vals, ddof=1))
                                       if len(vals) > 1 else None)
                entry[key + "_n"] = len(vals)
            else:
                entry[key + "_mean"] = None
                entry[key + "_std"] = None
                entry[key + "_n"] = 0
        agg[name] = entry
    return order, agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Per-attribute robustness metrics for finished run(s).")
    p.add_argument("--results_dir", required=True, nargs="+",
                   help="One or more results dirs (one per seed) of the SAME "
                        "variant and split.  >1 dirs -> per-seed scoring plus "
                        "a mean+/-std aggregate.")
    p.add_argument("--dataset_root", required=True,
                   help="Anti-UAV-RGBT split root (e.g. .../val or .../test)")
    p.add_argument("--attributes_json", default=None,
                   help="Sequence->attributes JSON.  Default: "
                        "<dataset_root>/../label_new/<split>.json")
    p.add_argument("--out", default=None,
                   help="Path for the multi-seed aggregate JSON (default: "
                        "<first results_dir>/summary_attribute_metrics_seedavg.json)")
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--cls_threshold", type=float, default=0.5)
    p.add_argument("--min_seq", type=int, default=3,
                   help="Flag attribute rows with fewer sequences than this "
                        "as unstable (still reported).")
    return p.parse_args()


def main():
    args = parse_args()

    attr_path = args.attributes_json
    if attr_path is None:
        root = Path(args.dataset_root)
        attr_path = str(root.parent / "label_new" / (root.name + ".json"))
    if not os.path.isfile(attr_path):
        raise SystemExit(f"attribute file not found: {attr_path} "
                         f"(pass --attributes_json explicitly)")
    with open(attr_path) as f:
        seq_attrs = json.load(f)

    multi = len(args.results_dir) > 1
    print(f"Dataset root    : {args.dataset_root}")
    print(f"Attributes json : {attr_path}")
    print(f"Runs            : {len(args.results_dir)}")
    print()

    run_types, all_rows, attrs_union = [], [], []
    for rd in args.results_dir:
        print(f"--- scoring {rd}")
        run_type, rows, attrs_seen = _score_run(
            rd, args.dataset_root, seq_attrs, args, verbose=not multi)
        run_types.append(run_type)
        all_rows.append(rows)
        for a in attrs_seen:
            if a not in attrs_union:
                attrs_union.append(a)
        _print_single_table(f"{os.path.basename(rd)}  ({run_type})",
                            rows, ["ALL"] + attrs_seen, args.min_seq)
        print()

    if len(set(run_types)) > 1:
        print(f"WARNING: mixed run types across dirs: {run_types} — the "
              f"aggregate averages different variants, which is probably "
              f"not what you want.")

    if not multi:
        return

    # ── Mean +/- std over seeds ────────────────────────────────────────────
    order, agg = aggregate_seed_rows(all_rows)

    out_path = args.out or os.path.join(
        args.results_dir[0], "summary_attribute_metrics_seedavg.json")
    out = {
        "run_type":        run_types[0],
        "results_dirs":    list(args.results_dir),
        "dataset_root":    args.dataset_root,
        "iou_threshold":   args.iou_threshold,
        "cls_threshold":   args.cls_threshold,
        "min_seq":         args.min_seq,
        "n_runs":          len(args.results_dir),
        "note": ("mean/std are computed over per-seed POOLED metrics (each "
                 "seed's frames pooled per attribute first, then averaged "
                 "across seeds; std is sample std, ddof=1).  Subsets overlap "
                 "(multi-label sequence attributes).  spearman_u_iou: "
                 "negative = high uncertainty tracks low IoU (informative)."),
        "attribute_names": {a: ATTRIBUTE_NAMES.get(a, a) for a in attrs_union},
        "per_attribute":   agg,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    _print_agg_table(f"{run_types[0]}", agg, order,
                     len(args.results_dir), args.min_seq)
    print(f"\nAggregate summary -> {out_path}")


if __name__ == "__main__":
    main()
