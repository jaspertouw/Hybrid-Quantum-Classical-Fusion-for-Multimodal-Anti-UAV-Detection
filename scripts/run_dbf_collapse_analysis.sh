#!/bin/bash
# Run analyse_dbf_naive_collapse.py over all seeds and both splits.
# CPU-only and fast — fine on the Snellius login node, no sbatch needed.
#
# Usage:
#   bash scripts/run_dbf_collapse_analysis.sh            # legacy runs (dogmatic_bg)
#   bash scripts/run_dbf_collapse_analysis.sh vacuous    # corrected runs (vacuous)
#
# Legacy mode diagnoses the original dogmatic-placeholder runs (explains the
# DBF==naive collapse).  Vacuous mode verifies the corrected runs: expect
# C = 0 on one-miss frames, n_diff_* = 0 for every operator pair, and
# decision_mismatch = 0 against the vacuous dbf/naive logs.
#
# Dir conventions:
#   legacy  seed 0  -> <split>_results_detector_only_<mode>_peak  (no suffix)
#   legacy  seed 1+ -> ..._peak_seed${SEED}   (val ir/rgb drop "_peak")
#   vacuous all     -> ..._peak_vacuous_seed${SEED}  (dbf/naive only; ir/rgb
#                      single-modality logs are shared with the legacy runs,
#                      and ABF is provably encoding-invariant)
# theta_cls: 0.5 for the legacy test runs; vacuous naive test used 0.55
#   (recalibrated), so vacuous mode uses 0.55 (inert for dbf either way).

set -e

MODE=${1:-legacy}

REPO=/home/ssuttorp/mm-uav-threat-detection
PRE=/home/ssuttorp/pretrained_models/edtc_fusion
OUTDIR=$REPO/outputs/dbf_collapse
mkdir -p $OUTDIR

if [ "$MODE" = "vacuous" ]; then
  CONVENTION=vacuous
  THETA=0.55
  TAG="vacuous_"
else
  CONVENTION=dogmatic_bg
  THETA=0.5
  TAG=""
fi

cd $REPO

for SPLIT in val test; do
  for SEED in 0 1 2; do
    # Single-modality logs (shared by both modes; per-stream beliefs are
    # encoding-independent).
    if [ "$SEED" -eq 0 ]; then
      IR=$PRE/${SPLIT}_results_detector_only_ir_peak
      RGB=$PRE/${SPLIT}_results_detector_only_rgb_peak
    else
      if [ "$SPLIT" = "val" ]; then
        IR=$PRE/val_results_detector_only_ir_seed${SEED}
        RGB=$PRE/val_results_detector_only_rgb_seed${SEED}
      else
        IR=$PRE/test_results_detector_only_ir_peak_seed${SEED}
        RGB=$PRE/test_results_detector_only_rgb_peak_seed${SEED}
      fi
    fi

    # Dual-mode logs for the cross-checks.
    if [ "$SEED" -eq 0 ]; then SUF=""; else SUF="_seed${SEED}"; fi
    ABF=$PRE/${SPLIT}_results_detector_only_abf_peak${SUF}
    if [ "$MODE" = "vacuous" ]; then
      # Vacuous dirs carry the seed suffix for ALL seeds, incl. 0.
      DBF=$PRE/${SPLIT}_results_detector_only_dbf_peak_vacuous_seed${SEED}
      NAIVE=$PRE/${SPLIT}_results_detector_only_naive_peak_vacuous_seed${SEED}
    else
      DBF=$PRE/${SPLIT}_results_detector_only_dbf_peak${SUF}
      NAIVE=$PRE/${SPLIT}_results_detector_only_naive_peak${SUF}
    fi

    for D in $IR $RGB $ABF $DBF $NAIVE; do
      if [ ! -d "$D" ]; then
        echo "SKIP ${SPLIT} seed${SEED}: missing $D"
        continue 2
      fi
    done

    echo "==================================================================="
    echo "=== ${MODE} ${SPLIT} seed ${SEED}"
    echo "==================================================================="
    python scripts/analyse_dbf_naive_collapse.py \
        --ir_dir    $IR \
        --rgb_dir   $RGB \
        --abf_dir   $ABF \
        --dbf_dir   $DBF \
        --naive_dir $NAIVE \
        --lam 1.0 --theta_cls $THETA \
        --miss_convention $CONVENTION \
        --out $OUTDIR/dbf_collapse_${TAG}${SPLIT}_seed${SEED}.json
    echo ""
  done
done

echo "All summaries in $OUTDIR"
