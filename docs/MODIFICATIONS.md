# EDTC Codebase Modifications — Living Changelog

Tracks every change made to the original EDTC codebase for the thesis:
**"Uncertainty-Aware Multimodal Anti-UAV Detection and Tracking via Evidential Fusion and Conflict-Discounted Belief Aggregation"**

Each section documents one file with: what changed, why it changed, and implementation status.
Use this as a reference when writing the methods and implementation sections of the paper.

---

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done

---

## Existing Files — Modified

### `resources/EDTC/lib/models/uavtrack/head.py`
**Original role:** Bounding box regression heads (Corner_Predictor, Pyramid_Corner_Predictor, MLP)

- [ ] Add `GaussianCornerPredictor` class (after line 258)
  - **Why:** The original `Corner_Predictor` outputs deterministic coordinates (b,4). To model spatial uncertainty per modality, each coordinate must be represented as a Gaussian distribution with a mean and variance. This enables the normalized spatial variance signal used in Dynamic Modality Selection (§3.1.4 of thesis).
  - **What:** Parallel variance branches (`conv1_var_tl … conv5_var_tl`, same depth as mean branches) that output `log_vars (b,4)`. Mean branches unchanged from `Corner_Predictor`.
  - **Output:** `coords (b,4)`, `log_vars (b,4)` — log-variance per coordinate (unconstrained float).

- [ ] Add `HEAD_TYPE = "CORNER_GAUSSIAN"` branch in `build_box_head()`
  - **Why:** Config-driven head selection allows easy ablation between deterministic and Gaussian heads.

---

### `resources/EDTC/lib/utils/box_ops.py`
**Original role:** Loss functions — `ciou_loss`, `edl_mse_loss`, `kl_divergence`, etc.

- [ ] Add `gaussian_nll_loss(pred_mean, pred_log_var, target)`
  - **Why:** CIoU loss is incompatible with Gaussian output heads. The negative log-likelihood of a Gaussian (`0.5 * (log_var + (target - mean)^2 / exp(log_var))`) jointly optimizes localization accuracy and variance calibration, following He et al. [10].
  - **Input shapes:** `pred_mean (N,4)`, `pred_log_var (N,4)`, `target (N,4)` — all in xyxy format.
  - **Note:** Applied with a mask — only on frames where a UAV is present (`label == 1`), since bbox regression is undefined for absent targets.

---

### `resources/EDTC/lib/train/run_training.py`
**Original role:** Training entry point — accepts `uavtrack` or `uavtrack_eh` as script names.

- [ ] Add `elif settings.script_name == "uavtrack_mm": net = build_uavtrack_mm(cfg, settings)` branch
  - **Why:** Registers the new multimodal model with the existing training harness without touching the existing branches (preserves reproducibility of baselines).

---

## New Files — Created

### `resources/EDTC/lib/models/uavtrack/fusion.py`
**Role:** Discounted Belief Fusion (DBF) module — pure math, no learnable parameters.

- [ ] `class DiscountedBeliefFusion(nn.Module)`
  - **Why:** EDTC's original uncertainty is a single-modality evidential score. To fuse two independent Dirichlet opinions under the assumption of dependent sources (same scene, simultaneous observation), DBF is used instead of Dempster's rule (which assumes independence) or plain ABF (which underestimates uncertainty in high-conflict scenarios). Follows Bezirganyan et al. [3].
  - **Input:** `alpha_ir (b,2)`, `alpha_rgb (b,2)` — Dirichlet concentration parameters per modality.
  - **Output:** `b_fused (b,2)`, `u_fused (b,1)` (∈[0,1]), `alpha_fused (b,2)`.
  - **Steps:** alpha → beliefs/uncertainty → conflict C → discount η = (1-C^λ)^(1/λ) → discounted opinions → ABF → fused opinion.
  - **Hyperparameter:** `lambda_` (default 2.0), tunable via `cfg.MODEL.DBF_LAMBDA`.

---

### `resources/EDTC/lib/models/uavtrack/uavtrack_mm.py`
**Role:** Dual-stream model — `UAVTrackMM` class and `build_uavtrack_mm()` builder.

- [ ] `class UAVTrackMM(nn.Module)`
  - **Why:** The original `UAVTrackEH` uses a single CvT backbone and single set of heads. The dual-stream design keeps modality-specific processing independent to prevent a degraded modality from corrupting the reliable one (§3.1.1 of thesis).
  - **Structure:** `backbone_ir`, `backbone_rgb` (separate CvT instances initially), per-modality `GaussianCornerPredictor` heads and `ScoreDecoder` branches, plus `DiscountedBeliefFusion` module.
  - **`forward()` output dict keys:** `pred_boxes_ir`, `pred_log_var_ir`, `pred_scores_ir`, `pred_boxes_rgb`, `pred_log_var_rgb`, `pred_scores_rgb`, `b_fused`, `u_fused`, `alpha_fused`.

- [ ] `build_uavtrack_mm(cfg, settings=None, train=True)`
  - **Why:** Mirrors `build_uavtrack_eh()` API for compatibility with training harness.

---

### `resources/EDTC/lib/train/dataset/antiuav_mm.py`
**Role:** Paired IR+RGB training dataset class.

- [ ] Paired dataset class returning `_ir`/`_rgb` suffixed keys per sample
  - **Why:** The standard `antiuav.py` dataset returns only IR frames. Training the dual-stream model requires synchronized IR+RGB pairs with independent bounding box annotations.
  - **Based on:** `antiuav.py` — extends to load from both `ir_frames/` and `rgb_frames/` subdirectories with `IR_label.json` and `visible_label.json` respectively.
  - **Output keys:** `template_images_ir`, `template_images_rgb`, `search_images_ir`, `search_images_rgb`, `search_anno_ir`, `search_anno_rgb`, `label_ir`, `label_rgb`.

---

### `resources/EDTC/lib/train/actors/uavtrack_mm.py`
**Role:** Training actor for the dual-stream model — `UAVTrackMMActor`.

- [ ] `class UAVTrackMMActor` forward pass + loss computation
  - **Why:** The existing `UAVTrackActor` handles a single stream. The multimodal actor must compute: per-modality EDL classification losses, per-modality Gaussian NLL bbox losses (masked), and the fused EDL classification loss on `alpha_fused`.
  - **Losses:**
    - `edl_mse_loss(pred_scores_ir, labels_ir, ...)` — IR evidential classification
    - `edl_mse_loss(pred_scores_rgb, labels_rgb, ...)` — RGB evidential classification
    - `edl_mse_loss(alpha_fused - 1, labels_fused, ...)` — fused evidential classification
    - `gaussian_nll_loss(pred_boxes_ir[mask], pred_log_var_ir[mask], gt_ir[mask])` — IR spatial
    - `gaussian_nll_loss(pred_boxes_rgb[mask], pred_log_var_rgb[mask], gt_rgb[mask])` — RGB spatial

---

### `resources/EDTC/lib/train/train_script_mm.py`
**Role:** Training orchestration script for the multimodal model.

- [ ] Adapted from `train_script.py` for multimodal actor, losses, and dataset
  - **Why:** Keeps baseline training scripts untouched (reproducibility). Multimodal-specific instantiation (model, actor, dataset, loss dict) lives here.

---

### `resources/EDTC/lib/config/uavtrack_mm/config.py`
**Role:** Config file for dual-stream training run.

- [ ] Config extending `uavtrack_eh` baseline with multimodal additions
  - **New fields:**
    - `cfg.MODEL.DBF_LAMBDA = 2.0` — DBF conflict discounting hyperparameter
    - `cfg.MODEL.HEAD_TYPE = "CORNER_GAUSSIAN"` — enables Gaussian bbox heads
    - `cfg.TRAIN.FUSED_SCORE_WEIGHT = 1.0` — loss weight for fused EDL loss
    - `cfg.DATA.TRAIN.DATASETS_NAME = ["ANTIUAV_RGBT"]` — paired dataset

---

### `resources/EDTC/lib/test/tracker/uavtrack_mm.py`
**Role:** Multimodal inference tracker — `MultimodalTracker` class.

- [ ] `initialize_yolo()` — loads two YOLO models (`yolo_ir`, `yolo_rgb`)
  - **Why:** Each modality needs its own detector instance since IR and RGB frames have different resolution and appearance characteristics.

- [ ] `initialize(image_ir, image_rgb, yolo_ir, yolo_rgb)` — dual detection
  - **Why:** Detection initializes the local tracker. Dual-condition gate (`u_fused < θ_cls_detect` AND `σ̂²_best < θ_loc_detect`) ensures the tracker is only initialized when both semantic confidence and spatial precision are sufficient.

- [ ] `track(image_ir, image_rgb)` — dual-stream tracking + dual-condition switch
  - **Why:** Replaces the single-condition switch at `uavtrack_eh.py:230`. The original condition (`preds==0 or uncertainty > threshold`) uses only semantic uncertainty from a single stream. The upgraded condition adds a spatial confidence gate per modality and fuses semantic uncertainty via DBF before comparing against state-dependent thresholds.
  - **Switching condition (replaces lines 219–232 of `uavtrack_eh.py`):**
    ```python
    # u_fused in [0,1] from DBF; sigma2_best = min(normalized spatial var per modality)
    if preds == 0 or u_fused > theta_cls_track or sigma2_best > theta_loc_track:
        → redetect
    ```
  - **Dynamic bbox selection:** Output bounding box from modality with lower normalized spatial variance `σ̂² = exp(log_var) / (pred_w * pred_h)`.

---

### `resources/EDTC/lib/test/parameter/uavtrack_mm.py`
**Role:** Parameter file for inference — four switching thresholds.

- [ ] Four-threshold parameter config
  - **Why:** The original single threshold (`evidential_threshold = 0.2`) is replaced by state-dependent thresholds: detection→tracking requires stricter confidence than maintaining tracking. Spatial and semantic conditions are decoupled because YOLO and the tracker optimize different objectives and operate in different feature spaces (§3.1.5 of thesis).
  - **Params:**
    - `theta_cls_detect = 0.6` — max allowed `u_fused` to start tracking (strict: want confident detection)
    - `theta_cls_track = 0.4` — max allowed `u_fused` to maintain tracking (looser)
    - `theta_loc_detect = 0.1` — max allowed normalized `σ̂²` to start tracking
    - `theta_loc_track = 0.15` — max allowed normalized `σ̂²` to maintain tracking
  - **Calibration:** Grid-search on validation set; see §3.1.5 Threshold Calibration Strategy.

---

---

## `tracking/` — CLI Entry Points

> The `tracking/` folder is the user-facing CLI layer. Each file is a thin wrapper over `lib/`. None of the model logic lives here, but several scripts need new arguments or multimodal-aware versions.

### `tracking/test.py`
**Original role:** CLI entry point for inference — parses args, instantiates `Tracker`, calls `run_dataset()`.

- [ ] Add `--tracker_name uavtrack_mm` as a recognized option (currently defaults to `uavtrack_eh`)
  - **Why:** `Tracker` dynamically imports `lib/test/tracker/<tracker_name>.py`, so the new tracker is picked up automatically once the file exists — but the argparse default and `--params__` args need updating.
- [ ] Replace `--params__evidential_threshold` with four new threshold args:
  ```
  --params__theta_cls_detect    (default 0.6)
  --params__theta_cls_track     (default 0.4)
  --params__theta_loc_detect    (default 0.1)
  --params__theta_loc_track     (default 0.15)
  ```
  - **Why:** The single threshold is replaced by the dual-condition switching logic (§3.1.5). Old arg kept for backward compatibility with `uavtrack_eh`.
- [ ] Add `--params__model_ir` and `--params__model_rgb` args (paths to IR and RGB YOLO checkpoints)
  - **Why:** Dual YOLO requires separate checkpoint paths.

---

### `tracking/train.py`
**Original role:** CLI wrapper — builds and runs the `python lib/train/run_training.py ...` shell command.

- [ ] No logic change needed — `--script uavtrack_mm` passes straight through to `run_training.py`.
  - **Why:** The script just forwards `--script` as a string; `run_training.py` handles the branching.
- [ ] Add note in comments that `uavtrack_mm` is a valid `--script` value.

---

### `tracking/evaluate_antiuav_performance.py`
**Original role:** Standalone eval script — reads saved `.txt` bbox results, computes EDTC "Acc" metric against `IR_label.json`. **Hardcoded to Snellius paths.**

- [ ] Parametrize dataset root and results root (replace hardcoded paths with `argparse` arguments)
  - **Why:** Current paths are hardcoded to `/home/ssuttorp/...`; needs to work both locally and on Snellius.
- [ ] Add support for evaluating multimodal results — read `visible_label.json` in addition to `IR_label.json`
  - **Why:** The dual-stream system outputs bounding boxes in IR coordinate space (from the IR modality) or RGB coordinate space (from Dynamic Modality Selection). Evaluation against the correct ground truth per output modality is needed for fair comparison.
- [ ] Add `--modality` flag (`ir` / `rgb` / `fused`) to select which ground truth to evaluate against
  - **Why:** Allows separate evaluation of IR-only, RGB-only, and fused outputs against their respective annotations.

---

### `tracking/profile_model.py`
**Original role:** FLOPs and inference speed profiling for the single-stream tracker.

- [ ] Update to profile `UAVTrackMM` with dual inputs (IR + RGB search patches)
  - **Why:** Computational efficiency is a secondary evaluation criterion (§3.3.4). The dual-stream model processes two search patches per frame; FLOPs must be measured end-to-end including both streams and DBF.
  - **Target:** ≥ 25 FPS on A100 (Snellius).

---

### `tracking/video_demo.py`
**Original role:** Visual demo — runs tracker on a video file and saves output with drawn bboxes.

- [ ] Update to accept paired IR+RGB video inputs and visualize: IR bbox, RGB bbox, selected bbox, `u_fused`
  - **Why:** Useful for qualitative evaluation and thesis figures. Low priority — update after inference is working.

---

### YOLOv5 Fine-tuning on RGB (future step)

> **Not yet started.** The current EDTC YOLO checkpoint is trained on thermal IR only. For the RGB detection branch, the YOLOv5 model needs fine-tuning on RGB frames from the Anti-UAV dataset.

- [ ] Fine-tune `yolov5/` on Anti-UAV RGB train split
  - **Entry point:** `yolov5/train.py` — completely independent from `lib/train/`
  - **Data:** RGB frames at `/home/ssuttorp/Anti-UAV-RGBT/train/<seq>/rgb_frames/`
  - **Why:** The pretrained YOLO was trained on thermal IR appearance. RGB has different contrast, color, and texture characteristics — a fine-tuned RGB YOLO is needed for reliable detection in the RGB branch.
  - **How:** Prepare a YOLO-format dataset from `visible_label.json` annotations; fine-tune from the EDTC IR YOLO checkpoint or a COCO-pretrained YOLOv5.

---

## Already Done (pre-existing)

- [x] `resources/EDTC/lib/test/evaluation/antiuavrgbtdataset.py` — `AntiUAVRGBTDataset` + `PairedSequence`
- [x] `resources/EDTC/lib/test/evaluation/datasets.py` — registered `antiuavrgbt` key
- [x] `resources/EDTC/lib/test/evaluation/antiuavdataset.py` — updated to read from `ir_frames/`
- [x] `scripts/preprocess_antiuav.py` — extracts IR+RGB frames into per-sequence subdirectories
