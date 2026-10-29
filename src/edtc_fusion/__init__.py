"""
EDTC fusion package — Phase 6.1
================================
Integrates Phase 5 belief fusion (DBF) and Dynamic Modality Selection (DMS)
into the EDTC switching logic.  Replaces the original single-modality
uncertainty signal with the fused semantic uncertainty u_fused and the
spatial uncertainty sigma2_best from the lower-uncertainty modality.

Modules
-------
- tracker.py  : EDTCFusionTracker — dual-stream YOLO + SiamCAR tracker with
                four-threshold dual-condition switching.
- run_edtc_fusion.py : sequence runner / evaluator on Anti-UAV RGB-T val.
"""

from .tracker import EDTCFusionTracker
from .tracker_single import EDTCSingleTracker

__all__ = ["EDTCFusionTracker", "EDTCSingleTracker"]
