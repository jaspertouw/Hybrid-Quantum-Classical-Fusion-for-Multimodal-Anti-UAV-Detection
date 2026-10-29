# Quantum and Classical Decision-Level Fusion for Multimodal Anti-UAV Detection

MSc thesis research building on the multimodal evidential-fusion work of
Suttorp et al. The research studies whether variational quantum representations improve
RGB--thermal decision-level fusion under clean, degraded, and missing-sensor conditions.

This repository is a successor to Suttorp's
[`mm-uav-threat-detection`](https://github.com/sharandaa/mm-uav-threat-detection) repository. A
substantial part of the detector, tracker, fusion, training, and evaluation code is reproduced
from that project and retained here to provide the experimental baseline. New contributions for
this thesis will be kept clearly distinguishable from the inherited implementation.

> **Project status:** research design and repository setup. The inherited detector and fusion
> code is retained as a baseline; the new VQC, classical comparison models, degradation pipeline,
> and evaluation workflow have not yet been implemented. Do not interpret the baseline results
> below as results from this thesis.

## Current thesis scope

The project keeps Suttorp et al.'s RGB and thermal detectors fixed and operates on their
decision-level outputs. The intended learned fusion inputs are detector beliefs and uncertainty
features rather than raw images. Planned comparisons include a variational quantum classifier,
an evidential MLP, an RBF-SVM, a random forest, and the inherited naive/ABF/DBF/gated fusion
baselines. The thesis manuscript and its experiment-design chapter are maintained privately and
are not published in this repository.