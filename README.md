# Quantum and Classical Decision-Level Fusion for Multimodal Anti-UAV Detection

MSc thesis research building on the multimodal evidential-fusion work of
Suttorp et al. The research studies whether variational quantum representations improve
RGB--thermal decision-level fusion under clean, degraded, and missing-sensor conditions.

This repository is a successor to Suttorp's
[`mm-uav-threat-detection`](https://github.com/sharandaa/mm-uav-threat-detection) repository. A
substantial part of the detector, tracker, fusion, training, and evaluation code is reproduced
from that project and retained here to provide the experimental baseline. New contributions for
this thesis will be kept clearly distinguishable from the inherited implementation.

> **Project status:** the replacement detector pair, degradation operators and clean
> development caches are verified. The original presence-fusion comparison is now
> under development. Test evaluation remains locked; inherited results are not
> results from this thesis.

## Current thesis scope

The project uses a frozen RGB–thermal detector pair trained with Suttorp et al.'s released
procedure. Fusion operates on detector beliefs and uncertainty features rather than raw images. Planned comparisons include a variational quantum classifier,
an evidential MLP, an RBF-SVM, a random forest, and the inherited naive/ABF/DBF/gated fusion
baselines. The thesis manuscript and its experiment-design chapter are maintained privately and
are not published in this repository.
