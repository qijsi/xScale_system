# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""S2 — Cost Model Skill.

Responsibility
--------------
Estimate latency, communication volume, and peak memory for any candidate
strategy without running it. These estimates drive the search engine's
acquisition function and fidelity routing.

Versions
--------
v0: [NOT STARTED]
    - Roofline latency: FLOPs / peak TFLOPS + comm_volume / bandwidth
    - Analytical memory: params + optimizer states + activations
    - Comm volume: allreduce = 2 * data / TP, alltoall = data * (EP-1) / EP

v1: [NOT STARTED]
    - Per-collective-type cost table from measured data
    - Kernel efficiency curves from MBS anchors (see adaptive.py calibration)
    - Feature-aware corrections: different multiplicative factors per TP/PP/EP region

v2: [NOT STARTED]
    - Per-op corrections from execution graph feedback
    - Cross-workload transfer via model-feature embeddings
    - Calibrated uncertainty intervals for each estimate

TODO
----
- [ ] Define CostEstimate dataclass (latency + std + comm_mb + peak_mem_gb + confidence)
- [ ] Implement v0 roofline + analytical memory
- [ ] Hook into FeedbackLearner for v1 online calibration
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostEstimate:
    """Analytical estimate of a strategy's performance."""

    latency_ms: float
    latency_std: float
    comm_volume_mb: float
    peak_memory_gb: float
    confidence: float


class CostModel:
    """Cost model skill — placeholder, v0 not yet implemented."""

    pass  # TODO: implement v0 roofline model
