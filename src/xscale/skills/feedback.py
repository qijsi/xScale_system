# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""S4 — Feedback Learning Skill.

Responsibility
--------------
Interpret execution results and update cost model / structural assumptions
over time. Convert raw measurements into corrections that improve future
predictions.

Relationship to existing code
------------------------------
The ``witness.adaptive`` module implements v0: per-workload multiplicative
latency calibration and analytical/replay/real fidelity routing. This
module will extend that with v1/v2 features.

Versions
--------
v0: [EXISTING in witness/adaptive.py]
    - Global log-ratio correction per workload family
    - Three-tier fidelity routing (analytical / replay / real)
    - Memory-headroom-based escalation to real

v1: [NOT STARTED]
    - Per-dimension corrections (TP region / PP region / EP region)
    - Structural similarity: nearby configs share correction
    - Online update of uncertainty estimates

v2: [NOT STARTED]
    - Cross-workload transfer via model-feature embeddings
    - Learned ranking-aware correction (prioritize top-K accuracy)
    - Per-class stability learning for symmetry classes

TODO
----
- [ ] Define stable skill-layer API wrapping adaptive.py
- [ ] Implement v1 per-dimension corrections
- [ ] Implement v1 confidence-calibrated uncertainty
- [ ] Implement v2 cross-workload transfer
"""

from ..witness.adaptive import (
    AdaptiveEvaluationController,
    CandidateEstimate,
    EvaluationDecision,
    EvaluationMode,
)

__all__ = [
    "AdaptiveEvaluationController",
    "CandidateEstimate",
    "EvaluationDecision",
    "EvaluationMode",
]
