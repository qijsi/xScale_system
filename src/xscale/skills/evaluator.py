# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""S5 — Evaluation Skill.

Responsibility
--------------
Execute a candidate strategy at the chosen fidelity level and return a
standardized result. Serves as the interface between the search engine and
the underlying evaluation substrate (analytical model, CrystalLLM replay,
real hardware).

Versions
--------
v0: [NOT STARTED]
    - Analytical evaluation only (calls CostModel)
    - Standardized ExecutionResult output type

v1: [NOT STARTED]
    - CrystalLLM replay integration
    - Single-config full pipeline (capture -> plan -> emulate)

v2: [NOT STARTED]
    - Incremental: strategy delta determines which stages re-run
    - Witness-rank-only execution with MBS anchor scheduling
    - Batch evaluation of multiple candidates

TODO
----
- [ ] Define ExecutionResult dataclass
- [ ] Implement v0 analytical evaluator
- [ ] Implement v1 CrystalLLM evaluator
- [ ] Hook into FeedbackLearner for automatic updates
"""

from __future__ import annotations

from dataclasses import dataclass

from ..witness.adaptive import EvaluationMode


@dataclass
class ExecutionResult:
    """Result from evaluating one strategy."""

    config: object  # ParallelConfig (will import when S1 is stable)
    latency_ms: float
    peak_memory_gb: float
    comm_volume_mb: float
    oom: bool
    success: bool
    mode: EvaluationMode
    notes: str = ""


class Evaluator:
    """Evaluation skill — v0 not yet implemented."""

    pass  # TODO: implement v0 analytical evaluator
