# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""S3 — Structure Skill.

Responsibility
--------------
Given an execution graph, determine which ranks and operations need full
fidelity, and which can be compressed. Provide strategy-delta analysis so
we can reuse previous results across configs.

Relationship to existing code
------------------------------
This module wraps the witness planning infrastructure in
``xscale.witness`` with a skill-layer interface. The witness module is
the implementation; this module defines the stable API the search engine
uses.

Versions
--------
v0: [EXISTING — needs fixes]
    - Topology-signature based rank symmetry (witness/partitioner.py)
    - Heuristic tier assignment (witness/planner.py)
    - Strategy delta computation (witness/delta.py)
    - FIS/CEC scoring (witness/confidence.py)

v0 known defects
----------------
1. [CRITICAL] Boundary symmetry class adds ALL members as witnesses —
   destroys compression ratio. Should pick one boundary representative.
2. [HIGH] CEC formula requires ~90% of ops at REAL tier — defeats
   compression. Should be FIS-weighted, not uniformly counted.
3. [MEDIUM] Tier assignment ignores causal criticality — all comm ops go
   SYMBOLIC even if they're on the critical path.

v1: [NOT STARTED]
    - Causal DAG criticality from real execution graph
    - Per-stage local re-capture for recompute changes
    - FIS-weighted CEC

v2: [NOT STARTED]
    - Learned class stability (which symmetry classes survive transitions)
    - Learned FIS from execution feedback
    - Cross-workload class transfer

TODO
----
- [ ] Fix defect 1: boundary representative selection
- [ ] Fix defect 2: FIS-weighted CEC
- [ ] Fix defect 3: critical-path tier assignment
- [ ] Define stable skill-layer API wrapping witness module
"""

# Skill-layer re-exports of the witness module implementation.
# As we add v1/v2 features, they live in this module; the witness module
# stays as the low-level implementation.

from ..witness.partitioner import (
    RankSymmetryPartitioner,
    SymmetryClass,
    WitnessSelection,
)
from ..witness.planner import WitnessPlan, WitnessPlanner
from ..witness.delta import DeltaKind, StrategyDelta, compute_delta
from ..witness.confidence import ConfidenceController, EscalationDecision, FISResult
from ..witness.types import ExecutionTier

__all__ = [
    "ConfidenceController",
    "DeltaKind",
    "EscalationDecision",
    "ExecutionTier",
    "FISResult",
    "RankSymmetryPartitioner",
    "StrategyDelta",
    "SymmetryClass",
    "WitnessPlan",
    "WitnessPlanner",
    "WitnessSelection",
    "compute_delta",
]
