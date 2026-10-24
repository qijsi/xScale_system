# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""xScale skill layer: domain knowledge modules for strategy search.

Each skill module has a versioned interface (v0 -> v1 -> v2) so the search
engine can depend on a stable API while internal models improve.

Skills:
    S1 strategy_space  Evolving strategy space (feasibility + coverage)
    S2 cost_model      Analytical latency / communication / memory estimates
    S3 structure       Rank symmetry compression + tier assignment + delta reuse
    S4 feedback        Learn from execution results to refine estimates
    S5 evaluator       Run candidate strategies at chosen fidelity levels

See docs/xscale-skill-layer-design.md for the full design document.
"""

from . import strategy_space
from . import heuristic_rules
from . import cost_model
from . import structure
from . import feedback
from . import evaluator

__all__ = [
    "strategy_space",
    "heuristic_rules",
    "cost_model",
    "structure",
    "feedback",
    "evaluator",
]

# Progress snapshot (updated as skills mature):
#   S1 strategy_space : v0 (seed + neighbors + hard constraints + heuristic ranking)
#   S1b heuristic_rules: v0 (23 rules, 4 categories)
#   S2 cost_model     : not started
#   S3 structure      : v0 exists (witness module), defects known
#   S4 feedback       : v0 exists (adaptive.py)
#   S5 evaluator      : not started
