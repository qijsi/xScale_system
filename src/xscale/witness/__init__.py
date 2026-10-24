# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Witness-guided strategy optimization for large-scale LLM training."""

from .adaptive import AdaptiveEvaluationController, CandidateEstimate, EvaluationDecision, EvaluationMode
from .confidence import ConfidenceController, EscalationDecision, FISResult
from .delta import DeltaKind, StrategyDelta, compute_delta
from .partitioner import RankSymmetryPartitioner, SymmetryClass, WitnessSelection
from .planner import WitnessPlanner, WitnessPlan
from .signature import ParametricSignature, RankDescriptor, TopologySignature, extract_signatures
from .types import ExecutionTier, StrategyDimension, StrategySpec

__all__ = [
    "AdaptiveEvaluationController",
    "CandidateEstimate",
    "ConfidenceController",
    "DeltaKind",
    "EscalationDecision",
    "ExecutionTier",
    "EvaluationDecision",
    "EvaluationMode",
    "FISResult",
    "ParametricSignature",
    "RankDescriptor",
    "RankSymmetryPartitioner",
    "StrategyDelta",
    "StrategyDimension",
    "StrategySpec",
    "SymmetryClass",
    "TopologySignature",
    "WitnessPlan",
    "WitnessPlanner",
    "WitnessSelection",
    "extract_signatures",
    "compute_delta",
]
