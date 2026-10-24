# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""Shared types for the xScale skill layer."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class FeasibilityStatus(enum.Enum):
    """How confident we are that a config can run.

    Order matters: UNKNOWN < LIKELY_INFEASIBLE < FEASIBLE == INFEASIBLE (terminal).
    """

    UNKNOWN = "unknown"
    LIKELY_FEASIBLE = "likely_feasible"
    LIKELY_INFEASIBLE = "likely_infeasible"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class ParallelConfig:
    """A complete parallel execution strategy.

    The tp*pp*dp*ep product must equal the target world size.
    """

    tp: int
    pp: int
    dp: int
    ep: int
    mbs: int
    recompute: str  # "none" | "selective" | "full"

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.dp * self.ep

    def __str__(self) -> str:
        return f"tp{self.tp}_pp{self.pp}_dp{self.dp}_ep{self.ep}_mbs{self.mbs}_rc{self.recompute}"


@dataclass(frozen=True)
class ModelDescriptor:
    """Model parameters relevant to strategy search."""

    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    vocab_size: int
    sequence_length: int
    is_moe: bool
    num_experts: int | None
    expert_top_k: int | None
    params_billion: float


@dataclass(frozen=True)
class HardwareDescriptor:
    """Target hardware topology for which we search strategies."""

    gpu_model: str
    num_gpus: int
    gpus_per_node: int
    nvlink_bandwidth_gbps: float
    internode_bandwidth_gbps: float
    vram_per_gpu_gb: float
    peak_tflops: float


@dataclass
class SpaceUpdate:
    """Result of a space state update (marking a config feasible/infeasible)."""

    newly_feasible: list[ParallelConfig]
    newly_infeasible: list[ParallelConfig]
    propagated_infeasible: list[ParallelConfig]
    new_coverage: float = 0.0
