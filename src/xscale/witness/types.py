# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Shared types for witness-guided strategy optimization."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class ExecutionTier(enum.IntEnum):
    """Execution abstraction tier, ordered from cheapest to most expensive."""

    SYMBOLIC = 0
    SHAPE_PARALLEL = 1
    PROXY = 2
    REAL = 3


class StrategyDimension(enum.Enum):
    """Dimensions of the parallel strategy space."""

    TOPOLOGY = "topology"
    MBS = "mbs"
    RECOMPUTE = "recompute"


@dataclass(frozen=True)
class StrategySpec:
    """A candidate parallel strategy specification.

    Attributes:
        topology: Parallelism layout, e.g. "tp1_pp4_ep8".
        mbs: Micro-batch size.
        recompute: Activation recomputation mode.
    """

    topology: str
    mbs: int
    recompute: str

    def dimension(self, dim: StrategyDimension) -> str | int:
        """Return the value of one strategy dimension."""
        if dim is StrategyDimension.TOPOLOGY:
            return self.topology
        if dim is StrategyDimension.MBS:
            return self.mbs
        return self.recompute


@dataclass
class OperationDescriptor:
    """A minimal description of one operation for planning purposes.

    Attributes:
        op_id: Unique operation identifier.
        rank: Owning logical rank.
        op_type: Coarse operation type, e.g. "allreduce", "matmul".
        shape: Tensor shape tuple.
        duration_ns: Measured or estimated duration in nanoseconds.
        is_comm: Whether this is a communication operation.
    """

    op_id: str
    rank: int
    op_type: str
    shape: tuple[int, ...]
    duration_ns: int
    is_comm: bool = False


@dataclass
class RankDescriptor:
    """A minimal description of one logical rank for signature extraction.

    Attributes:
        rank: Global rank identifier.
        comm_groups: Names of process groups this rank belongs to.
        op_sequence: Ordered list of operation type names.
        pp_stage: Pipeline-parallel stage index.
        is_dp_replica: Whether this rank is a data-parallel replica.
        is_ep_member: Whether this rank holds MoE experts.
    """

    rank: int
    comm_groups: tuple[str, ...] = ()
    op_sequence: tuple[str, ...] = ()
    pp_stage: int = 0
    is_dp_replica: bool = True
    is_ep_member: bool = False
    operations: list[OperationDescriptor] = field(default_factory=list)


@dataclass(frozen=True)
class TopologySignature:
    """Topology-invariant part of a rank signature.

    Two ranks with equal topology signatures are structurally symmetric under
    MBS and recomputation changes. The signature is derived only from the
    communication graph structure, dependency topology, and pipeline-stage
    role — not from tensor shapes or recomputation mode.

    Attributes:
        comm_groups_hash: Hash of sorted process-group membership.
        op_structure_hash: Hash of the operation-type sequence.
        pp_stage: Pipeline-parallel stage index.
        is_dp_replica: Data-parallel replica flag.
        is_ep_member: Expert-parallel member flag.
    """

    comm_groups_hash: str
    op_structure_hash: str
    pp_stage: int
    is_dp_replica: bool
    is_ep_member: bool


@dataclass(frozen=True)
class ParametricSignature:
    """MBS/recompute-dependent part of a rank signature.

    Unlike TopologySignature, this changes when micro-batch size or
    recomputation mode changes. It captures tensor shapes, operation
    granularity, and the recomputation mode itself.

    Attributes:
        shapes_hash: Hash of the ordered tensor-shape list.
        recompute_mode: Current recomputation mode.
        mbs: Current micro-batch size.
    """

    shapes_hash: str
    recompute_mode: str
    mbs: int
