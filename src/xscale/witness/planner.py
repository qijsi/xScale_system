# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Witness planning with tier assignment and MBS anchor scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field

from .delta import DeltaKind, StrategyDelta, compute_delta
from .partitioner import RankSymmetryPartitioner, WitnessSelection
from .types import ExecutionTier, RankDescriptor, StrategySpec


@dataclass
class TierAssignment:
    """Per-operation abstraction tier assignment.

    Attributes:
        op_id: The operation identifier.
        rank: The owning rank.
        tier: The assigned execution tier.
        reason: Human-readable justification for the tier choice.
    """

    op_id: str
    rank: int
    tier: ExecutionTier
    reason: str


@dataclass
class WitnessPlan:
    """A complete execution plan for one strategy evaluation.

    Attributes:
        strategy: The strategy being evaluated.
        witness_ranks: Ranks selected for execution.
        tier_assignments: Per-operation tier assignments for witness ranks.
        mbs_anchors: MBS values requiring REAL execution for calibration.
        estimated_cost: Relative estimated cost of this plan.
        delta: Strategy delta from the previous plan, if any.
    """

    strategy: StrategySpec
    witness_ranks: list[int] = field(default_factory=list)
    tier_assignments: list[TierAssignment] = field(default_factory=list)
    mbs_anchors: list[int] = field(default_factory=list)
    estimated_cost: float = 0.0
    delta: StrategyDelta | None = None

    @property
    def real_op_count(self) -> int:
        """Number of operations assigned to REAL tier."""
        return sum(1 for t in self.tier_assignments if t.tier is ExecutionTier.REAL)


class WitnessPlanner:
    """Plans the minimal witness execution for a strategy.

    The planner integrates three decisions:
    1. Which ranks to execute (from the symmetry partitioner).
    2. Which MBS values need REAL anchor execution for calibration.
    3. Which abstraction tier each operation uses.
    """

    def __init__(self, pp_size: int = 1, mbs_anchor_count: int = 3):
        """Initialize the planner.

        Args:
            pp_size: Pipeline-parallel size for boundary detection.
            mbs_anchor_count: Number of MBS anchor points (capped at 3).
        """
        self.partitioner = RankSymmetryPartitioner(pp_size=pp_size)
        self.mbs_anchor_count = min(3, max(1, mbs_anchor_count))

    def _mbs_anchors(self, mbs: int, mbs_range: tuple[int, int]) -> list[int]:
        low, high = mbs_range
        if high <= low:
            return [mbs]
        if self.mbs_anchor_count == 1:
            return [mbs]
        if self.mbs_anchor_count == 2:
            return sorted({low, high})
        quarter = (high - low) // 4
        return sorted({low, low + quarter, high})

    def plan(
        self,
        ranks: list[RankDescriptor],
        strategy: StrategySpec,
        previous: StrategySpec | None = None,
        mbs_range: tuple[int, int] = (1, 8),
    ) -> WitnessPlan:
        """Produce a witness execution plan.

        Args:
            ranks: All logical rank descriptors.
            strategy: The strategy to evaluate.
            previous: The previously evaluated strategy, for delta computation.
            mbs_range: Valid MBS range for anchor selection.

        Returns:
            A WitnessPlan with tier assignments and anchor schedule.
        """
        selection: WitnessSelection = self.partitioner.partition(ranks)
        delta = compute_delta(previous, strategy) if previous is not None else None

        anchors: list[int] = []
        if delta is None or delta.kind is not DeltaKind.NONE:
            if delta is None or DeltaKind.MBS in delta.changed_dimensions:
                anchors = self._mbs_anchors(strategy.mbs, mbs_range)

        plan = WitnessPlan(
            strategy=strategy,
            witness_ranks=selection.witness_ranks,
            mbs_anchors=anchors,
            delta=delta,
        )

        witness_set = set(selection.witness_ranks)
        for rank in ranks:
            if rank.rank not in witness_set:
                continue
            for op in rank.operations:
                tier = self._tier_for(op.op_type, op.is_comm, rank)
                plan.tier_assignments.append(
                    TierAssignment(
                        op_id=op.op_id,
                        rank=rank.rank,
                        tier=tier,
                        reason=self._tier_reason(tier, op.op_type, rank),
                    )
                )

        tier_cost = {ExecutionTier.SYMBOLIC: 0.1, ExecutionTier.SHAPE_PARALLEL: 0.3,
                     ExecutionTier.PROXY: 0.6, ExecutionTier.REAL: 1.0}
        plan.estimated_cost = sum(tier_cost[t.tier] for t in plan.tier_assignments) * (1 + len(anchors))
        return plan

    def _tier_for(self, op_type: str, is_comm: bool, rank: RankDescriptor) -> ExecutionTier:
        """Choose the execution tier for one operation."""
        if is_comm:
            return ExecutionTier.SYMBOLIC
        if op_type in ("layernorm", "dropout", "gelu"):
            return ExecutionTier.PROXY
        if rank.is_dp_replica:
            return ExecutionTier.SHAPE_PARALLEL
        return ExecutionTier.REAL

    def _tier_reason(self, tier: ExecutionTier, op_type: str, rank: RankDescriptor) -> str:
        if tier is ExecutionTier.SYMBOLIC:
            return "communication with stable pattern"
        if tier is ExecutionTier.PROXY:
            return f"{op_type} is elementwise, shape-preserving proxy sufficient"
        if tier is ExecutionTier.SHAPE_PARALLEL:
            return "DP-replica op, batched shape-parallel execution"
        return f"critical compute on rank {rank.rank}, full fidelity required"
