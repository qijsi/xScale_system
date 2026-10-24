# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Dual-objective FIS computation and CEC-guided escalation."""

from __future__ import annotations

from dataclasses import dataclass

from .planner import WitnessPlan
from .types import ExecutionTier


@dataclass(frozen=True)
class FISResult:
    """Fidelity Impact Score for one tier assignment.

    Attributes:
        op_id: The operation identifier.
        latency_fis: Sensitivity of latency ranking to abstraction replacement.
        memory_fis: Sensitivity of OOM classification to abstraction replacement.
    """

    op_id: str
    latency_fis: float
    memory_fis: float

    @property
    def combined(self) -> float:
        """Weighted combination with equal latency/memory weights."""
        return 0.5 * self.latency_fis + 0.5 * self.memory_fis


@dataclass(frozen=True)
class EscalationDecision:
    """Decision to escalate one operation to a higher tier.

    Attributes:
        op_id: The operation to escalate.
        from_tier: Current tier.
        to_tier: Target tier (one level higher).
        reason: Why escalation is needed.
    """

    op_id: str
    from_tier: ExecutionTier
    to_tier: ExecutionTier
    reason: str


class ConfidenceController:
    """Monitors plan confidence and triggers tier escalation.

    The controller computes a Compressed Execution Confidence (CEC) score
    from the fraction of REAL operations and the fraction of operations near
    the memory boundary. When CEC falls below a threshold, it identifies
    the highest-FIS non-REAL operation and escalates it by one tier.
    """

    def __init__(self, cec_threshold: float = 0.9, max_escalations: int = 100):
        """Initialize the controller.

        Args:
            cec_threshold: Minimum acceptable CEC score.
            max_escalations: Safety bound on escalation iterations.
        """
        self.cec_threshold = cec_threshold
        self.max_escalations = max_escalations

    def compute_fis(self, plan: WitnessPlan, memory_boundary_ops: set[str] | None = None) -> list[FISResult]:
        """Compute dual-objective FIS for all tier assignments.

        Args:
            plan: The witness plan to score.
            memory_boundary_ops: Operation IDs near the OOM boundary.

        Returns:
            One FISResult per tier assignment, sorted by combined impact.
        """
        boundary = memory_boundary_ops or set()
        results: list[FISResult] = []
        for assignment in plan.tier_assignments:
            tier_penalty = 1.0 - (assignment.tier.value / ExecutionTier.REAL.value)
            latency_fis = tier_penalty * 0.8
            memory_fis = tier_penalty * (1.0 if assignment.op_id in boundary else 0.2)
            results.append(FISResult(assignment.op_id, latency_fis, memory_fis))
        results.sort(key=lambda r: r.combined, reverse=True)
        return results

    def compute_cec(self, plan: WitnessPlan, memory_boundary_ops: set[str] | None = None) -> float:
        """Compute the Compressed Execution Confidence of a plan.

        CEC is estimated as the fraction of latency-critical weight already
        at REAL tier plus a penalty for boundary-memory ops not at REAL.
        """
        boundary = memory_boundary_ops or set()
        if not plan.tier_assignments:
            return 1.0
        real_weight = sum(1 for t in plan.tier_assignments if t.tier is ExecutionTier.REAL)
        base = real_weight / len(plan.tier_assignments)
        boundary_not_real = sum(
            1 for t in plan.tier_assignments if t.op_id in boundary and t.tier is not ExecutionTier.REAL
        )
        penalty = boundary_not_real / len(plan.tier_assignments)
        return max(0.0, base - penalty)

    def escalate(
        self, plan: WitnessPlan, memory_boundary_ops: set[str] | None = None
    ) -> list[EscalationDecision]:
        """Escalate operations until CEC meets the threshold.

        Args:
            plan: The plan to refine (mutated in place).
            memory_boundary_ops: Operation IDs near the OOM boundary.

        Returns:
            All escalation decisions taken, in order.
        """
        decisions: list[EscalationDecision] = []
        for _ in range(self.max_escalations):
            if self.compute_cec(plan, memory_boundary_ops) >= self.cec_threshold:
                break
            fis = self.compute_fis(plan, memory_boundary_ops)
            by_op = {t.op_id: t for t in plan.tier_assignments}
            target = None
            for candidate in fis:
                assignment = by_op[candidate.op_id]
                if assignment.tier is not ExecutionTier.REAL:
                    target = assignment
                    break
            if target is None:
                break
            new_tier = ExecutionTier(target.tier.value + 1)
            boundary = memory_boundary_ops or set()
            if target.op_id in boundary:
                reason = "near OOM boundary, memory fidelity required"
            else:
                reason = "CEC below threshold, latency ranking uncertainty"
            decisions.append(EscalationDecision(target.op_id, target.tier, new_tier, reason))
            target.tier = new_tier
        return decisions
