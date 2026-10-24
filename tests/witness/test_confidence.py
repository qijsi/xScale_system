# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""Unit tests for confidence scoring and tier escalation."""

import unittest

from crystalllm.witness.confidence import ConfidenceController
from crystalllm.witness.planner import TierAssignment, WitnessPlan
from crystalllm.witness.types import ExecutionTier, StrategySpec


def make_plan(*assignments: tuple[str, ExecutionTier]) -> WitnessPlan:
    """Build a minimal plan without involving rank partitioning."""
    return WitnessPlan(
        strategy=StrategySpec("tp1", 1, "none"),
        tier_assignments=[
            TierAssignment(op_id, rank=0, tier=tier, reason="test") for op_id, tier in assignments
        ],
    )


class TestComputeFis(unittest.TestCase):
    def test_scores_assignments_and_sorts_by_combined_impact(self):
        plan = make_plan(
            ("proxy", ExecutionTier.PROXY),
            ("symbolic", ExecutionTier.SYMBOLIC),
            ("real", ExecutionTier.REAL),
        )

        results = ConfidenceController().compute_fis(plan, {"proxy"})

        self.assertEqual([result.op_id for result in results], ["symbolic", "proxy", "real"])
        scores = {result.op_id: result for result in results}
        self.assertAlmostEqual(scores["symbolic"].latency_fis, 0.8)
        self.assertAlmostEqual(scores["symbolic"].memory_fis, 0.2)
        self.assertAlmostEqual(scores["proxy"].latency_fis, 0.8 / 3)
        self.assertAlmostEqual(scores["proxy"].memory_fis, 1 / 3)
        self.assertEqual(scores["real"].combined, 0.0)

    def test_boundary_assignment_gets_higher_memory_impact(self):
        plan = make_plan(("ordinary", ExecutionTier.SHAPE_PARALLEL), ("boundary", ExecutionTier.SHAPE_PARALLEL))

        boundary, ordinary = ConfidenceController().compute_fis(plan, {"boundary"})

        self.assertEqual(boundary.op_id, "boundary")
        self.assertGreater(boundary.memory_fis, ordinary.memory_fis)
        self.assertGreater(boundary.combined, ordinary.combined)

    def test_empty_plan_has_no_fis_results(self):
        self.assertEqual(ConfidenceController().compute_fis(make_plan()), [])


class TestComputeCec(unittest.TestCase):
    def test_empty_plan_has_full_confidence(self):
        self.assertEqual(ConfidenceController().compute_cec(make_plan(), {"missing"}), 1.0)

    def test_confidence_is_real_fraction_minus_boundary_penalty(self):
        plan = make_plan(
            ("real", ExecutionTier.REAL),
            ("boundary", ExecutionTier.SHAPE_PARALLEL),
            ("ordinary", ExecutionTier.SYMBOLIC),
        )

        self.assertAlmostEqual(ConfidenceController().compute_cec(plan, {"boundary"}), 0.0)
        self.assertAlmostEqual(ConfidenceController().compute_cec(plan), 1 / 3)


class TestEscalate(unittest.TestCase):
    def test_satisfied_threshold_does_not_mutate_plan(self):
        plan = make_plan(("op", ExecutionTier.REAL))

        decisions = ConfidenceController(cec_threshold=1.0).escalate(plan)

        self.assertEqual(decisions, [])
        self.assertIs(plan.tier_assignments[0].tier, ExecutionTier.REAL)

    def test_escalates_one_tier_at_a_time_until_threshold_is_met(self):
        plan = make_plan(("op", ExecutionTier.SYMBOLIC))

        decisions = ConfidenceController(cec_threshold=1.0).escalate(plan)

        self.assertEqual(
            [(decision.from_tier, decision.to_tier) for decision in decisions],
            [
                (ExecutionTier.SYMBOLIC, ExecutionTier.SHAPE_PARALLEL),
                (ExecutionTier.SHAPE_PARALLEL, ExecutionTier.PROXY),
                (ExecutionTier.PROXY, ExecutionTier.REAL),
            ],
        )
        self.assertEqual(plan.tier_assignments[0].tier, ExecutionTier.REAL)

    def test_memory_boundary_assignment_is_selected_and_explained(self):
        plan = make_plan(
            ("ordinary", ExecutionTier.PROXY),
            ("boundary", ExecutionTier.SHAPE_PARALLEL),
        )

        decisions = ConfidenceController(cec_threshold=1.0).escalate(plan, {"boundary"})

        self.assertEqual(decisions[0].op_id, "boundary")
        self.assertEqual(decisions[0].reason, "near OOM boundary, memory fidelity required")

    def test_max_escalations_bounds_mutations(self):
        plan = make_plan(("op", ExecutionTier.SYMBOLIC))

        decisions = ConfidenceController(cec_threshold=1.0, max_escalations=1).escalate(plan)

        self.assertEqual(len(decisions), 1)
        self.assertEqual(plan.tier_assignments[0].tier, ExecutionTier.SHAPE_PARALLEL)


if __name__ == "__main__":
    unittest.main()