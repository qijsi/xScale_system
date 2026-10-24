# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Tests for the witness-guided strategy optimization prototype."""

import unittest
from crystalllm.witness import compute_delta

from crystalllm.witness import (
    ConfidenceController,
    DeltaKind,
    ExecutionTier,
    RankSymmetryPartitioner,
    StrategySpec,
    WitnessPlanner,
    extract_signatures,
)
from crystalllm.witness.types import OperationDescriptor, RankDescriptor


def make_rank(rank_id, groups=("dp",), ops=("allreduce", "matmul"), stage=0, dp=True, ep=False):
    return RankDescriptor(
        rank=rank_id,
        comm_groups=groups,
        op_sequence=ops,
        pp_stage=stage,
        is_dp_replica=dp,
        is_ep_member=ep,
        operations=[
            OperationDescriptor(f"op{rank_id}_{i}", rank_id, op_type, (128, 128), 1000, op_type == "allreduce")
            for i, op_type in enumerate(ops)
        ],
    )


class TestSignature(unittest.TestCase):
    def test_topology_signature_invariant_to_mbs(self):
        rank = make_rank(0)
        sig_a = extract_signatures(rank, mbs=1, recompute_mode="none")[0]
        sig_b = extract_signatures(rank, mbs=8, recompute_mode="full")[0]
        self.assertEqual(sig_a, sig_b)

    def test_parametric_signature_changes_with_mbs(self):
        rank = make_rank(0)
        sig_a = extract_signatures(rank, mbs=1, recompute_mode="none")[1]
        sig_b = extract_signatures(rank, mbs=8, recompute_mode="none")[1]
        self.assertNotEqual(sig_a, sig_b)


class TestPartitioner(unittest.TestCase):
    def test_dp_replicas_collapse(self):
        ranks = [make_rank(i) for i in range(64)]
        selection = RankSymmetryPartitioner(pp_size=1).partition(ranks)
        self.assertEqual(len(selection.classes), 1)
        self.assertEqual(len(selection.witness_ranks), 1)
        self.assertEqual(selection.compression_ratio, 64.0)

    def test_pp_stages_separate(self):
        ranks = [make_rank(i, stage=i % 4) for i in range(64)]
        selection = RankSymmetryPartitioner(pp_size=4).partition(ranks)
        self.assertEqual(len(selection.classes), 4)


class TestDelta(unittest.TestCase):
    def test_mbs_only_reuses_witness(self):
        delta = compute_delta(StrategySpec("tp1", 1, "none"), StrategySpec("tp1", 8, "none"))
        self.assertIs(delta.kind, DeltaKind.MBS_ONLY)
        self.assertTrue(delta.reuse_witness_set)

    def test_topology_repartitions(self):
        delta = compute_delta(StrategySpec("tp1", 1, "none"), StrategySpec("tp2", 1, "none"))
        self.assertIs(delta.kind, DeltaKind.TOPOLOGY_ONLY)
        self.assertFalse(delta.reuse_witness_set)


class TestPlanner(unittest.TestCase):
    def test_plan_assigns_tiers(self):
        ranks = [make_rank(i) for i in range(32)]
        planner = WitnessPlanner(pp_size=1)
        plan = planner.plan(ranks, StrategySpec("tp1", 4, "none"))
        self.assertEqual(len(plan.witness_ranks), 1)
        self.assertTrue(len(plan.tier_assignments) > 0)
        tiers = {t.tier for t in plan.tier_assignments}
        self.assertIn(ExecutionTier.SHAPE_PARALLEL, tiers)

    def test_mbs_anchors_scheduled(self):
        ranks = [make_rank(i) for i in range(8)]
        planner = WitnessPlanner(pp_size=1, mbs_anchor_count=3)
        plan = planner.plan(ranks, StrategySpec("tp1", 4, "none"), mbs_range=(1, 8))
        self.assertEqual(plan.mbs_anchors, [1, 2, 8])


class TestConfidenceController(unittest.TestCase):
    def test_cec_increases_after_escalation(self):
        ranks = [make_rank(i, ops=("allreduce", "matmul", "layernorm")) for i in range(8)]
        planner = WitnessPlanner(pp_size=1)
        plan = planner.plan(ranks, StrategySpec("tp1", 4, "none"))
        controller = ConfidenceController(cec_threshold=0.95, max_escalations=50)
        before = controller.compute_cec(plan)
        decisions = controller.escalate(plan)
        after = controller.compute_cec(plan)
        self.assertTrue(len(decisions) > 0)
        self.assertGreaterEqual(after, before)

    def test_boundary_ops_escalated(self):
        ranks = [make_rank(i, ops=("matmul",)) for i in range(4)]
        planner = WitnessPlanner(pp_size=1)
        plan = planner.plan(ranks, StrategySpec("tp1", 8, "none"))
        boundary = {plan.tier_assignments[0].op_id}
        controller = ConfidenceController(cec_threshold=0.99, max_escalations=50)
        decisions = controller.escalate(plan, memory_boundary_ops=boundary)
        boundary_decisions = [d for d in decisions if "memory" in d.reason]
        self.assertTrue(len(boundary_decisions) > 0)


if __name__ == "__main__":
    unittest.main()
