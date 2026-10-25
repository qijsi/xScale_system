# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""Tests for S1 — Evolving Strategy Space (v0)."""

import unittest

from xscale.skills.strategy_space import (
    EvolvingStrategySpace,
    FeasibilityStatus,
    HardwareDescriptor,
    ModelDescriptor,
    ParallelConfig,
)


def _small_model() -> ModelDescriptor:
    return ModelDescriptor(
        name="tiny-dense",
        num_layers=12,
        hidden_size=512,
        num_heads=8,
        vocab_size=32000,
        sequence_length=1024,
        is_moe=False,
        num_experts=None,
        expert_top_k=None,
        params_billion=0.1,
    )


def _small_hardware() -> HardwareDescriptor:
    return HardwareDescriptor(
        gpu_model="H100",
        num_gpus=8,
        gpus_per_node=8,
        nvlink_bandwidth_gbps=900,
        internode_bandwidth_gbps=200,
        vram_per_gpu_gb=80,
        peak_tflops=1000,
    )


class TestSeedGeneration(unittest.TestCase):
    def test_seed_creates_valid_configs(self):
        space = EvolvingStrategySpace(_small_model(), _small_hardware())
        seeds = space.seed()
        self.assertTrue(len(seeds) > 0)
        for cfg in seeds:
            self.assertEqual(cfg.world_size, 8)

    def test_seed_includes_multiple_tp_values(self):
        space = EvolvingStrategySpace(_small_model(), _small_hardware())
        seeds = space.seed()
        tp_values = {c.tp for c in seeds}
        self.assertIn(1, tp_values)
        self.assertIn(2, tp_values)

    def test_dense_model_has_ep_one(self):
        space = EvolvingStrategySpace(_small_model(), _small_hardware())
        seeds = space.seed()
        for cfg in seeds:
            self.assertEqual(cfg.ep, 1)


class TestNeighborhood(unittest.TestCase):
    def setUp(self):
        self.space = EvolvingStrategySpace(_small_model(), _small_hardware())
        self.space.seed()

    def test_neighbors_preserve_world_size(self):
        cfg = ParallelConfig(2, 2, 2, 1, 4, "none")
        neighbors = self.space.propose_neighbors(cfg)
        for n in neighbors:
            self.assertEqual(n.world_size, 8)

    def test_neighbors_include_mbs_change(self):
        cfg = ParallelConfig(2, 2, 2, 1, 4, "none")
        neighbors = self.space.propose_neighbors(cfg)
        mbs_values = {n.mbs for n in neighbors}
        self.assertTrue(2 in mbs_values or 8 in mbs_values)


class TestMarkResult(unittest.TestCase):
    def setUp(self):
        self.space = EvolvingStrategySpace(_small_model(), _small_hardware())
        self.space.seed()

    def test_feasible_mark_changes_status(self):
        cfg = ParallelConfig(1, 1, 8, 1, 4, "none")
        before = self.space.status_of(cfg)
        update = self.space.mark_result(cfg, feasible=True)
        after = self.space.status_of(cfg)
        self.assertIs(after, FeasibilityStatus.FEASIBLE)
        self.assertIn(cfg, update.newly_feasible)

    def test_oom_propagates_to_dominated_configs(self):
        cfg = ParallelConfig(2, 2, 2, 1, 4, "none")
        update = self.space.mark_result(cfg, feasible=False, reason="OOM")
        self.assertTrue(len(update.propagated_infeasible) > 0)
        from xscale.skills.heuristic_rules import is_memory_dominant
        for p in update.propagated_infeasible:
            self.assertTrue(
                is_memory_dominant(p, cfg),
                f"{p} should dominate {cfg}"
            )

    def test_coverage_increases_after_marking(self):
        before = self.space.coverage()
        cfg = ParallelConfig(1, 1, 8, 1, 4, "none")
        self.space.mark_result(cfg, feasible=True)
        after = self.space.coverage()
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
