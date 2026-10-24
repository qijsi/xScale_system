"""Tests for the search orchestrator."""

import unittest

from xscale.orchestrator import SearchOrchestrator, TrustLevel, TrustTracker
from xscale.skills.strategy_space import (
    ModelDescriptor,
    HardwareDescriptor,
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


def _small_hw() -> HardwareDescriptor:
    return HardwareDescriptor(
        gpu_model="H100",
        num_gpus=8,
        gpus_per_node=8,
        nvlink_bandwidth_gbps=900,
        internode_bandwidth_gbps=200,
        vram_per_gpu_gb=80,
        peak_tflops=1000,
    )


class TestTrustTracker(unittest.TestCase):
    def test_starts_at_zero(self):
        tt = TrustTracker()
        self.assertAlmostEqual(tt.trust, 0.0)
        self.assertEqual(tt.level(), TrustLevel.L0_EXPLORATION)

    def test_increases_with_accurate_predictions(self):
        tt = TrustTracker()
        cfg = ParallelConfig(2, 2, 2, 1, 4, "none")
        for _ in range(20):
            tt.update(cfg, predicted_ms=100.0, measured_ms=105.0)
        self.assertGreater(tt.trust, 0.3)

    def test_stays_low_with_bad_predictions(self):
        tt = TrustTracker()
        cfg = ParallelConfig(2, 2, 2, 1, 4, "none")
        for i in range(20):
            # predictions are all over the place
            tt.update(cfg, predicted_ms=100.0, measured_ms=200.0 + i)
        self.assertLess(tt.trust, 0.6)


class TestSearchOrchestrator(unittest.TestCase):
    def test_runs_to_completion(self):
        orch = SearchOrchestrator(
            _small_model(), _small_hw(), max_iterations=15,
        )
        history = orch.run()
        self.assertGreater(history.count, 0)
        self.assertIsNotNone(history.best_config)

    def test_trust_increases_over_time(self):
        orch = SearchOrchestrator(
            _small_model(), _small_hw(), max_iterations=30,
        )
        initial_trust = orch.trust.trust
        orch.run()
        self.assertGreater(orch.trust.trust, initial_trust)

    def test_finds_feasible_configs(self):
        orch = SearchOrchestrator(
            _small_model(), _small_hw(), max_iterations=20,
        )
        history = orch.run()
        self.assertGreater(history.feasible_count, 0)

    def test_budget_limit_works(self):
        orch = SearchOrchestrator(
            _small_model(), _small_hw(), budget_gpu_hours=0.001, max_iterations=1000,
        )
        history = orch.run()
        # Should stop early due to budget
        self.assertLess(orch._iterations, 100)

    def test_iteration_limit_works(self):
        orch = SearchOrchestrator(
            _small_model(), _small_hw(), max_iterations=5,
        )
        orch.run()
        self.assertLessEqual(orch._iterations, 5 + 10)  # +10 for seeds


if __name__ == "__main__":
    unittest.main()
