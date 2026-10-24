# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""Tests for feedback-driven evaluation fidelity selection."""

import unittest

from crystalllm.witness.adaptive import (
    AdaptiveEvaluationController,
    CandidateEstimate,
    EvaluationMode,
)


def make_estimate(
    candidate_id: str = "candidate",
    workload_key: str = "dense",
    latency_ms: float = 100.0,
    uncertainty: float = 0.01,
    predicted_memory: int = 500,
    available_memory: int = 1000,
) -> CandidateEstimate:
    return CandidateEstimate(
        candidate_id,
        workload_key,
        latency_ms,
        uncertainty,
        predicted_memory,
        available_memory,
    )


class TestAdaptiveEvaluationController(unittest.TestCase):
    def test_requires_real_anchor_before_trusting_analysis(self):
        controller = AdaptiveEvaluationController()

        decision = controller.choose_mode(make_estimate(), incumbent_latency_ms=80.0)

        self.assertIs(decision.mode, EvaluationMode.REAL)
        self.assertIn("anchor", decision.reason)

    def test_real_feedback_corrects_future_latency_predictions(self):
        controller = AdaptiveEvaluationController()
        controller.observe_real("dense", analytical_latency_ms=100.0, real_latency_ms=120.0)

        predicted, uncertainty = controller.predict(make_estimate())

        self.assertAlmostEqual(predicted, 120.0)
        self.assertAlmostEqual(uncertainty, 0.01)
        self.assertEqual(controller.real_sample_count("dense"), 1)

    def test_calibration_is_isolated_by_workload(self):
        controller = AdaptiveEvaluationController()
        controller.observe_real("dense", analytical_latency_ms=100.0, real_latency_ms=120.0)

        dense_prediction, _ = controller.predict(make_estimate(workload_key="dense"))
        moe_prediction, _ = controller.predict(make_estimate(workload_key="moe"))

        self.assertAlmostEqual(dense_prediction, 120.0)
        self.assertAlmostEqual(moe_prediction, 100.0)

    def test_safely_dominated_candidate_stays_analytical(self):
        controller = AdaptiveEvaluationController()
        controller.observe_real("dense", 100.0, 100.0)

        decision = controller.choose_mode(make_estimate(), incumbent_latency_ms=80.0)

        self.assertIs(decision.mode, EvaluationMode.ANALYTICAL)

    def test_uncertain_contender_escalates_to_real_hardware(self):
        controller = AdaptiveEvaluationController()
        controller.observe_real("dense", 100.0, 100.0)

        decision = controller.choose_mode(make_estimate(uncertainty=0.25), incumbent_latency_ms=110.0)

        self.assertIs(decision.mode, EvaluationMode.REAL)
        self.assertIn("uncertainty", decision.reason)

    def test_moderately_uncertain_contender_uses_replay(self):
        controller = AdaptiveEvaluationController()
        controller.observe_real("dense", 100.0, 100.0)

        decision = controller.choose_mode(make_estimate(uncertainty=0.10), incumbent_latency_ms=110.0)

        self.assertIs(decision.mode, EvaluationMode.REPLAY)

    def test_memory_boundary_forces_real_measurement(self):
        controller = AdaptiveEvaluationController()
        controller.observe_real("dense", 100.0, 100.0)

        decision = controller.choose_mode(
            make_estimate(predicted_memory=950), incumbent_latency_ms=80.0
        )

        self.assertIs(decision.mode, EvaluationMode.REAL)
        self.assertIn("memory", decision.reason)

    def test_real_observations_update_bias_and_residual_uncertainty(self):
        controller = AdaptiveEvaluationController()
        controller.observe_real("dense", 100.0, 120.0)
        controller.observe_real("dense", 100.0, 150.0)

        predicted, uncertainty = controller.predict(make_estimate())

        self.assertAlmostEqual(predicted, (120.0 * 150.0) ** 0.5)
        self.assertGreater(uncertainty, 0.01)
        self.assertEqual(controller.real_sample_count("dense"), 2)


if __name__ == "__main__":
    unittest.main()