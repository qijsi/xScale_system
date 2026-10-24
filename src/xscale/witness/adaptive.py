# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Feedback-driven selection of strategy-evaluation fidelity."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field


class EvaluationMode(enum.IntEnum):
    """Increasing evaluation cost and fidelity."""

    ANALYTICAL = 0
    REPLAY = 1
    REAL = 2


@dataclass(frozen=True)
class CandidateEstimate:
    """Analysis available before spending evaluation resources."""

    candidate_id: str
    workload_key: str
    analytical_latency_ms: float
    relative_uncertainty: float
    predicted_peak_memory_bytes: int
    available_memory_bytes: int


@dataclass(frozen=True)
class EvaluationDecision:
    """Selected fidelity and calibrated estimate for one candidate."""

    mode: EvaluationMode
    predicted_latency_ms: float
    relative_uncertainty: float
    reason: str


@dataclass
class _Calibration:
    """Online statistics of log(real latency / analytical latency)."""

    count: int = 0
    mean_log_ratio: float = 0.0
    squared_deviation_sum: float = 0.0

    def update(self, analytical_latency_ms: float, real_latency_ms: float) -> None:
        log_ratio = math.log(real_latency_ms / analytical_latency_ms)
        self.count += 1
        delta = log_ratio - self.mean_log_ratio
        self.mean_log_ratio += delta / self.count
        self.squared_deviation_sum += delta * (log_ratio - self.mean_log_ratio)

    @property
    def residual_variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.squared_deviation_sum / (self.count - 1)


@dataclass
class AdaptiveEvaluationController:
    """Allocate analytical, replay, and real evaluation from feedback.

    The controller learns a multiplicative latency correction per workload
    family from real measurements. It reserves hardware runs for initial
    calibration, plausible ranking contenders, high-uncertainty predictions,
    and candidates near the predicted memory limit.
    """

    replay_uncertainty: float = 0.05
    real_uncertainty: float = 0.20
    ranking_margin: float = 0.03
    memory_safety_margin: float = 0.10
    exploration_z: float = 1.0
    _calibrations: dict[str, _Calibration] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.replay_uncertainty <= self.real_uncertainty:
            raise ValueError("uncertainty thresholds must satisfy 0 <= replay <= real")
        if self.ranking_margin < 0 or self.exploration_z < 0:
            raise ValueError("ranking margin and exploration factor must be non-negative")
        if not 0 <= self.memory_safety_margin < 1:
            raise ValueError("memory safety margin must be in [0, 1)")

    def predict(self, estimate: CandidateEstimate) -> tuple[float, float]:
        """Apply learned bias correction and return latency plus uncertainty."""
        self._validate_estimate(estimate)
        calibration = self._calibrations.get(estimate.workload_key)
        if calibration is None:
            return estimate.analytical_latency_ms, estimate.relative_uncertainty
        corrected = estimate.analytical_latency_ms * math.exp(calibration.mean_log_ratio)
        uncertainty = math.sqrt(estimate.relative_uncertainty**2 + calibration.residual_variance)
        return corrected, uncertainty

    def choose_mode(
        self,
        estimate: CandidateEstimate,
        incumbent_latency_ms: float | None = None,
    ) -> EvaluationDecision:
        """Choose the least costly fidelity that can safely inform the search."""
        predicted_latency, uncertainty = self.predict(estimate)
        calibration = self._calibrations.get(estimate.workload_key)

        if calibration is None:
            return EvaluationDecision(
                EvaluationMode.REAL,
                predicted_latency,
                uncertainty,
                "real-hardware anchor required for this workload",
            )

        memory_headroom = (
            estimate.available_memory_bytes - estimate.predicted_peak_memory_bytes
        ) / estimate.available_memory_bytes
        if memory_headroom <= self.memory_safety_margin:
            return EvaluationDecision(
                EvaluationMode.REAL,
                predicted_latency,
                uncertainty,
                "predicted memory use is near or above the limit",
            )

        optimistic_latency = predicted_latency * max(0.0, 1.0 - self.exploration_z * uncertainty)
        is_contender = (
            incumbent_latency_ms is None
            or optimistic_latency <= incumbent_latency_ms * (1.0 + self.ranking_margin)
        )
        if not is_contender:
            return EvaluationDecision(
                EvaluationMode.ANALYTICAL,
                predicted_latency,
                uncertainty,
                "candidate is safely dominated under the current uncertainty",
            )

        if uncertainty >= self.real_uncertainty:
            return EvaluationDecision(
                EvaluationMode.REAL,
                predicted_latency,
                uncertainty,
                "uncertainty could change the leading strategy",
            )
        if uncertainty >= self.replay_uncertainty:
            return EvaluationDecision(
                EvaluationMode.REPLAY,
                predicted_latency,
                uncertainty,
                "plausible contender needs an intermediate-fidelity check",
            )
        return EvaluationDecision(
            EvaluationMode.REPLAY,
            predicted_latency,
            uncertainty,
            "validate a likely improvement before final hardware confirmation",
        )

    def observe_real(
        self,
        workload_key: str,
        analytical_latency_ms: float,
        real_latency_ms: float,
    ) -> None:
        """Update workload calibration from an actual hardware measurement."""
        if not workload_key:
            raise ValueError("workload_key must not be empty")
        if analytical_latency_ms <= 0 or real_latency_ms <= 0:
            raise ValueError("latencies must be positive")
        calibration = self._calibrations.setdefault(workload_key, _Calibration())
        calibration.update(analytical_latency_ms, real_latency_ms)

    def real_sample_count(self, workload_key: str) -> int:
        """Return the number of real measurements used for calibration."""
        calibration = self._calibrations.get(workload_key)
        return calibration.count if calibration is not None else 0

    @staticmethod
    def _validate_estimate(estimate: CandidateEstimate) -> None:
        if not estimate.candidate_id or not estimate.workload_key:
            raise ValueError("candidate_id and workload_key must not be empty")
        if estimate.analytical_latency_ms <= 0:
            raise ValueError("analytical latency must be positive")
        if estimate.relative_uncertainty < 0:
            raise ValueError("relative uncertainty must be non-negative")
        if estimate.available_memory_bytes <= 0 or estimate.predicted_peak_memory_bytes < 0:
            raise ValueError("memory estimates must be non-negative and available memory positive")