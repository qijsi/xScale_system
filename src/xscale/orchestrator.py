# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""Trust-calibrated search orchestrator for xScale.

The orchestrator is the brain of xScale: it runs the main search loop,
manages the trust level, selects which candidates to evaluate and at
what fidelity, and decides when to stop.

Evolution happens at three levels:
1. Trust level: L0 (exploration) -> L1 (calibrating) -> L2 (confident) -> L3 (verifying)
2. Search policy: random / neighborhood -> Bayesian optimization -> verification
3. Fidelity policy: all REAL -> mixed fidelity -> minimal witness + top-K REAL
"""

from __future__ import annotations

import math
import time
import enum
from dataclasses import dataclass, field
from typing import Callable

from .skills.strategy_space import EvolvingStrategySpace, ParallelConfig, FeasibilityStatus
from .skills.cost_model import CostEstimate
from .skills.structure import WitnessPlan
from .skills.feedback import AdaptiveEvaluationController, EvaluationMode
from .skills.evaluator import ExecutionResult


class TrustLevel(enum.IntEnum):
    """Four trust levels defining the orchestrator operating regime."""

    L0_EXPLORATION = 0
    L1_CALIBRATING = 1
    L2_CONFIDENT = 2
    L3_VERIFYING = 3


@dataclass
class SearchHistory:
    """Record of all evaluated configurations."""

    results: list[ExecutionResult] = field(default_factory=list)
    best_latency_ms: float = float("inf")
    best_config: ParallelConfig | None = None

    def add(self, result: ExecutionResult) -> None:
        self.results.append(result)
        if result.success and not result.oom:
            if result.latency_ms < self.best_latency_ms:
                self.best_latency_ms = result.latency_ms
                self.best_config = result.config

    @property
    def count(self) -> int:
        return len(self.results)

    @property
    def feasible_count(self) -> int:
        return sum(1 for r in self.results if r.success and not r.oom)


@dataclass
class TrustTracker:
    """Tracks the system's trust in its analytical predictions.

    Trust is computed from three signals:
    (1) Historical prediction accuracy (how close predictions were to real)
    (2) Number of evaluated configurations (more data = more trust)
    (3) Diversity of observed configurations (more diverse = more trust)
    """

    trust: float = 0.0
    _prediction_errors: list[float] = field(default_factory=list)
    _seen_topology_classes: set[str] = field(default_factory=set)

    def update(
        self,
        config: ParallelConfig,
        predicted_ms: float,
        measured_ms: float,
    ) -> None:
        """Update trust with a new prediction-vs-real observation."""
        if predicted_ms <= 0 or measured_ms <= 0:
            return
        rel_error = abs(predicted_ms - measured_ms) / max(predicted_ms, measured_ms)
        self._prediction_errors.append(rel_error)
        self._seen_topology_classes.add(self._topo_key(config))
        self.trust = self._compute_trust()

    def _compute_trust(self) -> float:
        n = len(self._prediction_errors)
        if n == 0:
            return 0.0

        # Signal 1: accuracy (lower error = higher trust)
        avg_error = sum(self._prediction_errors) / n
        accuracy_component = max(0.0, 1.0 - avg_error)

        # Signal 2: sample count (more data = more confidence in accuracy)
        # Saturates at ~30 samples
        count_component = min(1.0, n / 30.0)

        # Signal 3: diversity (more topology classes = more general trust)
        diversity_component = min(1.0, len(self._seen_topology_classes) / 6.0)

        # Weighted combination: accuracy matters most, count calibrates it
        trust = 0.5 * accuracy_component + 0.3 * count_component + 0.2 * diversity_component
        return max(0.0, min(1.0, trust))

    def level(self) -> TrustLevel:
        if self.trust < 0.3:
            return TrustLevel.L0_EXPLORATION
        if self.trust < 0.6:
            return TrustLevel.L1_CALIBRATING
        if self.trust < 0.85:
            return TrustLevel.L2_CONFIDENT
        return TrustLevel.L3_VERIFYING

    @staticmethod
    def _topo_key(config: ParallelConfig) -> str:
        return f"tp{config.tp}_pp{config.pp}_ep{config.ep}"


class BayesianSurrogate:
    """Minimal Bayesian surrogate for ranking predictions.

    This is a lightweight stand-in for a full GP-based BO. It fits a simple
    per-dimension multiplicative correction model and returns uncertainty
    estimates. The purpose is not to be state-of-the-art BO but to
    demonstrate the orchestrator pattern; a full GP can be plugged in
    later.
    """

    def __init__(self):
        self._measurements: list[tuple[ParallelConfig, float]] = []
        self._correction_factor: float = 1.0
        self._uncertainty: float = 1.0

    def update(self, config: ParallelConfig, latency_ms: float) -> None:
        self._measurements.append((config, latency_ms))
        if len(self._measurements) >= 3:
            ratios = []
            for cfg, real in self._measurements[-10:]:
                est = self._analytical_estimate(cfg)
                if est > 0:
                    ratios.append(real / est)
            if ratios:
                self._correction_factor = sum(ratios) / len(ratios)
                self._uncertainty = max(0.05, 0.5 / math.sqrt(len(self._measurements)))

    def predict(self, config: ParallelConfig) -> tuple[float, float]:
        """Return (predicted_latency_ms, uncertainty)."""
        est = self._analytical_estimate(config) * self._correction_factor
        return est, self._uncertainty

    def expected_improvement(self, config: ParallelConfig, best_ms: float) -> float:
        """Simple expected improvement approximation."""
        pred, unc = self.predict(config)
        if best_ms == float("inf"):
            return 1.0
        improvement = max(0.0, (best_ms - pred) / max(best_ms, 1.0))
        exploration_bonus = unc * 0.5
        return improvement + exploration_bonus

    @staticmethod
    def _analytical_estimate(config: ParallelConfig) -> float:
        """Very rough analytical estimate used before real data arrives.

        Computes a relative latency score based on parallelism dimensions.
        Not accurate enough for ranking, only for coarse ordering.
        """
        tp_cost = config.tp * 0.5
        pp_bubble = (config.pp - 1) * 0.1
        ep_comm = math.log2(config.ep) * 0.3
        mbs_benefit = math.log2(config.mbs) * 0.15
        base = 100.0
        return base * (1.0 + tp_cost + pp_bubble + ep_comm - mbs_benefit)


class SearchOrchestrator:
    """Main orchestrator: runs the trust-calibrated search loop.

    The orchestrator evolves through four phases (L0-L3) as trust in
    analytical predictions grows. It coordinates the five skill modules
    and decides which candidates to evaluate at what fidelity.

    Usage:
        orch = SearchOrchestrator(model_desc, hw_desc, budget_gpu_hours=100)
        result = orch.run()
        print(result.best_config, result.best_latency_ms)
    """

    def __init__(
        self,
        model_desc,
        hardware_desc,
        budget_gpu_hours: float = 100.0,
        max_iterations: int = 200,
        top_k: int = 3,
    ):
        self.model_desc = model_desc
        self.hardware_desc = hardware_desc
        self.budget_gpu_hours = budget_gpu_hours
        self.max_iterations = max_iterations
        self.top_k = top_k

        # Skill modules
        self.space = EvolvingStrategySpace(model_desc, hardware_desc)
        self.trust = TrustTracker()
        self.history = SearchHistory()
        self.surrogate = BayesianSurrogate()

        # Internal state
        self._start_time: float | None = None
        self._iterations = 0
        self._gpu_hours_used = 0.0
        self._consecutive_no_improvement = 0

    def run(self, evaluator: Callable | None = None) -> SearchHistory:
        """Run the full search loop until budget exhausted or converged.

        Args:
            evaluator: Optional callable(config, fidelity_mode) -> ExecutionResult.
                If None, uses a simple analytical oracle for testing.

        Returns:
            SearchHistory with all results and best configuration.
        """
        self._start_time = time.time()

        # Phase 0: seed the space
        seeds = self.space.seed()
        seeds_to_eval = self._select_seed_candidates(seeds, n=min(10, len(seeds)))

        for config in seeds_to_eval:
            if self._budget_exhausted():
                break
            self._evaluate(config, EvaluationMode.REAL, evaluator)

        # Main loop: iterate until budget or convergence
        while not self._should_stop():
            self._iterations += 1

            # Select next candidate based on current trust level
            candidates = self._propose_candidates(top_n=5)
            if not candidates:
                break

            config = candidates[0]
            fidelity = self._choose_fidelity(config)
            self._evaluate(config, fidelity, evaluator)

        # Phase L3: verify top-K at full fidelity if we reached enough trust
        if self.trust.level() >= TrustLevel.L2_CONFIDENT:
            self._verify_top_k(evaluator)

        return self.history

    # -- candidate selection -------------------------------------------------

    def _select_seed_candidates(
        self, seeds: list[ParallelConfig], n: int
    ) -> list[ParallelConfig]:
        """Pick diverse seed configs spanning the topology space."""
        # Simple diversity: pick one per tp value, one per pp value
        selected = []
        seen_tp = set()
        seen_pp = set()
        for cfg in seeds:
            if len(selected) >= n:
                break
            if cfg.tp not in seen_tp or cfg.pp not in seen_pp:
                selected.append(cfg)
                seen_tp.add(cfg.tp)
                seen_pp.add(cfg.pp)
        # Fill rest with first available
        for cfg in seeds:
            if len(selected) >= n:
                break
            if cfg not in selected:
                selected.append(cfg)
        return selected

    def _propose_candidates(self, top_n: int = 5) -> list[ParallelConfig]:
        """Propose candidates to evaluate next, based on current trust level."""
        level = self.trust.level()
        best = self.history.best_config

        # Gather candidate pool from neighbors of top-performing configs
        pool: list[ParallelConfig] = []
        for cfg in self._top_feasible(5):
            neighbors = self.space.propose_neighbors(cfg)
            pool.extend(neighbors)
        pool = list({c for c in pool if self.space.status_of(c) not in (
            FeasibilityStatus.FEASIBLE, FeasibilityStatus.INFEASIBLE
        )})

        if not pool and best is not None:
            # If all neighbors explored, expand seed coverage
            pool = [
                c for c in self.space.seed()
                if self.space.status_of(c) == FeasibilityStatus.UNKNOWN
            ]

        if not pool:
            return []

        if level == TrustLevel.L0_EXPLORATION:
            # L0: random exploration for diverse data
            import random
            random.shuffle(pool)
            return pool[:top_n]

        if level == TrustLevel.L1_CALIBRATING:
            # L1: prefer configs with different topologies (calibrate broadly)
            pool.sort(key=lambda c: self._diversity_score(c), reverse=True)
            return pool[:top_n]

        # L2/L3: BO-driven selection
        best_ms = self.history.best_latency_ms
        scored = [(self.surrogate.expected_improvement(c, best_ms), c) for c in pool]
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:top_n]]

    def _diversity_score(self, config: ParallelConfig) -> float:
        """Score how different a config is from already-measured ones."""
        topo_key = TrustTracker._topo_key(config)
        if topo_key not in self.trust._seen_topology_classes:
            return 1.0
        # Prefer configs with MBS values we haven't seen
        seen_mbs = {r.config.mbs for r in self.history.results if self._same_topo(r.config, config)}
        if config.mbs not in seen_mbs:
            return 0.5
        return 0.1

    @staticmethod
    def _same_topo(a: ParallelConfig, b: ParallelConfig) -> bool:
        return a.tp == b.tp and a.pp == b.pp and a.ep == b.ep

    def _top_feasible(self, n: int) -> list[ParallelConfig]:
        """Return the n best feasible configs from history."""
        feasible = [r for r in self.history.results if r.success and not r.oom]
        feasible.sort(key=lambda r: r.latency_ms)
        return [r.config for r in feasible[:n]]

    # -- fidelity selection --------------------------------------------------

    def _choose_fidelity(self, config: ParallelConfig) -> EvaluationMode:
        """Choose evaluation fidelity based on trust and candidate importance."""
        level = self.trust.level()

        if level == TrustLevel.L0_EXPLORATION:
            return EvaluationMode.REAL

        if level == TrustLevel.L1_CALIBRATING:
            # Top candidates get real, others analytical
            pred, _ = self.surrogate.predict(config)
            if self.history.best_latency_ms == float("inf"):
                return EvaluationMode.REAL
            ratio = pred / self.history.best_latency_ms
            if ratio < 1.3:
                return EvaluationMode.REAL
            return EvaluationMode.ANALYTICAL

        # L2: most candidates analytical, only promising ones real
        ei = self.surrogate.expected_improvement(config, self.history.best_latency_ms)
        if ei > 0.1:
            return EvaluationMode.REAL
        return EvaluationMode.ANALYTICAL

    # -- evaluation wrapper --------------------------------------------------

    def _evaluate(
        self,
        config: ParallelConfig,
        fidelity: EvaluationMode,
        evaluator: Callable | None,
    ) -> ExecutionResult | None:
        """Evaluate a config and update all state."""
        if evaluator is None:
            result = self._mock_evaluator(config, fidelity)
        else:
            result = evaluator(config, fidelity)

        if result is None:
            return None

        # Update space state
        feasible = result.success and not result.oom
        reason = "success" if feasible else ("OOM" if result.oom else "failed")
        self.space.mark_result(config, feasible, reason)

        # Update history
        self.history.add(result)

        # Update trust and surrogate (only if real evaluation)
        if fidelity == EvaluationMode.REAL and result.success:
            pred, _ = self.surrogate.predict(config)
            self.trust.update(config, pred, result.latency_ms)
            self.surrogate.update(config, result.latency_ms)

        # Track budget
        self._gpu_hours_used += self._estimate_cost(config, fidelity)
        return result

    @staticmethod
    def _estimate_cost(config: ParallelConfig, fidelity: EvaluationMode) -> float:
        """Estimate GPU-hours consumed by one evaluation."""
        base = 0.01  # 0.01 GPU-hours base cost
        world = config.tp * config.pp * config.dp * config.ep
        if fidelity == EvaluationMode.ANALYTICAL:
            return base * 0.01
        if fidelity == EvaluationMode.REPLAY:
            return base * 0.3
        return base * (world / 64.0)

    # -- stopping and verification -------------------------------------------

    def _should_stop(self) -> bool:
        if self._iterations >= self.max_iterations:
            return True
        if self._budget_exhausted():
            return True
        if self._consecutive_no_improvement >= 20:
            return True
        return False

    def _budget_exhausted(self) -> bool:
        return self._gpu_hours_used >= self.budget_gpu_hours

    def _verify_top_k(self, evaluator: Callable | None) -> None:
        """Verify the top-K candidates at full REAL fidelity."""
        top_configs = self._top_feasible(self.top_k)
        for config in top_configs:
            self._evaluate(config, EvaluationMode.REAL, evaluator)

    # -- mock evaluator for testing -----------------------------------------

    @staticmethod
    def _mock_evaluator(config: ParallelConfig, fidelity: EvaluationMode) -> ExecutionResult:
        """Mock evaluator for testing the orchestrator loop.

        Produces noisy but realistic latency estimates based on config.
        """
        from .skills.evaluator import ExecutionResult as Result

        base = BayesianSurrogate._analytical_estimate(config)
        if fidelity == EvaluationMode.ANALYTICAL:
            noise = 0.3
            latency = base * (1.0 + (hash(str(config)) % 100 - 50) / 100.0 * noise)
        else:
            noise = 0.05
            latency = base * (1.0 + (hash(str(config) + "real") % 100 - 50) / 100.0 * noise)

        oom = config.mbs >= 16 and config.tp <= 2
        return Result(
            config=config,
            latency_ms=max(1.0, latency),
            peak_memory_gb=20.0 * config.mbs / config.tp,
            comm_volume_mb=config.tp * 100.0,
            oom=oom,
            success=not oom,
            mode=fidelity,
        )
