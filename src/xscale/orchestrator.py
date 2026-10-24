# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""Trust-calibrated search orchestrator for xScale.

The orchestrator runs the main search loop, managing trust level,
candidate selection, and fidelity control.

Design principle: start with heuristic rules, gradually upgrade to stronger
search methods as data accumulates. BO is a pluggable option for the L2
phase, not a core requirement.

Search policy evolution:
  L0 Exploration  : heuristic — diverse seed + random neighbors
  L1 Calibrating  : heuristic — topology-diverse neighborhood search
  L2 Confident    : optional BO (EI) with heuristic fallback
  L3 Verifying    : heuristic — top-K full-fidelity verification
"""

from __future__ import annotations

import math
import time
import random
import enum
from dataclasses import dataclass, field
from typing import Callable

from .skills.strategy_space import EvolvingStrategySpace, ParallelConfig, FeasibilityStatus
from .skills.feedback import EvaluationMode
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
    (1) Historical prediction accuracy
    (2) Number of evaluated configurations
    (3) Diversity of observed topology classes
    """

    trust: float = 0.0
    _prediction_errors: list[float] = field(default_factory=list)
    _seen_topology_classes: set[str] = field(default_factory=set)

    def update(self, config: ParallelConfig, predicted_ms: float, measured_ms: float) -> None:
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

        # Signal 2: sample count — saturates at ~30
        count_component = min(1.0, n / 30.0)

        # Signal 3: diversity — more classes = more general trust
        diversity_component = min(1.0, len(self._seen_topology_classes) / 6.0)

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


# ============================================================================
# Search policies: heuristic-first, BO as upgrade
# ============================================================================

class HeuristicSearchPolicy:
    """Rule-based search policy — always available, no ML needed.

    Three modes matching the three active search phases:
    - explore  : pick diverse seeds, random neighbors
    - calibrate: prefer topology-diverse candidates
    - exploit  : pick neighbors of the best config
    """

    def __init__(self, space: EvolvingStrategySpace, history: SearchHistory):
        self.space = space
        self.history = history

    def propose(self, mode: str, top_n: int = 5) -> list[ParallelConfig]:
        """Propose candidates according to the current mode.

        Args:
            mode: 'explore', 'calibrate', or 'exploit'
            top_n: number of candidates to return
        """
        pool = self._candidate_pool()
        if not pool:
            return []

        if mode == "explore":
            return self._policy_explore(pool, top_n)
        if mode == "calibrate":
            return self._policy_calibrate(pool, top_n)
        # exploit
        return self._policy_exploit(pool, top_n)

    def _candidate_pool(self) -> list[ParallelConfig]:
        """Gather all unevaluated neighbors of top feasible configs."""
        pool: set[ParallelConfig] = set()
        for cfg in self._top_feasible(10):
            neighbors = self.space.propose_neighbors(cfg)
            pool.update(neighbors)
        # Filter out already evaluated
        result = [
            c for c in pool
            if self.space.status_of(c) not in (FeasibilityStatus.FEASIBLE, FeasibilityStatus.INFEASIBLE)
        ]
        # If all neighbors explored, fall back to unexplored seeds
        if not result:
            all_seeds = self.space.seed()
            result = [c for c in all_seeds if self.space.status_of(c) == FeasibilityStatus.UNKNOWN]
        return result

    def _policy_explore(self, pool: list[ParallelConfig], top_n: int) -> list[ParallelConfig]:
        """Random selection — maximizes diversity of initial data."""
        shuffled = list(pool)
        random.shuffle(shuffled)
        return shuffled[:top_n]

    def _policy_calibrate(self, pool: list[ParallelConfig], top_n: int) -> list[ParallelConfig]:
        """Prefer topologically diverse candidates — learns broad corrections fast."""
        # Group by topology class, pick one per class
        by_topo: dict[str, list[ParallelConfig]] = {}
        for c in pool:
            key = TrustTracker._topo_key(c)
            by_topo.setdefault(key, []).append(c)

        selected: list[ParallelConfig] = []
        # First round: one from each new topology class
        seen_classes = self._seen_topology_classes()
        for key, candidates in sorted(by_topo.items()):
            if key not in seen_classes and candidates:
                selected.append(candidates[0])
                if len(selected) >= top_n:
                    return selected
        # Fill rest with anything
        for candidates in by_topo.values():
            for c in candidates:
                if c not in selected:
                    selected.append(c)
                    if len(selected) >= top_n:
                        return selected
        return selected

    def _policy_exploit(self, pool: list[ParallelConfig], top_n: int) -> list[ParallelConfig]:
        """Focus on neighbors of the best configs — hill-climbing style."""
        # Prefer candidates with best analytical estimate
        # (for now just pick neighbors of top configs in order)
        selected: list[ParallelConfig] = []
        for cfg in self._top_feasible(10):
            for n in self.space.propose_neighbors(cfg):
                if n in pool and n not in selected:
                    selected.append(n)
                    if len(selected) >= top_n:
                        return selected
        return selected

    def _top_feasible(self, n: int) -> list[ParallelConfig]:
        feasible = [r for r in self.history.results if r.success and not r.oom]
        feasible.sort(key=lambda r: r.latency_ms)
        return [r.config for r in feasible[:n]]

    def _seen_topology_classes(self) -> set[str]:
        return {TrustTracker._topo_key(r.config) for r in self.history.results}


class BayesianSearchPolicy:
    """BO-based search policy — upgrade for L2 phase when data is sufficient.

    Wraps a surrogate model (currently a simple multiplicative correction
    with uncertainty; replaceable with full GP). Uses Expected Improvement
    for candidate selection.
    """

    def __init__(self, history: SearchHistory):
        self.history = history
        self._correction_factor = 1.0
        self._uncertainty = 1.0
        self._fitted = False

    def fit(self) -> None:
        """Fit surrogate from history. Called before proposing."""
        if self.history.feasible_count < 3:
            return
        ratios = []
        for r in self.history.results[-20:]:
            if not r.success or r.oom:
                continue
            est = self._analytical_estimate(r.config)
            if est > 0:
                ratios.append(r.latency_ms / est)
        if ratios:
            self._correction_factor = sum(ratios) / len(ratios)
            self._uncertainty = max(0.05, 0.5 / math.sqrt(self.history.feasible_count))
            self._fitted = True

    def predict(self, config: ParallelConfig) -> tuple[float, float]:
        est = self._analytical_estimate(config) * self._correction_factor
        return est, self._uncertainty

    def expected_improvement(self, config: ParallelConfig, best_ms: float) -> float:
        """Simple EI approximation."""
        if not self._fitted:
            return 0.5  # neutral
        pred, unc = self.predict(config)
        if best_ms == float("inf"):
            return 1.0
        improvement = max(0.0, (best_ms - pred) / max(best_ms, 1.0))
        exploration_bonus = unc * 0.5
        return improvement + exploration_bonus

    def rank(self, candidates: list[ParallelConfig], best_ms: float) -> list[ParallelConfig]:
        self.fit()
        scored = [(self.expected_improvement(c, best_ms), c) for c in candidates]
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored]

    @staticmethod
    def _analytical_estimate(config: ParallelConfig) -> float:
        """Rough analytical estimate used as base for correction."""
        tp_cost = config.tp * 0.5
        pp_bubble = (config.pp - 1) * 0.1
        ep_comm = math.log2(config.ep) * 0.3
        mbs_benefit = math.log2(config.mbs) * 0.15
        return 100.0 * (1.0 + tp_cost + pp_bubble + ep_comm - mbs_benefit)


# ============================================================================
# Orchestrator
# ============================================================================

class SearchOrchestrator:
    """Trust-calibrated search orchestrator.

    Evolves through four phases (L0-L3) as trust grows. Uses heuristic
    search policies by default; Bayesian optimization is an optional
    upgrade for the L2 phase.
    """

    def __init__(
        self,
        model_desc,
        hardware_desc,
        budget_gpu_hours: float = 100.0,
        max_iterations: int = 200,
        top_k: int = 3,
        use_bo: bool = True,
    ):
        self.model_desc = model_desc
        self.hardware_desc = hardware_desc
        self.budget_gpu_hours = budget_gpu_hours
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.use_bo = use_bo

        # Core state
        self.space = EvolvingStrategySpace(model_desc, hardware_desc)
        self.trust = TrustTracker()
        self.history = SearchHistory()
        self.heuristic = HeuristicSearchPolicy(self.space, self.history)
        self.bo = BayesianSearchPolicy(self.history)

        # Internal counters
        self._start_time: float | None = None
        self._iterations = 0
        self._gpu_hours_used = 0.0
        self._consecutive_no_improvement = 0

    def run(self, evaluator: Callable | None = None) -> SearchHistory:
        """Run the full search loop until budget exhausted or converged."""
        self._start_time = time.time()

        # --- Phase 0: seed evaluation ---
        seeds = self.space.seed()
        seeds_to_eval = self._pick_diverse_seeds(seeds, n=min(10, len(seeds)))
        for cfg in seeds_to_eval:
            if self._budget_exhausted():
                break
            self._evaluate(cfg, EvaluationMode.REAL, evaluator)

        # --- Main loop ---
        while not self._should_stop():
            self._iterations += 1
            level = self.trust.level()

            # Select candidates
            candidates = self._propose_candidates(top_n=5)
            if not candidates:
                break

            cfg = candidates[0]
            fidelity = self._choose_fidelity(cfg, level)
            result = self._evaluate(cfg, fidelity, evaluator)

            if result and result.success and not result.oom:
                if result.latency_ms < self.history.best_latency_ms - 0.01:
                    self._consecutive_no_improvement = 0
                else:
                    self._consecutive_no_improvement += 1

        # --- Phase L3: top-K verification ---
        if self.trust.level() >= TrustLevel.L2_CONFIDENT:
            self._verify_top_k(evaluator)

        return self.history

    # -- candidate selection ------------------------------------------------

    def _pick_diverse_seeds(self, seeds: list[ParallelConfig], n: int) -> list[ParallelConfig]:
        """Pick diverse seeds spanning different topology classes."""
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
        # Fill rest
        for cfg in seeds:
            if len(selected) >= n:
                break
            if cfg not in selected:
                selected.append(cfg)
        return selected

    def _propose_candidates(self, top_n: int = 5) -> list[ParallelConfig]:
        """Propose next candidates based on current trust level."""
        level = self.trust.level()

        if level == TrustLevel.L0_EXPLORATION:
            return self.heuristic.propose("explore", top_n)

        if level == TrustLevel.L1_CALIBRATING:
            return self.heuristic.propose("calibrate", top_n)

        if level == TrustLevel.L2_CONFIDENT:
            pool = self.heuristic._candidate_pool()
            if not pool:
                return []
            if self.use_bo and self.history.feasible_count >= 5:
                # Upgrade to BO if enough data
                return self.bo.rank(pool, self.history.best_latency_ms)[:top_n]
            # Fall back to heuristic exploit
            return self.heuristic.propose("exploit", top_n)

        # L3: still propose, but we focus on top-K verification
        return self.heuristic.propose("exploit", top_n)

    # -- fidelity selection --------------------------------------------------

    def _choose_fidelity(self, config: ParallelConfig, level: TrustLevel) -> EvaluationMode:
        """Choose evaluation fidelity based on trust and candidate value."""
        if level == TrustLevel.L0_EXPLORATION:
            return EvaluationMode.REAL

        if level == TrustLevel.L1_CALIBRATING:
            # Only run top candidates at REAL, skip rest via analytical
            if self._is_contender(config):
                return EvaluationMode.REAL
            return EvaluationMode.ANALYTICAL

        if level == TrustLevel.L2_CONFIDENT:
            if self.use_bo:
                ei = self.bo.expected_improvement(config, self.history.best_latency_ms)
                if ei > 0.1:
                    return EvaluationMode.REAL
                return EvaluationMode.ANALYTICAL
            # Heuristic: only run if close to best
            if self._is_contender(config):
                return EvaluationMode.REAL
            return EvaluationMode.ANALYTICAL

        # L3: full REAL for top-K, skip rest
        return EvaluationMode.ANALYTICAL

    def _is_contender(self, config: ParallelConfig) -> bool:
        """Heuristic check: is this config plausibly competitive?"""
        if self.history.best_latency_ms == float("inf"):
            return True
        # Compare analytical estimates
        est = BayesianSearchPolicy._analytical_estimate(config)
        best_est = BayesianSearchPolicy._analytical_estimate(self.history.best_config) \
            if self.history.best_config else est
        return est < best_est * 1.5

    # -- evaluation wrapper --------------------------------------------------

    def _evaluate(
        self,
        config: ParallelConfig,
        fidelity: EvaluationMode,
        evaluator: Callable | None,
    ) -> ExecutionResult | None:
        if evaluator is None:
            result = self._mock_evaluator(config, fidelity)
        else:
            result = evaluator(config, fidelity)

        if result is None:
            return None

        # Update space
        feasible = result.success and not result.oom
        reason = "success" if feasible else ("OOM" if result.oom else "failed")
        self.space.mark_result(config, feasible, reason)

        # Update history
        self.history.add(result)

        # Update trust and BO surrogate (only from real evaluations)
        if fidelity == EvaluationMode.REAL and result.success:
            pred, _ = self.bo.predict(config)
            self.trust.update(config, pred, result.latency_ms)
            self.bo.fit()

        # Track budget
        self._gpu_hours_used += self._estimate_cost(config, fidelity)
        return result

    @staticmethod
    def _estimate_cost(config: ParallelConfig, fidelity: EvaluationMode) -> float:
        base = 0.01  # GPU-hours
        world = config.tp * config.pp * config.dp * config.ep
        if fidelity == EvaluationMode.ANALYTICAL:
            return base * 0.001
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
        """Verify top-K candidates at full REAL fidelity."""
        top = self.heuristic._top_feasible(self.top_k)
        for cfg in top:
            self._evaluate(cfg, EvaluationMode.REAL, evaluator)

    # -- mock evaluator for testing -----------------------------------------

    @staticmethod
    def _mock_evaluator(config: ParallelConfig, fidelity: EvaluationMode) -> ExecutionResult:
        """Mock evaluator for testing the loop without real hardware."""
        base = BayesianSearchPolicy._analytical_estimate(config)
        if fidelity == EvaluationMode.ANALYTICAL:
            noise = 0.3
        else:
            noise = 0.05
        latency = base * (1.0 + (hash(str(config)) % 100 - 50) / 100.0 * noise)

        oom = config.mbs >= 16 and config.tp <= 2
        return ExecutionResult(
            config=config,
            latency_ms=max(1.0, latency),
            peak_memory_gb=20.0 * config.mbs / config.tp,
            comm_volume_mb=config.tp * 100.0,
            oom=oom,
            success=not oom,
            mode=fidelity,
        )
