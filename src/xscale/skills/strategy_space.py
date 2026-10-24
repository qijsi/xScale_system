# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""S1 — Evolving Strategy Space.

Responsibility
--------------
Maintain a partial, evolving model of the parallel-strategy space. Start from
a provably-safe seed set; discover new legal strategies through neighborhood
exploration; prune infeasible regions from learned evidence.

Design notes
------------
- The space is never claimed to be complete. Coverage() reports how much has
  been evaluated, and the search engine decides when to stop.
- Infeasibility propagation is soft: "likely infeasible" candidates are
  deprioritized but never permanently removed.

Versions
--------
v0: seed generation (divisibility + memory check),
    1-neighborhood exploration,
    MBS-monotonic infeasibility propagation,
    coverage tracking.

v1: + communication feasibility (NVLink domain / cross-node bandwidth),
    + structural similarity neighbors (nearby PP stages share shape).

v2: + learned class stability (which equivalence classes survive transitions),
    + learned proposal policy (which neighbors tend to help),
    + cross-workload transfer via model-feature embeddings.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Types (defined in types.py, re-exported for backwards compatibility)
# ---------------------------------------------------------------------------

from .types import (
    FeasibilityStatus,
    HardwareDescriptor,
    ModelDescriptor,
    ParallelConfig,
    SpaceUpdate,
)

from . import heuristic_rules



# ---------------------------------------------------------------------------
# Space state
# ---------------------------------------------------------------------------


class EvolvingStrategySpace:
    """State + operations for the evolving strategy space.

    v0 implementation.
    """

    def __init__(self, model: ModelDescriptor, hardware: HardwareDescriptor):
        self.model = model
        self.hardware = hardware
        self._status: dict[ParallelConfig, FeasibilityStatus] = {}
        self._infeasibility_reason: dict[ParallelConfig, str] = {}

    # -- seeding -----------------------------------------------------------

    def seed(self) -> list[ParallelConfig]:
        """Generate initial seed set and populate the space state.

        Enumerates all topologically-valid configurations, filters with
        hard constraints (H1–H7), classifies by conservative memory
        estimate, and returns them sorted by heuristic priority so the
        most promising candidates are evaluated first.

        Returns the list of seed configs placed in LIKELY_FEASIBLE state
        (passed the conservative memory check) or LIKELY_INFEASIBLE state
        (failed the check, still retained for later evaluation).
        Configurations that violate hard constraints are not added at all.
        """
        world_size = self.hardware.num_gpus
        candidates: list[ParallelConfig] = []
        for tp in _divisors(world_size):
            rest = world_size // tp
            for pp in _divisors(rest):
                rest2 = rest // pp
                for ep in _divisors(rest2):
                    dp = rest2 // ep
                    if self.model.is_moe and ep < 2:
                        continue
                    if not self.model.is_moe and ep != 1:
                        continue
                    for mbs in (1, 2, 4, 8, 16):
                        for recompute in ("none", "selective", "full"):
                            cfg = ParallelConfig(tp, pp, dp, ep, mbs, recompute)
                            ok, reason = heuristic_rules.check_hard_constraints(
                                cfg, self.model, self.hardware
                            )
                            if not ok:
                                continue
                            candidates.append(cfg)

        # Sort by heuristic priority — most promising first
        candidates.sort(
            key=lambda c: heuristic_rules.heuristic_priority_score(
                c, self.model, self.hardware
            ),
            reverse=True,
        )

        for cfg in candidates:
            if self._conservative_memory_ok(cfg):
                self._status[cfg] = FeasibilityStatus.LIKELY_FEASIBLE
            else:
                self._status[cfg] = FeasibilityStatus.LIKELY_INFEASIBLE
                self._infeasibility_reason[cfg] = "conservative OOM estimate"

        return candidates

    # -- neighborhood ------------------------------------------------------

    def propose_neighbors(
        self, config: ParallelConfig, radius: int = 1
    ) -> list[ParallelConfig]:
        """Generate 1-step neighbors of a config.

        Only dimensions that preserve the world-size product are tried.
        Configs already in FEASIBLE or INFEASIBLE state are excluded.
        """
        world = config.world_size
        candidates: set[ParallelConfig] = set()

        # TP / PP / DP / EP: swap one unit between two dimensions
        dims = [config.tp, config.pp, config.dp, config.ep]
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                new_dims = list(dims)
                # try doubling one and halving the other
                if new_dims[i] % 2 == 0:
                    new_dims[i] //= 2
                    new_dims[j] *= 2
                    if new_dims[0] * new_dims[1] * new_dims[2] * new_dims[3] == world:
                        if self.model.is_moe and new_dims[3] < 2:
                            continue
                        if not self.model.is_moe and new_dims[3] != 1:
                            continue
                        for mbs in (config.mbs,):
                            for rc in (config.recompute,):
                                cand = ParallelConfig(new_dims[0], new_dims[1], new_dims[2], new_dims[3], mbs, rc)
                                candidates.add(cand)

        # MBS: double / halve
        for mbs_factor in (2, 0.5):
            new_mbs = int(config.mbs * mbs_factor)
            if new_mbs < 1:
                continue
            cand = ParallelConfig(config.tp, config.pp, config.dp, config.ep, new_mbs, config.recompute)
            candidates.add(cand)

        # Recompute mode: toggle
        for rc in ("none", "selective", "full"):
            if rc != config.recompute:
                cand = ParallelConfig(config.tp, config.pp, config.dp, config.ep, config.mbs, rc)
                candidates.add(cand)

        # Filter: hard constraints + not already decided
        result = []
        for c in candidates:
            ok, _ = heuristic_rules.check_hard_constraints(
                c, self.model, self.hardware
            )
            if not ok:
                continue
            if self._status.get(c, FeasibilityStatus.UNKNOWN) in (
                FeasibilityStatus.FEASIBLE,
                FeasibilityStatus.INFEASIBLE,
            ):
                continue
            result.append(c)

        # Sort by heuristic priority — most promising first
        result.sort(
            key=lambda c: heuristic_rules.heuristic_priority_score(
                c, self.model, self.hardware
            ),
            reverse=True,
        )
        return result

    # -- feedback ----------------------------------------------------------

    def estimate_witnesses(self, config: ParallelConfig) -> int:
        """Estimate minimum witness ranks needed to evaluate ``config``.

        Uses structural symmetry rules (S1–S4) to compute how many
        unique rank types must run at full fidelity. Higher symmetry →
        fewer witnesses → cheaper evaluation.
        """
        breakdown = heuristic_rules.estimate_witness_count(
            config, self.model, self.hardware
        )
        return breakdown.total_witnesses

    def mark_result(
        self, config: ParallelConfig, feasible: bool, reason: str = ""
    ) -> SpaceUpdate:
        """Record an execution outcome and propagate infeasibility if OOM.

        Returns the set of changes so the caller can log them.
        """
        newly_feasible: list[ParallelConfig] = []
        newly_infeasible: list[ParallelConfig] = []
        propagated: list[ParallelConfig] = []

        old = self._status.get(config, FeasibilityStatus.UNKNOWN)
        if feasible:
            if old is not FeasibilityStatus.FEASIBLE:
                self._status[config] = FeasibilityStatus.FEASIBLE
                newly_feasible.append(config)
        else:
            if old is not FeasibilityStatus.INFEASIBLE:
                self._status[config] = FeasibilityStatus.INFEASIBLE
                self._infeasibility_reason[config] = reason
                newly_infeasible.append(config)
            # Soft propagation: same topology, larger MBS -> likely infeasible
            propagated = self._propagate_mbs_infeasibility(config)

        return SpaceUpdate(
            newly_feasible=newly_feasible,
            newly_infeasible=newly_infeasible,
            propagated_infeasible=propagated,
            new_coverage=self.coverage(),
        )

    # -- queries -----------------------------------------------------------

    def feasible_region(self) -> list[ParallelConfig]:
        return [c for c, s in self._status.items() if s is FeasibilityStatus.FEASIBLE]

    def infeasible_reason(self, config: ParallelConfig) -> str | None:
        return self._infeasibility_reason.get(config)

    def coverage(self) -> float:
        """Fraction of total estimated configs that have been decided."""
        if not self._status:
            return 0.0
        decided = sum(
            1 for s in self._status.values()
            if s in (FeasibilityStatus.FEASIBLE, FeasibilityStatus.INFEASIBLE)
        )
        return decided / len(self._status)

    def status_of(self, config: ParallelConfig) -> FeasibilityStatus:
        return self._status.get(config, FeasibilityStatus.UNKNOWN)

    # -- internals ---------------------------------------------------------

    def _conservative_memory_ok(self, cfg: ParallelConfig) -> bool:
        """Very rough analytical memory check.

        Only used for seed-time filtering. Real OOM detection comes from
        execution feedback.
        """
        # Simplified: params + optimizer states + per-microbatch activations
        params_gb = self.model.params_billion * 2 * 2 / cfg.tp  # bf16 + grad, reduced by TP
        per_layer_act = (
            cfg.mbs * self.model.sequence_length * self.model.hidden_size
            * 4  # rough multiplier for activations per layer
        ) / (1024 ** 3)  # bytes -> GB, bf16 so * 2 bytes... very rough
        act_gb = self.model.num_layers * per_layer_act / cfg.tp / cfg.pp
        total_gb = params_gb * 4 + act_gb  # 4x for optimizer states
        return total_gb < self.hardware.vram_per_gpu_gb * 0.9

    def _propagate_mbs_infeasibility(self, oom_config: ParallelConfig) -> list[ParallelConfig]:
        """Mark same-topology configs with larger MBS as likely infeasible."""
        propagated: list[ParallelConfig] = []
        for cfg, status in list(self._status.items()):
            if cfg.tp != oom_config.tp:
                continue
            if cfg.pp != oom_config.pp or cfg.dp != oom_config.dp or cfg.ep != oom_config.ep:
                continue
            if cfg.recompute != oom_config.recompute:
                continue
            if cfg.mbs <= oom_config.mbs:
                continue
            if status in (FeasibilityStatus.FEASIBLE, FeasibilityStatus.INFEASIBLE):
                continue
            self._status[cfg] = FeasibilityStatus.LIKELY_INFEASIBLE
            self._infeasibility_reason[cfg] = (
                f"propagated from OOM at MBS={oom_config.mbs}"
            )
            propagated.append(cfg)
        return propagated


def _divisors(n: int) -> list[int]:
    """Return all positive divisors of n, sorted."""
    divs: list[int] = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)
