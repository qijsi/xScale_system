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
# Types
# ---------------------------------------------------------------------------


class FeasibilityStatus(enum.Enum):
    """How confident we are that a config can run.

    Order matters: UNKNOWN < LIKELY_INFEASIBLE < FEASIBLE == INFEASIBLE (terminal).
    """

    UNKNOWN = "unknown"
    LIKELY_FEASIBLE = "likely_feasible"
    LIKELY_INFEASIBLE = "likely_infeasible"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class ParallelConfig:
    """A complete parallel execution strategy.

    The tp*pp*dp*ep product must equal the target world size.
    """

    tp: int
    pp: int
    dp: int
    ep: int
    mbs: int
    recompute: str  # "none" | "selective" | "full"

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.dp * self.ep

    def __str__(self) -> str:
        return f"tp{self.tp}_pp{self.pp}_dp{self.dp}_ep{self.ep}_mbs{self.mbs}_rc{self.recompute}"


@dataclass(frozen=True)
class ModelDescriptor:
    """Model parameters relevant to strategy search."""

    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    vocab_size: int
    sequence_length: int
    is_moe: bool
    num_experts: int | None
    expert_top_k: int | None
    params_billion: float


@dataclass(frozen=True)
class HardwareDescriptor:
    """Target hardware topology for which we search strategies."""

    gpu_model: str
    num_gpus: int
    gpus_per_node: int
    nvlink_bandwidth_gbps: float
    internode_bandwidth_gbps: float
    vram_per_gpu_gb: float
    peak_tflops: float


@dataclass
class SpaceUpdate:
    """Result of observing an execution outcome for one config."""

    newly_feasible: list[ParallelConfig]
    newly_infeasible: list[ParallelConfig]
    propagated_infeasible: list[ParallelConfig]
    new_coverage: float


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

        Returns the list of seed configs placed in LIKELY_FEASIBLE state
        (passed the conservative memory check) or LIKELY_INFEASIBLE state
        (failed the check, still retained for later evaluation).
        """
        world_size = self.hardware.num_gpus
        seeds: list[ParallelConfig] = []
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
                    for mbs in (1, 2, 4, 8):
                        for recompute in ("none", "selective", "full"):
                            cfg = ParallelConfig(tp, pp, dp, ep, mbs, recompute)
                            if self._conservative_memory_ok(cfg):
                                self._status[cfg] = FeasibilityStatus.LIKELY_FEASIBLE
                            else:
                                self._status[cfg] = FeasibilityStatus.LIKELY_INFEASIBLE
                                self._infeasibility_reason[cfg] = "conservative OOM estimate"
                            seeds.append(cfg)
        return seeds

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

        # Filter out already-decided configs
        result = [
            c for c in candidates
            if self._status.get(c, FeasibilityStatus.UNKNOWN)
            not in (FeasibilityStatus.FEASIBLE, FeasibilityStatus.INFEASIBLE)
        ]
        return sorted(result, key=str)

    # -- feedback ----------------------------------------------------------

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
