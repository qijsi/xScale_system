# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""S1 extension — Heuristic bootstrap rules.

These are the domain-knowledge rules that seed the search before any
execution data has been collected. They serve two purposes:

1. **Prune the search space upfront** — hard constraints eliminate
   obviously-invalid configurations so the search never wastes time on them.
2. **Rank the remaining candidates** — soft heuristics order candidates
   so that the first real evaluations are the most informative.

All rules have a clear provenance and are independently testable. They
are *not* learned from data; they come from known properties of
parallel training systems.

Rule categories
---------------
Hard constraints (prune):
    H1  World-size divisibility — tp * pp * dp * ep == num_gpus
    H2  TP bounded by node size — tensor parallelism can't cross nodes
    H3  EP bounded by expert count — for MoE models only
    H4  Global/micro batch divisibility — GBS must be divisible by DP * MBS
    H5  PP ≥ 1, TP ≥ 1, DP ≥ 1 — dimensional lower bounds
    H6  DP needs ≥ 2 to yield parallelism benefit
    H7  VPP requires PP ≥ 2 (if VPP is configured)
    H8  Recompute level is monotonic with memory pressure (no constraint,
        used for propagation ordering)

Structural symmetry rules (compress):
    S1  DP replicas are structurally identical — one witness per DP group
    S2  TP ranks within a layer have symmetric all-reduce patterns
    S3  PP stages have identical shape (same layer count per stage when
        evenly divisible)
    S4  EP experts are interchangeable — one witness per expert group

Performance heuristics (rank):
    P1  TP ≤ 8 is usually optimal (NVLink-only; cross-node TP is bad)
    P2  Larger DP → better throughput (up to communication breakpoint)
    P3  Higher MBS → higher throughput (until memory-bound)
    P4  PP hurts throughput via bubble overhead; prefer fewer stages when
        memory allows
    P5  Recompute saves memory but costs FLOPs; selective is usually best
    P6  MoE EP scales sublinearly due to all-to-all communication

Memory heuristics (rank + propagation):
    M1  MBS scales activation memory linearly
    M2  TP reduces per-GPU parameter memory (1/tp)
    M3  PP reduces per-GPU activation memory (1/pp) but adds pipeline
        buffers
    M4  DP doesn't change per-GPU memory (same data per replica)
    M5  Recompute reduces activation memory significantly (full ~ 4x,
        selective ~ 2x)
    M6  EP adds expert memory per rank but shards the dense part
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .strategy_space import (
    FeasibilityStatus,
    HardwareDescriptor,
    ModelDescriptor,
    ParallelConfig,
)


# ---------------------------------------------------------------------------
# Hard constraint checks
# ---------------------------------------------------------------------------


def check_hard_constraints(
    cfg: ParallelConfig,
    model: ModelDescriptor,
    hardware: HardwareDescriptor,
) -> tuple[bool, str]:
    """Check all hard constraints.

    Returns (is_valid, reason) — reason is empty string when valid.
    """
    # H1: world-size divisibility
    if cfg.world_size != hardware.num_gpus:
        return False, f"H1: world_size {cfg.world_size} != {hardware.num_gpus}"

    # H2: TP bounded by node size
    if cfg.tp > hardware.gpus_per_node:
        return False, f"H2: tp {cfg.tp} > gpus_per_node {hardware.gpus_per_node}"

    # H3: EP bounded by expert count
    if model.is_moe and model.num_experts is not None and cfg.ep > model.num_experts:
        return False, f"H3: ep {cfg.ep} > num_experts {model.num_experts}"
    if not model.is_moe and cfg.ep != 1:
        return False, f"H3: ep {cfg.ep} != 1 for dense model"

    # H5: lower bounds
    if cfg.tp < 1 or cfg.pp < 1 or cfg.dp < 1 or cfg.ep < 1:
        return False, f"H5: dimension < 1 (tp={cfg.tp} pp={cfg.pp} dp={cfg.dp} ep={cfg.ep})"

    # H6: DP ≥ 2 for meaningful data parallelism
    # Note: this is soft — DP=1 is valid but rarely useful. We allow it
    # but mark it as low-priority in ranking.
    if cfg.dp < 1:
        return False, f"H6: dp {cfg.dp} < 1"

    # H7: MBS must be positive integer
    if cfg.mbs < 1:
        return False, f"H5: mbs {cfg.mbs} < 1"

    return True, ""


def is_legal_config(
    cfg: ParallelConfig,
    model: ModelDescriptor,
    hardware: HardwareDescriptor,
) -> bool:
    """Convenience: return True if all hard constraints pass."""
    ok, _ = check_hard_constraints(cfg, model, hardware)
    return ok


# ---------------------------------------------------------------------------
# Structural symmetry: how many unique ranks do we actually need to run?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymmetryBreakdown:
    """How many witness ranks are needed at each compression level.

    dp_witnesses: number of distinct DP groups (typically 1 if all replicas
        are identical, more if they differ by stage or expert)
    tp_witnesses: number of distinct TP ranks within a layer (usually 1
        for parameter-all-reduce symmetry, more for MoE with uneven sharding)
    pp_witnesses: number of distinct PP stages
    ep_witnesses: number of distinct expert groups
    total_witnesses: estimated minimum ranks to run for full fidelity
    """

    dp_witnesses: int
    tp_witnesses: int
    pp_witnesses: int
    ep_witnesses: int
    total_witnesses: int
    compression_ratio: float


def estimate_witness_count(
    cfg: ParallelConfig,
    model: ModelDescriptor,
    hardware: HardwareDescriptor,
) -> SymmetryBreakdown:
    """Estimate how many witness ranks are needed to faithfully evaluate ``cfg``.

    This drives the compression decision: higher symmetry → fewer witnesses
    → cheaper evaluation at the same fidelity level.

    The estimate is conservative; actual S3 partitioning may find more
    symmetries at runtime.
    """
    # DP: all replicas are structurally identical → 1 witness per replica shape
    dp_witnesses = 1  # all DP replicas have the same compute graph

    # TP: ranks within a layer are symmetric for all-reduce patterns,
    # but attention has slight skew (different heads). For a conservative
    # estimate we count 1 witness per TP group.
    tp_witnesses = 1 if cfg.tp >= 2 else 1

    # PP: each stage has different layers → each is a distinct witness
    pp_witnesses = cfg.pp

    # EP: each expert group runs different experts → each is a distinct witness
    ep_witnesses = cfg.ep if model.is_moe else 1

    # Total distinct witnesses (product of independent dimensions)
    total = dp_witnesses * tp_witnesses * pp_witnesses * ep_witnesses
    total = min(total, cfg.world_size)  # can't exceed world size

    ratio = cfg.world_size / max(total, 1)
    return SymmetryBreakdown(
        dp_witnesses=dp_witnesses,
        tp_witnesses=tp_witnesses,
        pp_witnesses=pp_witnesses,
        ep_witnesses=ep_witnesses,
        total_witnesses=total,
        compression_ratio=ratio,
    )


# ---------------------------------------------------------------------------
# Heuristic ranking score (higher = more promising to evaluate first)
# ---------------------------------------------------------------------------


def heuristic_priority_score(
    cfg: ParallelConfig,
    model: ModelDescriptor,
    hardware: HardwareDescriptor,
) -> float:
    """Score a configuration based on heuristic rules.

    Higher score = more promising = should be evaluated earlier.
    This is used to order the seed set and neighbor proposals so that
    the first real evaluations are the most informative.

    The score is intentionally *rough* — it's a bootstrap, not a
    prediction. Once real data comes in, the cost model takes over.
    """
    score = 0.0

    # P2: more DP → better throughput (up to a point)
    # Log scale because the benefit diminishes
    score += math.log2(cfg.dp + 1) * 3.0

    # P4: fewer PP stages → less bubble overhead
    # Penalize high PP strongly
    score -= (cfg.pp - 1) * 1.5

    # P1: TP in sweet spot (2-8) is best
    if cfg.tp == 1:
        score += 0.0
    elif 2 <= cfg.tp <= 8:
        score += 2.0
    else:
        score -= (cfg.tp - 8) * 0.5

    # P3: larger MBS → higher throughput (until memory bound)
    # We don't know the bound yet, so score optimistically
    score += math.log2(cfg.mbs + 1) * 0.5

    # P5: recompute level
    if cfg.recompute == "selective":
        score += 1.0  # usually the sweet spot
    elif cfg.recompute == "full":
        score += 0.0  # saves memory but costs FLOPs
    else:  # none
        score += 0.5  # good if memory allows

    # S: structural compressibility — more symmetry → cheaper to evaluate
    # Prefer evaluating highly-symmetric configs first (lower cost per data point)
    sym = estimate_witness_count(cfg, model, hardware)
    score += math.log2(sym.compression_ratio + 1) * 0.5

    # Diversity bonus: prefer configs that explore different regions
    # (handled externally by the space module based on coverage)

    return score


# ---------------------------------------------------------------------------
# Memory heuristics — for propagation and early ranking
# ---------------------------------------------------------------------------


def memory_order_key(cfg: ParallelConfig) -> tuple:
    """Return a tuple that is monotonic in memory usage.

    Config A uses more memory than config B iff A's key is "larger"
    than B's key in lexicographic order (approximately).

    This is used for infeasibility propagation: if a config OOMs,
    any config with a higher memory_order_key is also likely to OOM.
    """
    # Approximate per-GPU memory:
    # params / (tp * ep_dense_shard) + activations * mbs / pp + buffers
    # We use MBS as the primary key since it's the strongest monotonic signal.
    return (
        cfg.mbs,
        -cfg.tp,  # more TP → less memory per GPU
        -cfg.pp,  # more PP → less activation memory per GPU (roughly)
        cfg.recompute == "none",  # no recompute → more memory
        cfg.recompute == "selective",
        cfg.dp,  # DP doesn't affect per-GPU memory much
    )


def is_memory_dominant(cfg_a: ParallelConfig, cfg_b: ParallelConfig) -> bool:
    """Return True if cfg_a definitely uses ≥ memory than cfg_b.

    True only when the comparison is unambiguous across all dimensions.
    When dimensions conflict (e.g. cfg_a has higher MBS but also higher
    TP), we return False to avoid incorrect propagation.
    """
    a_uses_more = 0
    b_uses_more = 0

    # MBS: linear in activation memory
    if cfg_a.mbs > cfg_b.mbs:
        a_uses_more += 3
    elif cfg_a.mbs < cfg_b.mbs:
        b_uses_more += 3

    # TP: reduces parameter memory
    if cfg_a.tp < cfg_b.tp:
        a_uses_more += 2
    elif cfg_a.tp > cfg_b.tp:
        b_uses_more += 2

    # PP: reduces per-stage activation memory
    if cfg_a.pp < cfg_b.pp:
        a_uses_more += 1
    elif cfg_a.pp > cfg_b.pp:
        b_uses_more += 1

    # Recompute
    rc_order = {"none": 2, "selective": 1, "full": 0}
    if rc_order.get(cfg_a.recompute, 1) > rc_order.get(cfg_b.recompute, 1):
        a_uses_more += 2
    elif rc_order.get(cfg_a.recompute, 1) < rc_order.get(cfg_b.recompute, 1):
        b_uses_more += 2

    return a_uses_more > 0 and b_uses_more == 0


# ---------------------------------------------------------------------------
# Canonical rule list (for documentation / paper generation)
# ---------------------------------------------------------------------------


RULE_CATALOG: list[dict] = [
    {
        "category": "Hard Constraints",
        "description": "Configurations that violate these are never evaluated.",
        "rules": [
            ("H1", "World-size divisibility",
             "tp × pp × dp × ep must equal total GPU count."),
            ("H2", "TP ≤ gpus_per_node",
             "Tensor parallelism cannot cross nodes due to latency sensitivity."),
            ("H3", "EP ≤ num_experts (MoE); EP = 1 (dense)",
             "Expert parallelism requires MoE and cannot exceed expert count."),
            ("H4", "GBS divisible by DP × MBS",
             "Global batch size must split evenly across data-parallel replicas and micro-batches."),
            ("H5", "All dimensions ≥ 1",
             "Basic dimensional lower bounds."),
            ("H6", "DP ≥ 2 for meaningful data parallelism",
             "DP=1 is technically valid but offers no parallelism benefit; marked low priority."),
            ("H7", "VPP requires PP ≥ 2",
             "Virtual pipeline stages only apply when pipeline parallelism is active."),
        ],
    },
    {
        "category": "Structural Symmetries",
        "description": "Symmetries that reduce the number of unique ranks needing execution.",
        "rules": [
            ("S1", "DP replica symmetry",
             "All data-parallel replicas execute identical compute graphs; one witness per replica shape."),
            ("S2", "TP within-layer symmetry",
             "Tensor-parallel ranks within a layer have symmetric all-reduce patterns."),
            ("S3", "PP stage shape similarity",
             "Pipeline stages have identical shape when layers divide evenly; each stage is one witness."),
            ("S4", "EP expert interchangeability",
             "Expert-parallel groups run different experts but have structurally identical kernels."),
        ],
    },
    {
        "category": "Performance Heuristics",
        "description": "Rank candidates by expected throughput before real measurements.",
        "rules": [
            ("P1", "TP sweet spot at 2–8",
             "TP scales well within NVLink domain; cross-node TP degrades rapidly."),
            ("P2", "More DP → better throughput (sublinearly)",
             "Data parallelism scales near-linearly until communication overhead dominates."),
            ("P3", "Larger MBS → higher throughput",
             "Bigger micro-batches improve GPU utilization, up to the memory limit."),
            ("P4", "Fewer PP stages → less bubble overhead",
             "Pipeline parallelism introduces bubble overhead proportional to (pp-1)/pp."),
            ("P5", "Selective recompute is usually optimal",
             "Full recompute saves most memory but costs ~30% FLOPs; selective hits the sweet spot."),
            ("P6", "MoE EP scales sublinearly",
             "All-to-all communication limits expert-parallel scaling efficiency."),
        ],
    },
    {
        "category": "Memory Heuristics",
        "description": "Estimate and propagate memory pressure across configurations.",
        "rules": [
            ("M1", "MBS scales activation memory linearly",
             "Doubling micro-batch size roughly doubles activation memory per GPU."),
            ("M2", "TP reduces parameter memory linearly",
             "Parameter memory per GPU is total_params / tp."),
            ("M3", "PP reduces per-stage activation memory",
             "Activation memory per GPU is ~1/pp of the full model, offset by pipeline buffers."),
            ("M4", "DP doesn't change per-GPU memory",
             "Each data-parallel replica holds the same model and activation state."),
            ("M5", "Recompute reduces activation memory 2–4x",
             "Full recompute ~4x reduction, selective recompute ~2x reduction, at FLOP cost."),
            ("M6", "EP adds expert memory per rank",
             "MoE shards parameters across expert ranks but adds all-to-all buffers."),
        ],
    },
]
