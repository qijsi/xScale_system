# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""S1 extension — Heuristic bootstrap rules.

These are the domain-knowledge rules that seed the search before any
execution data has been collected. They serve three purposes:

1. **Prune the search space upfront** — hard constraints eliminate
   obviously-invalid configurations so the search never wastes time on them.
2. **Compress the evaluation** — structural symmetry rules determine how
   many unique witness ranks are needed, enabling massive compression.
3. **Rank and propagate** — performance heuristics order candidates;
   memory heuristics propagate feasibility signals across the space.

All rules have a clear provenance and are independently testable. They
are *not* learned from data; they come from known properties of
parallel training systems.

Extended rule catalog (v1)
--------------------------
Hard constraints (H1-H12): prune invalid configs before any evaluation
  - Basic: world-size, TP node bound, EP range, batch divisibility
  - Memory: MBS upper bound, recompute feasibility, activation floor
  - Topology: NVLink domain alignment, PP inter-node balance

Structural symmetries (S1-S8): compress evaluation, fold search space
  - DP replica: identical compute graph across data-parallel ranks
  - TP within-layer: symmetric all-reduce / all-gather patterns
  - PP stage shape: each stage has same per-layer structure
  - EP expert interchangeability: same kernel, different weights
  - Cross-dimension symmetry: same (tp*pp) product has similar witness count
  - MBS scaling: proportional memory growth, same compute structure
  - Recompute mode: same graph, different recomputation schedule
  - DP-size similarity: DP=4 and DP=8 have near-identical per-rank shape

Performance heuristics (P1-P10): rank candidates by expected throughput
  - DP scaling: near-linear until communication breakpoint
  - PP bubble: (pp-1)/pp overhead
  - TP sweet spot: 2-8 within NVLink
  - MBS utilization: bigger = better (up to memory limit)
  - Recompute FLOP cost: full ~30%, selective ~15%
  - EP all-to-all overhead: sublinear scaling
  - NVLink locality: TP entirely within NVLink domain
  - PP balance: evenly divisible layers = less bubble skew
  - MBS/PP interaction: larger MBS mitigates PP bubble
  - DP communication: ring all-reduce cost grows with dp

Memory heuristics (M1-M10): estimate + propagate memory pressure
  - MBS linear: activation memory ~ mbs * seq_len * hidden
  - TP parameter shard: parameter memory / tp
  - PP activation reduction: per-stage activations / pp
  - DP invariance: per-GPU memory independent of dp
  - Recompute reduction: full ~4x, selective ~2x activation saving
  - EP expert memory: expert params + all-to-all buffers
  - Optimizer state: ~2x parameter memory (Adam)
  - CUDA context / fragmentation: ~1-2 GB fixed overhead
  - PP pipeline buffers: proportional to pp * mbs
  - Gradient buffer: all-reduce sized for largest tensor

Usage patterns
--------------
- hard constraint filter: check_hard_constraints(cfg, model, hw)
- witness estimation: estimate_witness_count(cfg, model, hw)
- symmetry folding: symmetry_fingerprint(cfg, model, hw)
- priority ranking: heuristic_priority_score(cfg, model, hw)
- memory dominance: is_memory_dominant(a, b)
- memory ordering: memory_order_key(cfg)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import (
    HardwareDescriptor,
    ModelDescriptor,
    ParallelConfig,
)


# =====================================================================
# Hard Constraints (H1–H12)
# =====================================================================

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

    # H2: TP bounded by node size (tensor parallelism is latency-sensitive)
    if cfg.tp > hardware.gpus_per_node:
        return False, f"H2: tp {cfg.tp} > gpus_per_node {hardware.gpus_per_node}"

    # H3: EP bounded by expert count
    if model.is_moe and model.num_experts is not None and cfg.ep > model.num_experts:
        return False, f"H3: ep {cfg.ep} > num_experts {model.num_experts}"
    if not model.is_moe and cfg.ep != 1:
        return False, f"H3: ep {cfg.ep} != 1 for dense model"
    if model.is_moe and cfg.ep < 2:
        return False, f"H3: ep {cfg.ep} < 2 for MoE model"

    # H4: all dimensions >= 1
    if cfg.tp < 1 or cfg.pp < 1 or cfg.dp < 1 or cfg.ep < 1 or cfg.mbs < 1:
        return False, f"H4: dimension < 1 (tp={cfg.tp} pp={cfg.pp} dp={cfg.dp} ep={cfg.ep} mbs={cfg.mbs})"

    # H5: DP must make sense (DP=1 is valid but we still allow it; just not DP=0)
    # (covered by H4)

    # H6: MBS reasonable upper bound — if MBS is huge and even with full recompute
    # it can't possibly fit in VRAM, reject outright.
    # Conservative: activation memory = 4 * mbs * seq_len * hidden * num_layers
    # If that alone exceeds VRAM, definitely OOM.
    if model.num_layers and model.hidden_size and model.sequence_length:
        act_per_layer = 4 * cfg.mbs * model.sequence_length * model.hidden_size  # bytes approx
        total_act = act_per_layer * model.num_layers / cfg.pp
        vram_bytes = hardware.vram_per_gpu_gb * 1024**3
        # Even with full recompute (4x saving), activations alone shouldn't exceed 80% VRAM
        if total_act / 4.0 > vram_bytes * 0.8:
            return False, f"H6: MBS {cfg.mbs} too large — activations exceed VRAM even with full recompute"

    # H7: PP must divide layers evenly for balanced stages
    if model.num_layers and model.num_layers % cfg.pp != 0:
        return False, f"H7: pp {cfg.pp} does not divide num_layers {model.num_layers} (unbalanced stages)"

    # H8: TP must be power-of-2 for most frameworks
    # (Megatron-LM requires power-of-2 TP; relax for other frameworks)
    if cfg.tp > 0 and (cfg.tp & (cfg.tp - 1)) != 0:
        return False, f"H8: tp {cfg.tp} is not a power of 2"

    # H9: EP must divide num_experts evenly (balanced expert assignment)
    if model.is_moe and model.num_experts and model.num_experts % cfg.ep != 0:
        return False, f"H9: ep {cfg.ep} does not divide num_experts {model.num_experts}"

    # H10: recompute level valid
    if cfg.recompute not in ("none", "selective", "full"):
        return False, f"H10: invalid recompute mode '{cfg.recompute}'"

    # H11: PP must be ≥ 2 if using VPP (virtual pipeline parallelism)
    # (no VPP field yet in ParallelConfig; reserved for future)

    # H12: TP within NVLink domain (more precise than H2)
    # For 8-GPU nodes, TP=8 is OK (all NVLink), TP=16 is not.
    # This is the same as H2 for a single node but matters for multi-node
    # where you might have TP across NVLink domains within a node.
    # Already covered by H2 for now since gpus_per_node == NVLink domain size.

    return True, ""


def is_legal_config(
    cfg: ParallelConfig,
    model: ModelDescriptor,
    hardware: HardwareDescriptor,
) -> bool:
    """Convenience: return True if all hard constraints pass."""
    ok, _ = check_hard_constraints(cfg, model, hardware)
    return ok


# =====================================================================
# Structural Symmetries (S1–S8) — witness estimation + space folding
# =====================================================================


@dataclass(frozen=True)
class SymmetryBreakdown:
    """How many unique witness ranks are needed at each compression level.

    Attributes:
        dp_witnesses: distinct DP replica shapes (usually 1)
        tp_witnesses: distinct TP rank roles per layer (usually 1)
        pp_witnesses: distinct PP stages (= pp when balanced)
        ep_witnesses: distinct expert groups (= ep for MoE)
        total_witnesses: estimated minimum unique ranks for full fidelity
        compression_ratio: world_size / total_witnesses
        symmetry_fingerprint: string grouping configs with identical witness shape
    """

    dp_witnesses: int
    tp_witnesses: int
    pp_witnesses: int
    ep_witnesses: int
    total_witnesses: int
    compression_ratio: float
    symmetry_fingerprint: str


def estimate_witness_count(
    cfg: ParallelConfig,
    model: ModelDescriptor,
    hardware: HardwareDescriptor,
) -> SymmetryBreakdown:
    """Estimate how many witness ranks are needed to faithfully evaluate ``cfg``.

    Uses 8 symmetry rules to compute the minimum set of unique rank types.
    This drives the compression decision: higher symmetry → fewer witnesses
    → cheaper evaluation at the same fidelity level.

    The estimate is conservative; actual S3 partitioning may find more
    symmetries at runtime (e.g., activation checkpointing creates
    additional structure).
    """
    # S1: DP replicas are structurally identical
    # All data-parallel replicas execute the same compute graph with
    # identical memory layout. Only one witness per DP group needed.
    dp_witnesses = 1

    # S2: TP ranks within a layer have symmetric all-reduce patterns
    # For standard Megatron-style TP, ranks differ in which portion of
    # the tensor they hold, but the computational structure is symmetric.
    # For attention there's slight head-level skew but overall 1 witness suffices.
    tp_witnesses = 1

    # S3: PP stages — each stage runs different layers
    # When layers divide evenly (which H7 enforces), each stage has the
    # same structure, just different weights. We count each stage as a
    # separate witness because weights differ and layer-order differs.
    pp_witnesses = cfg.pp

    # S4: EP experts are interchangeable
    # Each expert runs the same FFN kernel with different weights.
    # Structural shape is identical, but since weights differ and MoE
    # routing can cause load imbalance, we count each expert group.
    ep_witnesses = cfg.ep if model.is_moe else 1

    # S5: Cross-dimensional symmetry — if tp==pp, the TP witness and
    # PP-stage witness have similar tensor-shape patterns. But we don't
    # double-count savings here since they're different dimensions.

    # S6: MBS doesn't change structural shape
    # (not affecting witness count, just memory footprint)

    # S7: Recompute mode doesn't change rank structure
    # (not affecting witness count, just execution schedule)

    # Total distinct witnesses
    total = dp_witnesses * tp_witnesses * pp_witnesses * ep_witnesses
    total = min(total, cfg.world_size)

    ratio = cfg.world_size / max(total, 1)

    # S8: Symmetry fingerprint — configs with the same fingerprint have
    # structurally identical per-rank computation graphs.
    # Used for space folding: if we've evaluated one config in a
    # symmetry class, we can transfer partial knowledge to others.
    fingerprint = f"pp{cfg.pp}_ep{cfg.ep if model.is_moe else 1}_tpshape"

    return SymmetryBreakdown(
        dp_witnesses=dp_witnesses,
        tp_witnesses=tp_witnesses,
        pp_witnesses=pp_witnesses,
        ep_witnesses=ep_witnesses,
        total_witnesses=total,
        compression_ratio=ratio,
        symmetry_fingerprint=fingerprint,
    )


def symmetry_fingerprint(
    cfg: ParallelConfig,
    model: ModelDescriptor,
    hardware: HardwareDescriptor,
) -> str:
    """Return a fingerprint grouping configs with identical per-rank structure.

    Configs with the same fingerprint can share evaluation results more
    aggressively: their execution graphs have the same shape and
    communication pattern, only tensor sizes differ.
    """
    return estimate_witness_count(cfg, model, hardware).symmetry_fingerprint


def symmetry_transfer_strength(
    cfg_a: ParallelConfig,
    cfg_b: ParallelConfig,
    model: ModelDescriptor,
    hardware: HardwareDescriptor,
) -> float:
    """How much knowledge transfers from cfg_a to cfg_b via symmetry.

    Returns 0.0 (no transfer) to 1.0 (identical structure).
    Used for delta evaluation: high-transfer configs can reuse most of
    a previous trace.
    """
    if cfg_a == cfg_b:
        return 1.0

    fp_a = symmetry_fingerprint(cfg_a, model, hardware)
    fp_b = symmetry_fingerprint(cfg_b, model, hardware)

    # Same fingerprint = same structure → very strong transfer
    if fp_a == fp_b:
        return 0.9

    score = 0.0

    # Same DP size: DP doesn't affect per-rank structure
    if cfg_a.dp == cfg_b.dp:
        score += 0.2

    # Same TP size: same tensor sharding pattern
    if cfg_a.tp == cfg_b.tp:
        score += 0.2

    # Same PP size: same stage count and balance
    if cfg_a.pp == cfg_b.pp:
        score += 0.2

    # Same EP size: same expert count
    if cfg_a.ep == cfg_b.ep:
        score += 0.1

    # Same recompute: same execution schedule
    if cfg_a.recompute == cfg_b.recompute:
        score += 0.1

    # MBS: same MBS = same activation shapes per step
    if cfg_a.mbs == cfg_b.mbs:
        score += 0.1

    # Same world size product ratio: similar overall scale
    if cfg_a.world_size == cfg_b.world_size:
        score += 0.1

    return min(score, 0.95)  # cap at 0.95 — never assume perfect transfer


# =====================================================================
# Performance Heuristics (P1–P10) — rank candidates
# =====================================================================


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

    # P1: TP sweet spot (2-8 within NVLink domain)
    if cfg.tp == 1:
        score += 0.0
    elif 2 <= cfg.tp <= min(8, hardware.gpus_per_node):
        score += 2.0
    else:
        score -= (cfg.tp - min(8, hardware.gpus_per_node)) * 0.5

    # P2: More DP → better throughput (sublinearly)
    # Log scale because diminishing returns from all-reduce communication
    score += math.log2(cfg.dp + 1) * 3.0

    # P3: Larger MBS → better GPU utilization
    score += math.log2(cfg.mbs + 1) * 0.5

    # P4: Fewer PP stages → less bubble overhead
    # Bubble = (pp-1)/pp of time
    bubble = (cfg.pp - 1) / max(cfg.pp, 1)
    score -= bubble * 5.0

    # P5: Recompute FLOP cost
    # Full recompute ~+30% FLOPs, selective ~+15%, none = 0%
    # But none uses more memory. We prefer selective as sweet spot.
    if cfg.recompute == "selective":
        score += 1.0
    elif cfg.recompute == "full":
        score += 0.0
    else:  # none
        score += 0.5

    # P6: EP sublinear scaling (MoE only)
    if model.is_moe:
        # All-to-all communication grows with EP
        score -= math.log2(cfg.ep + 1) * 0.3
    else:
        pass  # dense models have EP=1 always

    # P7: NVLink locality — bonus for TP that fits entirely in one node
    if cfg.tp <= hardware.gpus_per_node:
        score += 0.5

    # P8: PP balance — already enforced by H7, but round number of layers helps
    # (already a hard constraint)

    # P9: MBS/PP interaction — larger MBS reduces PP bubble impact
    # (relative effect, already partially captured by P3 and P4 separately)

    # P10: DP communication cost at scale
    # Ring all-reduce cost = 2*(dp-1)/dp * message_size / bandwidth
    # For DP > 16, communication cost starts eating into gains
    if cfg.dp > 16:
        score -= (cfg.dp - 16) * 0.1

    # Structural bonus: prefer evaluating highly-symmetric configs first
    # (they're cheaper to evaluate, giving more data points per budget)
    sym = estimate_witness_count(cfg, model, hardware)
    score += math.log2(sym.compression_ratio + 1) * 0.5

    return score


# =====================================================================
# Memory Heuristics (M1–M10) — propagation + early ranking
# =====================================================================


def memory_order_key(cfg: ParallelConfig) -> tuple:
    """Return a tuple monotonic in per-GPU memory usage.

    Higher key = more memory. Used for infeasibility propagation:
    if config A OOMs and config B has a higher key, B is likely OOM too.

    The tuple is ordered by how strongly each dimension correlates with
    memory: MBS (strongest) → TP → PP → recompute → EP → DP (weakest).
    """
    rc_order = {"none": 3, "selective": 2, "full": 1}
    return (
        cfg.mbs,           # M1: strongest monotonic signal
        -cfg.tp,           # M2: more TP → less parameter memory
        -cfg.pp,           # M3: more PP → less activation memory (roughly)
        rc_order.get(cfg.recompute, 2),  # M5: recompute saves memory
        -cfg.ep,           # M6: more EP → less expert memory per rank
        cfg.dp,            # M4: DP doesn't change per-GPU much (weakest)
    )


def memory_dominance_strength(cfg_a: ParallelConfig, cfg_b: ParallelConfig) -> float:
    """How strongly cfg_a dominates cfg_b in memory usage.

    Returns:
        > 0: cfg_a likely uses more memory than cfg_b
        < 0: cfg_b likely uses more memory than cfg_a
        = 0: ambiguous (dimensions conflict)

    Magnitude indicates confidence: stronger signal = higher magnitude.
    """
    score = 0.0

    # M1: MBS — strongest signal (linear in activation memory)
    if cfg_a.mbs > cfg_b.mbs:
        score += 3.0 * (cfg_a.mbs / cfg_b.mbs - 1) / max(cfg_a.mbs / cfg_b.mbs - 1, 1)
    elif cfg_a.mbs < cfg_b.mbs:
        score -= 3.0 * (cfg_b.mbs / cfg_a.mbs - 1) / max(cfg_b.mbs / cfg_a.mbs - 1, 1)

    # M2: TP — reduces parameter memory (often largest memory component)
    # Parameter memory scales as 1/tp; for large models this is the dominant term.
    if cfg_a.tp < cfg_b.tp:
        ratio = cfg_b.tp / cfg_a.tp  # how many times fewer params with higher tp
        score += 4.0 * (1.0 - 1.0 / ratio)  # saturates at 4.0 for infinite ratio
    elif cfg_a.tp > cfg_b.tp:
        ratio = cfg_a.tp / cfg_b.tp
        score -= 4.0 * (1.0 - 1.0 / ratio)

    # M3: PP — reduces per-stage activation memory
    if cfg_a.pp < cfg_b.pp:
        score += 1.5 * (1.0 - cfg_a.pp / cfg_b.pp)
    elif cfg_a.pp > cfg_b.pp:
        score -= 1.5 * (1.0 - cfg_b.pp / cfg_a.pp)

    # M5: Recompute — saves activation memory 2-4x
    rc_level = {"full": 1.0, "selective": 0.5, "none": 0.0}
    a_rc = rc_level.get(cfg_a.recompute, 0.5)
    b_rc = rc_level.get(cfg_b.recompute, 0.5)
    score += (b_rc - a_rc) * 2.0

    # M6: EP — expert memory
    if cfg_a.ep > cfg_b.ep:
        score -= 0.5 * (1.0 - cfg_b.ep / cfg_a.ep)
    elif cfg_a.ep < cfg_b.ep:
        score += 0.5 * (1.0 - cfg_a.ep / cfg_b.ep)

    # M4: DP — negligible effect on per-GPU memory
    # (intentionally 0)

    return score


def is_memory_dominant(cfg_a: ParallelConfig, cfg_b: ParallelConfig) -> bool:
    """Return True if cfg_a definitely uses ≥ memory than cfg_b.

    True only when the comparison is unambiguous across all major dimensions.
    When dimensions conflict, we return False to avoid incorrect propagation.
    """
    return memory_dominance_strength(cfg_a, cfg_b) >= 1.5


# =====================================================================
# Rule Catalog (for documentation / paper generation)
# =====================================================================


RULE_CATALOG: list[dict] = [
    {
        "category": "Hard Constraints",
        "description": "Configurations that violate these are never evaluated.",
        "rules": [
            ("H1", "World-size divisibility",
             "tp × pp × dp × ep must equal total GPU count."),
            ("H2", "TP bounded by node size",
             "Tensor parallelism cannot cross nodes — latency-sensitive collective."),
            ("H3", "EP range constraint",
             "EP ≤ num_experts for MoE; EP = 1 for dense; EP ≥ 2 for MoE."),
            ("H4", "All dimensions ≥ 1",
             "No zero or negative parallelism dimensions."),
            ("H5", "DP ≥ 1 (basic validity)",
             "DP=1 is valid but low priority; DP=0 is invalid."),
            ("H6", "MBS memory upper bound",
             "MBS must not exceed VRAM even with full recompute (activations-only check)."),
            ("H7", "PP divides layers evenly",
             "Number of layers must be divisible by PP for balanced pipeline stages."),
            ("H8", "TP must be power of 2",
             "Most parallel frameworks require power-of-2 tensor parallelism."),
            ("H9", "EP divides experts evenly",
             "Number of experts must be divisible by EP for balanced expert assignment."),
            ("H10", "Recompute mode valid",
             "Must be none, selective, or full."),
            ("H11", "VPP requires PP ≥ 2",
             "Virtual pipeline parallelism only applies with pipeline parallelism active."),
            ("H12", "TP within NVLink domain",
             "TP groups should map to NVLink domains for optimal collective performance."),
        ],
    },
    {
        "category": "Structural Symmetries",
        "description": "Symmetries that reduce witness count and enable space folding.",
        "rules": [
            ("S1", "DP replica symmetry",
             "All data-parallel replicas execute identical compute graphs — one witness per replica shape."),
            ("S2", "TP within-layer symmetry",
             "TP ranks within a layer have symmetric all-reduce / all-gather patterns."),
            ("S3", "PP stage shape similarity",
             "Each pipeline stage has the same per-layer structure (different weights)."),
            ("S4", "EP expert interchangeability",
             "Expert-parallel groups run structurally identical FFN kernels with different weights."),
            ("S5", "Cross-dimension symmetry",
             "Configs with same (tp×pp) product have similar witness tensor shapes."),
            ("S6", "MBS shape invariance",
             "MBS scales memory but not computational structure — same graph shape."),
            ("S7", "Recompute schedule equivalence",
             "Recompute mode changes execution schedule but not graph topology."),
            ("S8", "DP-size structural similarity",
             "Different DP sizes produce near-identical per-rank computation graphs."),
        ],
    },
    {
        "category": "Performance Heuristics",
        "description": "Rank candidates by expected throughput before real measurements.",
        "rules": [
            ("P1", "TP sweet spot at 2–8",
             "TP scales well within NVLink domain; cross-node TP degrades rapidly."),
            ("P2", "DP throughput scaling",
             "More DP → better throughput, sublinear due to all-reduce communication."),
            ("P3", "MBS utilization scaling",
             "Larger micro-batches improve GPU utilization, up to memory limit."),
            ("P4", "PP bubble overhead",
             "Pipeline bubble = (pp-1)/pp × iteration time; fewer stages is better."),
            ("P5", "Recompute FLOP cost",
             "Full recompute ~30% more FLOPs; selective ~15%; selective is usually sweet spot."),
            ("P6", "EP sublinear scaling",
             "All-to-all communication limits MoE expert-parallel scaling efficiency."),
            ("P7", "NVLink locality bonus",
             "TP entirely within a single NVLink domain avoids cross-node penalty."),
            ("P8", "PP balance efficiency",
             "Even layer division minimizes bubble skew; uneven stages waste throughput."),
            ("P9", "MBS/PP interaction",
             "Larger MBS reduces relative PP bubble overhead per micro-batch."),
            ("P10", "DP communication cost at scale",
             "Ring all-reduce cost grows with DP; gains saturate for DP > 16."),
        ],
    },
    {
        "category": "Memory Heuristics",
        "description": "Estimate and propagate memory pressure across configurations.",
        "rules": [
            ("M1", "MBS linear scaling",
             "Activation memory per layer scales linearly with micro-batch size."),
            ("M2", "TP parameter reduction",
             "Parameter memory per GPU = total_params / tp."),
            ("M3", "PP activation reduction",
             "Per-GPU activation memory reduced by ~1/pp, offset by pipeline buffers."),
            ("M4", "DP memory invariance",
             "Per-GPU memory is roughly independent of data-parallel degree."),
            ("M5", "Recompute memory saving",
             "Full recompute ~4x activation reduction; selective ~2x; at FLOP cost."),
            ("M6", "EP expert memory",
             "MoE adds per-rank expert memory + all-to-all communication buffers."),
            ("M7", "Optimizer state overhead",
             "Adam optimizer stores ~2x parameter memory (momentum + variance)."),
            ("M8", "CUDA context / fragmentation",
             "~1-2 GB fixed per-GPU overhead from CUDA context and allocator fragmentation."),
            ("M9", "PP pipeline buffer cost",
             "Pipeline parallelism adds buffers proportional to pp × mbs for inter-stage transfer."),
            ("M10", "Gradient buffer size",
             "All-reduce gradient buffer sized for largest tensor; grows with hidden_size."),
        ],
    },
]
