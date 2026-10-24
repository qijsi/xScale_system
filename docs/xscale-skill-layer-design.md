# xScale Skill Layer Design

## Purpose

The skill layer encodes domain knowledge that the search engine consults
before spending evaluation budget. It replaces generic black-box search with
task-aware decisions. Each skill is independently testable, has a clear
input/output contract, and evolves from analytical v0 to feedback-corrected
v2 without changing its interface.

## Architecture

    Search Engine (BO / Jev / heuristic — pluggable)
         |
         v
    ┌────────────────────────────────────────────┐
    │ Skill Layer (this document)                 │
    │                                             │
    │  S1. StrategySpace    → what to search      │
    │  S2. CostModel        → how good is it      │
    │  S3. Structure        → how to compress     │
    │  S4. Feedback         → how to learn        │
    │  S5. Evaluation       → how to measure      │
    └────────────────────────────────────────────┘
         |
         v
    Evaluation Substrate (CrystalLLM / analytical)

## Module Definitions

### S1. Evolving Strategy Space Skill

**Responsibility.** Maintain a partial, evolving model of the strategy space.
Start from a provably safe seed set; discover new legal strategies through
execution feedback and neighborhood exploration; prune infeasible regions
from learned evidence.

**Design rationale.** A complete static enumeration is impractical for real
workloads: legality depends on framework internals, allocator behavior, MoE
routing structure, and hardware topology interactions that resist analytical
capture. Treating the space itself as evolvable — rather than fixed — aligns
with xScale's self-evolving search narrative and produces a more honest
system: the space model improves with every execution result.

**State model.**

```
StrategySpaceState:
    known_feasible:    set of configs that have executed successfully
    known_infeasible:  set of configs that failed (OOM, timeout, error)
    unexplored:        set of configs not yet evaluated
    estimated:         set of configs with analytical estimates only
```

**Key invariant.** Every observed result updates the space state; the search
engine never queries a static list but always the current state.

**Interface.**

```python
class EvolvingStrategySpace:
    def __init__(self, model: ModelDescriptor, hardware: HardwareDescriptor): ...

    def seed(self) -> list[ParallelConfig]:
        """Return a minimal, provably safe starting set."""

    def propose_neighbors(
        self, config: ParallelConfig, radius: int = 1
    ) -> list[ParallelConfig]:
        """Generate nearby unexplored candidates."""

    def mark_result(
        self, config: ParallelConfig, result: ExecutionResult
    ) -> SpaceUpdate:
        """Record execution outcome and update feasible/infeasible regions."""

    def feasible_region(self) -> list[ParallelConfig]:
        """Current best-effort estimate of feasible strategies."""

    def infeasible_reason(self, config: ParallelConfig) -> str | None:
        """Return learned reason if config is known infeasible."""

    def coverage(self) -> float:
        """Fraction of estimated total space that has been evaluated."""
```

**Neighborhood definition.**

```
neighbors(config) =
    {config with tp±1} ∪ {config with pp±1} ∪ {config with mbs×2 or ÷2} ∪
    {config with recompute toggled} ∪ {config with ep changed}
filtered by:
    - world_size divisibility (hard constraint)
    - not already known_feasible or known_infeasible
```

**Infeasibility propagation.**

When a config fails (e.g., OOM at `tp=2, pp=4, mbs=8`):

1. Mark that exact config as `known_infeasible`.
2. Propagate to neighbors with higher memory pressure: same topology with
   `mbs ≥ 8` → mark as `likely_infeasible` (soft, still queryable).
3. If multiple OOMs in same topology region, mark entire region.

**Evolution path.**

| Version | Seed generation | Neighbor proposal | Infeasibility learning |
|---|---|---|---|
| v0 | Divisibility + basic memory check | ±1 on TP/PP/EP, ×2/÷2 on MBS | Exact config only |
| v1 | + communication feasibility (NVLink domain, cross-node) | + structural similarity (nearby PP stages) | + propagation to higher-MBS neighbors |
| v2 | + learned from cross-workload transfer | + learned proposal policy (which neighbors tend to help) | + region-level: "TP>4 on this interconnect is slow" |

**Maps to existing code.** `witness/types.py:StrategySpec` is the input
representation. `witness/delta.py:compute_delta` can inform neighbor proposal
(which dimensions to perturb). New module needed.

**Key design decision.** The space never claims to be complete. It reports
`coverage()` — what fraction of the estimated total space has been evaluated —
and the search engine decides whether coverage is sufficient to stop or
whether to explore further. This avoids the false precision of a static
enumeration while still providing a bounded search contract.

**Roadmap milestone.** M1 (Optimization Contract), M2 (Baselines).

---

### S2. Cost Model Skill

**Responsibility.** Estimate latency, communication volume, and peak memory
for any candidate strategy without running it.

**Interface.**

```python
class CostModel:
    def __init__(self, model: ModelDescriptor, hardware: HardwareDescriptor): ...
    def estimate(self, config: ParallelConfig) -> CostEstimate: ...
    def update(self, config: ParallelConfig, result: ExecutionResult) -> None: ...
    def confidence(self, config: ParallelConfig) -> float: ...
```

**Output type.**

```python
@dataclass
class CostEstimate:
    latency_ms: float
    latency_std: float          # uncertainty
    comm_volume_mb: float
    peak_memory_gb: float
    memory_headroom_gb: float   # negative = infeasible
```

**Evolution path.**

| Version | Latency model | Memory model | Calibration |
|---|---|---|---|
| v0 | Roofline (FLOPs / peak TFLOPS) + communication volume / bandwidth | Analytical (params + activations + optimizer states) | None |
| v1 | Per-collective-type cost table; kernel efficiency from MBS anchors | Analytical + measured peak from one run | Global multiplicative correction (current `adaptive.py`) |
| v2 | Per-op corrections from execution graph; workload-family-specific | Learned boundary refinement | Feature-aware: different correction per TP/PP/EP region |

**Maps to existing code.** `witness/adaptive.py` provides v1 latency
calibration (log-ratio correction per workload). `witness/confidence.py`
provides FIS scoring. Needs extension for communication and memory models.

**Roadmap milestone.** M2 (Baselines), M3 (Task-Aware Search).

---

### S3. Structural Compression Skill

**Responsibility.** Given an execution graph, determine which ranks and
operations need full fidelity execution, and which can be compressed.

**Interface.**

```python
class StructuralCompressor:
    def analyze(self, graph: ExecutionGraph) -> CompressionPlan: ...
    def plan_for_strategy(self, config: ParallelConfig) -> WitnessPlan: ...
    def delta(self, old: ParallelConfig, new: ParallelConfig) -> StrategyDelta: ...
```

**Evolution path.**

| Version | Symmetry detection | Criticality | Delta reuse |
|---|---|---|---|
| v0 | Topology signature (current `partitioner.py`) | Op-type heuristic (current `planner.py`) | `compute_delta` (current `delta.py`) |
| v1 | + causal DAG criticality from real graph | Critical-path ops forced to REAL; others compressed | Per-stage local re-capture for recompute changes |
| v2 | + learned class stability (which classes survive strategy transitions) | Learned FIS from execution feedback | Cross-workload class transfer |

**Maps to existing code.** Fully mapped to `witness/partitioner.py`,
`witness/planner.py`, `witness/delta.py`, `witness/confidence.py`. This is the
most complete module.

**Known v0 defects to fix.**

1. Boundary class adds all members as witnesses (destroys compression).
2. CEC formula requires 90% REAL (defeats compression purpose).
3. Tier assignment ignores critical path.

**Roadmap milestone.** M3 (Task-Aware Search).

---

### S4. Feedback Learning Skill

**Responsibility.** Interpret execution results and update cost model /
structural assumptions over time.

**Interface.**

```python
class FeedbackLearner:
    def observe(self, config: ParallelConfig, result: ExecutionResult) -> None: ...
    def correction(self, config: ParallelConfig) -> CostCorrection: ...
    def recommend_fidelity(self, config: ParallelConfig, incumbent: float | None) -> EvaluationMode: ...
    def export_knowledge(self) -> SkillKnowledge: ...
```

**Evolution path.**

| Version | What it learns | Transfer mechanism |
|---|---|---|
| v0 | Global log-ratio correction per workload family | None (cold start each family) |
| v1 | Per-dimension corrections (TP region vs. PP region vs. EP region) | Structural similarity: nearby configs share correction |
| v2 | Cross-workload transfer via model features (layer count, hidden size, MoE ratio) | Embedding-based warm start |

**Maps to existing code.** `witness/adaptive.py` implements v0 (log-ratio
calibration, analytical/replay/real fidelity routing).

**Roadmap milestone.** M3 (Task-Aware Search), M4 (Evaluation Integration).

---

### S5. Evaluation Orchestration Skill

**Responsibility.** Execute a candidate strategy at the chosen fidelity level
and return a standardized result.

**Interface.**

```python
class Evaluator:
    def evaluate(
        self, config: ParallelConfig, mode: EvaluationMode
    ) -> ExecutionResult: ...
    def evaluate_batch(
        self, configs: list[ParallelConfig], mode: EvaluationMode
    ) -> list[ExecutionResult]: ...
```

**Evolution path.**

| Version | Analytical | CrystalLLM replay | Real execution |
|---|---|---|---|
| v0 | Roofline estimate | Not connected | Not connected |
| v1 | Calibrated estimate | Single-config full pipeline (capture → plan → emulate) | Sandbox ranks only |
| v2 | Per-op corrected | Incremental: strategy delta determines which stages re-run | Witness ranks only, with MBS anchor scheduling |

**Maps to existing code.** Not started. This is the integration layer that
connects the optimizer to CrystalLLM's `run_batch_step.sh` pipeline.

**Roadmap milestone.** M4 (Evaluation Substrate Integration).

---

## Shared Descriptors

These data classes are shared across all skill modules.

```python
@dataclass(frozen=True)
class ModelDescriptor:
    """Model architecture parameters relevant to strategy evaluation."""
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
    """Target hardware topology."""
    gpu_model: str          # e.g. "H100"
    num_gpus: int           # total target GPUs
    gpus_per_node: int     # e.g. 8
    nvlink_bandwidth_gbps: float     # intra-node
    internode_bandwidth_gbps: float  # inter-node (IB/RoCE)
    vram_per_gpu_gb: float
    peak_tflops: float      # BF16 peak

@dataclass(frozen=True)
class ParallelConfig:
    """A complete parallel execution strategy."""
    tp: int
    pp: int
    dp: int
    ep: int
    mbs: int
    recompute: str          # "none" | "selective" | "full"
    world_size: int          # tp * pp * dp * ep (with overlap accounted)

@dataclass
class ExecutionResult:
    """Result from evaluating one strategy."""
    config: ParallelConfig
    latency_ms: float
    peak_memory_gb: float
    comm_volume_mb: float
    oom: bool
    success: bool
    mode: EvaluationMode
    notes: str
```

## Implementation Order

Priority is determined by roadmap blocking and research value.

| Phase | Module | Why first | Est. effort |
|---|---|---|---|
| 1 | S1 StrategySpace | Defines what we search; blocks all experiments | 1-2 days |
| 2 | S2 CostModel v0 | Provides analytical baseline for M2 | 2-3 days |
| 3 | S3 fix defects | Compression must work before claiming results | 1 day |
| 4 | S4 FeedbackLearner v1 | Connects evaluation to improved decisions | 2-3 days |
| 5 | S5 Evaluator v0 | Unblocks end-to-end search loop (without CrystalLLM) | 1-2 days |
| 6 | S2→v1 calibration | Uses real CrystalLLM data to improve cost model | 1 week |
| 7 | S5→v1 CrystalLLM | Full integration for replay evaluation | 1-2 weeks |

## Relationship to Existing Code

```
crystalllm/src/crystalllm/witness/
  types.py         → S3 (StrategySpec → becomes ParallelConfig)
  partitioner.py   → S3 (rank symmetry)
  planner.py       → S3 (witness planning, tier assignment)
  delta.py         → S3 (strategy delta)
  confidence.py    → S3/S4 (FIS/CEC → feeds FeedbackLearner)
  adaptive.py      → S4 (fidelity routing, calibration)

New modules needed:
  strategy_space.py → S1
  cost_model.py     → S2
  feedback.py      → S4 (extends adaptive.py)
  evaluator.py      → S5
  descriptors.py    → shared ModelDescriptor, HardwareDescriptor
```

## Next Step

Implement S1 (StrategySpace) and shared descriptors first, because:

1. It unblocks M1 (Optimization Contract) and M2 (Baselines).
2. It defines the exact search space for all subsequent experiments.
3. It produces a testable module in 1-2 days with clear pass/fail criteria.
