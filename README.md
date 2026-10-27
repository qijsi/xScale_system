# xScale System

Research prototype for **decision-directed emulation of LLM parallelism search**.

The current research design separates the complete logical training plan from physical evaluation work on a small GPU platform. Its three proposed contributions are context-aware execution folding, timing/memory-consistent summary execution, and decision-directed refinement. Training still maximizes target throughput subject to per-rank memory capacity; xScale seeks to reduce the measured cost of selecting a high-quality strategy.

Read the [2026-10-27 design and paper reconciliation](docs/refinable-training-execution-design.md) first. The [paper repository](https://github.com/qijsi/xScale) uses `xScale.tex` as its active entry point.

## Implementation boundary

The new graph abstraction and runtime are not implemented yet. Existing code provides reusable scaffolding:

| Component | Current state | Role in the new design |
| --- | --- | --- |
| S1 | Candidate seeds, neighborhoods and heuristic checks | Candidate generator and baseline |
| S2 | Cost model stub and result type | Lightweight summary estimates, pending |
| S3 / witness | Signature partitioning, planning and strategy deltas; known defects | Reuse candidates and invalidation metadata, pending validation |
| S4 | Separate calibration component; heuristic TrustTracker | Historical baseline; local evidence integration pending |
| S5 | Evaluator stub and result type | Real-window backend integration, pending |
| Orchestrator | Synthetic default evaluator, optional policies | Search scaffolding; new refinement control pending |

RSS/FIS/CEC and global trust are historical heuristics, not probability guarantees or the new paper's main contributions. REAL/REPLAY labels from the default evaluator are synthetic mode labels. See [evidence audit](docs/project-alignment-status.md) and [implementation gaps](GAPS.md).

## Development checks

```bash
PYTHONPATH=src python3 -m unittest discover tests/skills -v
PYTHONPATH=src python3 -m unittest discover tests/witness -v
```

The CrystalLLM integration recorded 85 passing focused tests on 2026-10-25. That historical result does not validate the proposed mechanisms. There is no established target-scale GPU accuracy or measured tuning speedup.

## Next milestones

1. Establish target/full-emulation reference executions and measure candidate-evaluation cost.
2. Implement contextual segment boundaries and a full-model lifetime ledger for static dense models.
3. Connect representative windows to CrystalLLM and validate timing, memory, and invalidation.
4. Add budgeted decision refinement and compare quality versus total measured tuning cost.
