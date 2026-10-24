# Implementation Gaps

This document tracks gaps between the paper's claims and the current
implementation, ordered by priority.

## Critical Gaps (block end-to-end system)

### G1: No orchestrator / search loop ✅ DONE
- **Status:** Missing entirely
- **Paper claim:** "Trust-calibrated orchestrator coordinates five skill modules"
- **What's needed:** An `Orchestrator` class with the main search loop
  (Algorithm 1 from the paper), managing trust level, acquisition,
  escalation, and stopping conditions.
- **Effort:** Medium — 2-3 days

### G2: Cost model v0 missing (only stub)
- **Status:** Empty class, no implementation
- **Paper claim:** "S2 provides analytical estimates"
- **What's needed:** Roofline + communication-volume + analytical memory
  model. Even a simple one is needed for the search loop to work.
- **Effort:** Small — 1 day

### G3: Evaluator v0 missing (only stub)
- **Status:** Empty class, no implementation
- **Paper claim:** "S5 executes candidates at chosen fidelity"
- **What's needed:** v0 analytical evaluator that calls S2 and returns
  standardized ExecutionResult. Required before the loop can run.
- **Effort:** Small — 0.5 days

### G4: Trust level / calibration mechanism missing
- **Status:** Adaptive controller exists but no trust levels (L0-L3)
- **Paper claim:** "Four trust levels governing orchestration"
- **What's needed:** A TrustTracker that computes trust from prediction
  accuracy, evaluation count, and diversity. Integrates with fidelity
  routing.
- **Effort:** Medium — 1-2 days

## High-Priority Defects (existing code that's wrong)

### D1: S3 boundary class adds all members as witnesses
- **Status:** Known defect in `witness/partitioner.py`
- **Impact:** Kills compression ratio — DP-replica symmetry doesn't help if
  every boundary rank is kept.
- **Fix:** Pick one boundary representative per class, not all members.
- **Effort:** Small — 0.5 days

### D2: CEC requires ~90% REAL ops
- **Status:** Known defect in `witness/confidence.py`
- **Impact:** Defeats the purpose of compression — threshold of 0.9 means
  almost everything must be REAL.
- **Fix:** Replace uniform count with FIS-weighted CEC.
- **Effort:** Small — 0.5 days

### D3: Tier assignment ignores critical path
- **Status:** Known defect in `witness/planner.py`
- **Impact:** All comm ops go SYMBOLIC regardless of criticality.
- **Fix:** Use causal DAG criticality to assign tiers.
- **Effort:** Medium — 1 day

## Medium-Priority Gaps (improve quality but not blocking)

### G8: Heuristic bootstrap rules ✅ DONE
- **Status:** Implemented in `skills/heuristic_rules.py`, integrated into S1
- **Paper claim:** "System starts with domain-knowledge rules before any data"
- **What was done:** 23 rules across 4 categories (hard constraints,
  structural symmetries, performance heuristics, memory heuristics).
  Integrated into S1 seeding, neighbor proposal, and witness estimation.
- **Effort:** Small — done

### G5: RSS not explicitly named / scored
- **Status:** Partitioning exists but not called "RSS score"
- **Impact:** Naming inconsistency with paper
- **Fix:** Add explicit RSS computation function to S3
- **Effort:** Small — 0.5 days

### G6: No strategy-delta incremental evaluation
- **Status:** Delta computation exists but evaluator doesn't use it
- **Impact:** Every evaluation is from scratch, even for small deltas
- **Fix:** S5 v2: use delta to reuse captured traces
- **Effort:** Large — 1-2 weeks

### G7: No CrystalLLM integration
- **Status:** Evaluator only has analytical stub
- **Impact:** No real evaluation, all experiments are simulated
- **Fix:** Integrate with CrystalLLM capture-plan-emulate pipeline
- **Effort:** Large — 1-2 weeks

## Low-Priority / Future Work

- Cross-workload transfer (S2 v2 / S4 v2)
- Learned FIS from execution feedback
- Learned symmetry class stability
- LLM-augmented candidate proposal
- Multi-objective optimization (throughput + memory)

## Implementation Roadmap (by order)

- [ ] **Phase 1: End-to-end loop works**
  - [ ] G2: S2 cost model v0
  - [ ] G3: S5 evaluator v0 (analytical)
  - [ ] G4: Trust tracker v0
  - [x] G1: Orchestrator + search loop (trust-calibrated, BO surrogate, four trust levels)
  - [x] G8: Heuristic bootstrap rules (23 rules in 4 categories, integrated into S1)
  - [ ] Demo with simulated data

- [ ] **Phase 2: Compression works correctly**
  - [ ] D1: Fix boundary witness selection
  - [ ] D2: FIS-weighted CEC
  - [ ] D3: Critical-path tier assignment

- [ ] **Phase 3: Real evaluation**
  - [ ] G7: CrystalLLM integration
  - [ ] G6: Delta-based incremental evaluation
  - [ ] Real-workload experiments

- [ ] **Phase 4: Polish**
  - [ ] G5: Explicit RSS scoring
  - [ ] Cross-workload transfer
  - [ ] Full ablation suite
