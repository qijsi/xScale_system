# xScale Skill Layer — Progress Tracker

Five skill modules for xScale's self-improving strategy search.
Each module has a v0 → v1 → v2 evolution path and a stable API.

| Module | Name | Version | Status | Key TODOs |
|---|---|---|---|---|
| **S1** | Strategy Space | v0 | ✅ Implemented | v1: comm feasibility; v2: learned proposal |
| **S2** | Cost Model | v0 | ⬜ Not started | v0: roofline + analytical memory |
| **S3** | Structure | v0 | ⚠️ Has defects | Fix 3 known defects before v1 |
| **S4** | Feedback Learner | v0 | ✅ Exists (adaptive.py) | v1: per-dimension corrections |
| **S5** | Evaluator | v0 | ⬜ Not started | v0: analytical; v1: CrystalLLM |

## File Map

```
crystalllm/src/crystalllm/skills/
  __init__.py          # Package + progress snapshot
  strategy_space.py    # S1 v0 — evolving space (seed + neighborhood + propagation)
  cost_model.py        # S2    — placeholder
  structure.py         # S3 v0 — wraps witness module (partitioner/planner/delta/confidence)
  feedback.py          # S4 v0 — wraps adaptive controller
  evaluator.py         # S5    — placeholder

crystalllm/tests/skills/
  test_strategy_space.py  # S1 unit tests
```

## Known v0 Defects

### S3 (Structure) — must fix before evaluation
1. **CRITICAL** Boundary class adds all members → kills compression
2. **HIGH** CEC requires 90% REAL → defeats compression purpose
3. **MEDIUM** Tier assignment ignores causal critical path

## Evolution Path Per Module

### S1 StrategySpace
- v0: seed (divisibility + memory check), 1-neighborhood, MBS propagation, coverage
- v1: comm feasibility, structural similarity neighbors
- v2: learned proposal policy, cross-workload transfer

### S2 CostModel
- v0: roofline FLOPs + comm volume / bandwidth
- v1: per-collective cost table + MBS efficiency curves
- v2: per-op corrections from execution graph

### S3 Structure
- v0: topology symmetry + heuristic tier + delta classification
- v1: causal DAG criticality + local re-capture + FIS-weighted CEC
- v2: learned class stability + learned FIS

### S4 FeedbackLearner
- v0: global log-ratio correction per workload
- v1: per-dimension corrections + calibrated uncertainty
- v2: cross-workload transfer via embeddings

### S5 Evaluator
- v0: analytical evaluation (calls CostModel)
- v1: CrystalLLM replay full pipeline
- v2: incremental evaluation via strategy delta

## Next Actions (priority order)

1. [ ] Fix S3 v0 defect #1 (boundary representative)
2. [ ] Fix S3 v0 defect #2 (FIS-weighted CEC)
3. [ ] Fix S3 v0 defect #3 (critical-path tiering)
4. [ ] Implement S2 v0 (roofline + analytical memory)
5. [ ] Implement S5 v0 (analytical evaluator)
6. [ ] End-to-end v0 loop: S1 seed → S2 estimate → S3 plan → S4 update
7. [ ] Integrate S5 v1 with CrystalLLM
