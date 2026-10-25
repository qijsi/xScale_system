# xScale System

Self-evolving parallel strategy search for large-scale LLM training on limited GPUs.

## Overview

xScale is a research prototype for finding efficient parallel execution strategies for large-scale LLM
training using a trust-calibrated self-evolving loop. Its current orchestrator uses domain heuristics, optional BO, and heuristic trust tracking. The default evaluator is synthetic; S2/S5 and CrystalLLM integration remain incomplete.

## Architecture

Five skill modules:

| Module | Name | Version | Status |
|---|---|---|---|
| **S1** | Strategy Space | v0 | ✅ Implemented |
| **S2** | Cost Model | v0 | ⚠️ Stub only |
| **S3** | Structure (RSS/FIS/CEC) | v0 | ⚠️ Known defects |
| **S4** | Feedback + Trust Calibration | v0 | ⚠️ Adaptive controller + heuristic TrustTracker; integration pending |
| **S5** | Evaluator | v0 | ⚠️ Stub only |

Three compression metrics:
- **RSS** (Rank Substitutability Score) — proposed metric; explicit scoring is not implemented
- **FIS** (Fidelity Impact Score) — how sensitive ranking is to compressing each op
- **CEC** (Compressed Execution Confidence) — heuristic REAL fraction minus boundary penalty; no probability bound

Four trust levels (L0 → L3): Exploration → Calibrating → Confident → Verifying.

## Quick Start

```bash
PYTHONPATH=src python3 -m unittest discover tests/skills -v
PYTHONPATH=src python3 -m unittest discover tests/witness -v
```

## Project Structure

```
src/xscale/
  skills/          # Five skill modules (S1-S5)
  witness/         # Low-level witness planning (used by S3)
tests/
  skills/          # Skill-layer tests
  witness/         # Witness-module tests
docs/
  xscale-skill-layer-design.md    # Full design doc
  xscale-skill-layer-progress.md  # Progress tracker
```

## Related Paper

Companion paper repository: [github.com/qijsi/xScale](https://github.com/qijsi/xScale)

## Current evidence

See [implementation status](docs/project-alignment-status.md). The 85 focused tests pass in the CrystalLLM integration; no target-scale GPU evaluation or measured search speedup is established.
