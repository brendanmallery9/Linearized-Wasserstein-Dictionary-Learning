# GPU checkout context

Updated: 2026-11-21 (America/Los_Angeles).

## Scope

Prepare WDL code from GitHub main and the rebuttal workflow structure on Vast instance 61714022. No local experiment artifacts or dirty working-tree code transferred. No training launched.

## Environment

- Checkout: `/workspace/WDL_repo`; branch `codex/rebuttal-gpu`.
- Python: `/venv/main/bin/python`; PyTorch 2.14.1+cu132, CUDA 13.2.
- RTX 3060, 12288 MiB VRAM; CUDA matrix multiplication passed.
- WDL dependency installation and pipeline checks remain pending.
- No persistent volume; results require off-instance backup.

## Research continuity

The primary Mac checkout retains the existing rebuttal context, reviews, experiments and evidence. EXP-001 through EXP-006 are already allocated. Coordinate IDs and obtain the relevant context before planning new research. Existing completed findings are not reproduced by this scaffold.

## Next actions

1. Select an experiment and obtain its scientific context and authorized budget.
2. Install compatible WDL dependencies and run a bounded integration check.
3. Specify data, splits, seeds, baselines and provenance before scientific execution.
