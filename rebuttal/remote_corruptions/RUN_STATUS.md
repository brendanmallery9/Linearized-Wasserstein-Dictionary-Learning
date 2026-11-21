# Active execution — check before launching a duplicate

Launched 2026-11-21 on the existing Vast remote machine. These are execution/ownership notes, not scientific results.

- Supervisor: `wdl-noise-20261121`, started PID3328.
- Worktree: `/workspace/wdl-noise-handoff`, pinned commit `49121614b9007a216ff8fa977a0de657c51279bb`.
- Python: `/workspace/wdl-venv/bin/python`.
- Root: `/workspace/wdl-results/remote-noise-20261121`.
- Sequence: EXP-007 iid (540 shapes, clean+12 conditions), then EXP-008 depth (clean+36), then numerical reports. CUDA encoder,8 CPU exact-OT workers; no training.
- Status: inspect `supervisorctl status wdl-noise-20261121`, root `stages.json`, each stage `status.json`, completion/failure/budget markers and stage logs. Do not infer completion from Supervisor alone. Do not restart an existing output root without explicit validated `--resume`.
- CPU/CUDA smoke and six-shape all-family pilot passed; pilot numerical report audit passed. Validation files are integration only.
- Main research report on the Mac: `rebuttal/reports/2026-11-21_remote_noise_handoff/REPORT.md`.

This file is a launch snapshot, not a live promise that the process is still running. Read current state first; do not launch a second copy just because this README offers launch commands. This status-only commit follows the pinned experiment commit; do not update source mid-run.
