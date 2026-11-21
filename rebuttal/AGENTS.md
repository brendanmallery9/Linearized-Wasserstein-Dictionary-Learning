# Rebuttal working instructions

## Start each session

Read `README.md`, `CONTEXT.md`, `REVIEWS.md`, `EXPERIMENTS.md`, `DECISIONS.md`, and the latest entries in `RESEARCH_LOG.md` in this directory. These files carry context across chats; update them as work progresses. User instructions take precedence.

## Maintain evidence

- Assign reviewer concerns stable IDs (`R1-C01`) after the actual reviews arrive. Keep source wording and our interpretation separate.
- Assign experiments stable IDs (`EXP-001`) before running them. Record the concern, hypothesis, comparison, controls, metrics, seeds, compute budget, and success criterion before inspecting results.
- Each execution gets a unique run directory. Record the exact command, working directory, configuration, code commit and local diff, environment, device, data identity and split, seeds, timestamps with timezone, exit status, and artifact paths. Preserve stdout/stderr and failures as well as successes.
- A Git commit alone does not identify dirty working-tree code. Save a diff and preserve relevant untracked source files with each run. Do not commit unrelated existing changes.
- Label smoke tests as integration checks. Use completed, inspected experiment outputs for scientific claims. Distinguish planned, running, failed, completed, and validated work.
- Keep raw outputs immutable. Put bulky artifacts in existing ignored results directories; keep small manifests, selected evidence, and prose here. Record artifact locations and checksums for evidence used in the rebuttal. Do not assume ignored files are backed up.

## End each work session

Keep `RESULTS_SUMMARY.md` current when experimental findings are added or revised: one or two sentences per result, evidence links, and explicit completed/review-pending/partial labels. Keep corruption types and severity levels distinct; append summary snapshots or corrections to `RESEARCH_LOG.md` without rewriting its history.

Append a dated entry to `RESEARCH_LOG.md` with work performed, commands or run links, findings, limitations, decisions, and next steps. Correct prior entries by appending a correction, preserving history. Never invent results or reconstruct unknown details as fact.

Update `CONTEXT.md` with the current state and next actions; update the experiment and concern registers. Append substantive decisions to `DECISIONS.md`. Tie each substantive draft claim to evidence. Keep review-dependent text explicitly pending until the review is available.

## User shorthand: report to the results summary

When the user says “report to the results summary,” “add this to the results summary,” or similar wording, treat it as an instruction to update `rebuttal/RESULTS_SUMMARY.md` with the experimental results from the current task or the experiments they identify. Do the update directly; do not merely summarize in chat or ask for confirmation.

Use one or two sentences per result, with an evidence link and the essential comparison, settings/sample size, and completion status. Add new results or update existing stable result IDs without duplicating entries; preserve unfavorable results and limitations, distinguish plans/integration checks from scientific results, and keep corruption types/severities separate. Append a dated record of the additions or corrections to `rebuttal/RESEARCH_LOG.md`, preserving prior log history, and return a link to the updated summary. This shorthand does not authorize launching experiments or archiving the user's prompt.

## GPU setup

This is the workflow-only GPU checkout on Vast instance 61714022, at `/workspace/WDL_repo`, based on GitHub `main`. The workflow branch is `codex/rebuttal-gpu`.

Use `/venv/main/bin/python` or `source /venv/main/bin/activate`; noninteractive SSH shells do not activate the venv automatically. Initial integration check: RTX 3060, 12288 MiB VRAM, PyTorch 2.14.1+cu132, CUDA runtime 13.2; CUDA matrix multiplication passed. WDL dependencies have not been installed or validated. Inspect requirements before modifying the environment; the Mac environment used a different PyTorch version.

Read `/etc/vast-agents-guide.md` for instance operation. Recheck device and storage before substantial runs. There is no attached volume: copy irreplaceable results off-instance before recycle/destroy. Keep credentials out of the checkout. No local experiment data, results, dirty code changes, reviews, or historical reports were transferred.

Git sparse checkout omits tracked dataset and result directories. Inspect `.git/info/sparse-checkout` before changing it; do not disable sparse checkout casually. Fetch or generate only data required by an authorized experiment.

The empty registers here do not supersede the primary research record on the Mac. EXP-001 through EXP-006 are already allocated there; reserve new IDs with the primary checkout before creating experiments. No experiments are authorized solely by this setup.

## Independently launched agents

The user launches agents, points them at `rebuttal/`, and assigns a task. Do not spawn additional agents unless explicitly requested. This is a task-and-report workflow; autonomous research loops are a possible future extension, not currently configured.

Read relevant existing reports before working. Use a unique task slug and save a final report to `reports/YYYY-MM-DD_task-slug/REPORT.md` using `templates/report.md`. Save a report even for partial, failed, or blocked work, and return its path in the final handoff.

Keep edits within the assigned scope. Where agents share a checkout, avoid overlapping shared-file edits; append log entries carefully without rewriting other agents' entries. If shared registers are owned by another active task, put proposed updates in the report for integration. Do not overwrite another agent's work or commit unrelated changes. Report any processes left running.

The primary agent reviews reports and incorporates supported conclusions into context and registers. Reports are evidence inputs, not automatic scientific validation or concern resolution.

## Opt-in saved prompts

Save specific prompts only when the user directs you to do so. Do not automatically archive every task or message. Use `prompts/PROMPT-NNN_slug.md`, record the date/source and transcription status, and preserve the requested text. Index each saved prompt in its owning report under **Prompt index**, with its ID and relative path; update `prompts/README.md` with the report link. Later reports may reference an existing prompt. If none were requested, say “None requested” in the report. See the prompt directory for the initial point-cloud brief and report.
