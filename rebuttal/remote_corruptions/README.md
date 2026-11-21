# Remote frozen point-cloud noise experiments — agent handoff

**Check [RUN_STATUS.md](RUN_STATUS.md) first: an execution has already been launched. Inspect its current status before starting any duplicate.**

Read this document in full. It is self-contained: prior chat history and Mac files are not needed. The user requested (A) stronger Gaussian measurement noise and then (B) correlated depth noise/scan stripes, comparing our constrained dictionary against **both PCA and positive sparse coding**. This is evaluation, **not training**.

## 1. Scientific question and context

On 540 held-out meshes, at the previous maximum Gaussian sigma=.05, constrained/PCA/positive-SC clean W2² were .004769/.002394/.004594 and noisy W2² .004917/.002715/.004851. Our absolute error was worse, although its increase was smaller. Test a crossover at stronger noise; do not extrapolate one or assume a classification win. Positive SC is a strong required comparator, not an optional baseline.

The holdout is six classes (bed, chair, monitor, sofa, table, toilet), 90 previously unused official-test meshes per class. It excludes the original 1,800 model-development meshes and earlier 60-test-mesh study. These 540 objects have since been inspected in EXP-004/005: this is exploratory follow-up, **not a newly blind confirmation or the full official ModelNet-C benchmark**.

Primary record is on the Mac; the remote scaffold's empty registers do not supersede it. IDs already allocated: EXP-001–007; use **EXP-007 remote continuation for stage A** and **EXP-008 for stage B**, reserved by this handoff. No other models, fine-tuning, extra families, or hyperparameter search are requested.

## 2. Get exactly this branch, preserving other work

Repository: https://github.com/brendanmallery9/Linearized-Wasserstein-Dictionary-Learning . Branch: **codex/rebuttal-gpu**.

Existing Vast checkout may have uncommitted changes. Recommended commands on the remote machine:

```sh
cd /workspace/WDL_repo
git status --short
git fetch origin codex/rebuttal-gpu
git worktree add --detach /workspace/wdl-noise-handoff FETCH_HEAD
cd /workspace/wdl-noise-handoff
test -f rebuttal/remote_corruptions/assets/SHA256SUMS.json
```

Choose a different new worktree path if that path already exists; never delete/replace another agent's directory. For a **clean**, idle checkout already on this branch, `git pull --ff-only origin codex/rebuttal-gpu` is also valid. Sparse checkout must include `rebuttal/remote_corruptions`; all required binary assets are deliberately under this path, outside excluded result directories. Do not disable sparse checkout or reset/stash unrelated work.

On a new machine:

```sh
git clone --branch codex/rebuttal-gpu --single-branch \
  https://github.com/brendanmallery9/Linearized-Wasserstein-Dictionary-Learning.git WDL_repo
cd WDL_repo
```

Record `git rev-parse HEAD` with the run. Do not use an unreviewed newer branch revision mid-run.

## 3. Data and model access — included by Git

**No further data download is required for the primary evaluation.** `git fetch/pull` downloads the exact prepared holdout, reference support, models, probes, and smoke-test fixtures:

- `assets/holdout_540.pt`: raw normalized clouds [540,1024,3], clean barycentric maps [540,1000,3], labels, mesh archive paths and class order. These exact arrays are the evaluation data.
- `assets/reference_X.pt`: original 1,000-point source reference.
- `assets/frozen_models.joblib`: all 15 frozen models represented by cached residual dictionaries/baselines, neural encoder state or fitted sklearn dictionary, and **all 30 clean-trained linear probes/scalers per model**.
- `assets/checkpoints/*.pt`: 12 original full neural checkpoints, including potentials/free atoms and configuration; kept for provenance or later inspection. Primary evaluation uses the verified cached atoms, which avoids rebuilding Gibbs cost tensors.
- `assets/smoke_reference.pt`: prior 60-cloud codes/predictions/W2² for portability checks, **not scientific test results for this new study**.
- `assets/holdout_preparation.json`, `holdout_checks.json`, `previous60_preparation.json`, `original_split.json`, `source_checkpoints.json`, `SHA256SUMS.json`: identities, sampling/seeds and source/artifact checksums.
- `assets/prior_iid_anchors.json`: prior .00–.05 full-code/q1–5 values; label them prior EXP-005 anchors on the same objects, not newly independent replications.

`run.py --smoke` verifies **every asset hash** before unpickling. Load only this user-owned trusted bundle; package versions below matter for sklearn joblib compatibility. Never replace holdout rows or resample the surfaces for the primary run.

### Original raw dataset download (optional auditing/reconstruction)

Princeton **ModelNet10**, orientation-aligned OFF meshes; official page https://modelnet.cs.princeton.edu/download.html . Direct download verified by HTTP HEAD on2026-11-21:

```sh
mkdir -p /workspace/wdl-raw
curl --fail --location --retry 3 \
  https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip \
  -o /workspace/wdl-raw/ModelNet10.zip
sha256sum /workspace/wdl-raw/ModelNet10.zip
```

Expected archive: **473402300 bytes**, SHA256 **9d8679435fc07d1d26f13009878db164a7aa8ea5e7ea3c8880e42794b7307d51**. Original code used the HTTP URL, which currently redirects to this HTTPS URL. If unavailable, report it; do not silently substitute ModelNet40, a different split, pre-sampled normals, or different coordinates. The prepared arrays in Git suffice and avoid dependence on the legacy download host. Attribute Princeton ModelNet in any released results; included metadata preserves original identities. Sampling was uniform mesh surface sampling to1024 points, existing unit-cube normalization with.05 padding, exact uniform-mass EMD to the fixed1000 source support; metadata and the preparation source in `provenance/` document the precise procedure.

## 4. Environment and machine checks

Known instance setup: RTX3060/12GiB, WDL environment `/workspace/wdl-venv`, Python3.12, PyTorch2.5.1+cu124. Prefer that environment to the unrelated base image `/venv/main`. On Vast read `/etc/vast-agents-guide.md`, inspect running jobs, available GPU memory/CPU quota/disk first, and coordinate with other agents. No stop/destroy/billing/driver changes. There is no persistent volume: copy results off-instance before recycle/destroy.

```sh
nvidia-smi
nproc
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
df -h /workspace
/workspace/wdl-venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
uv pip check --python /workspace/wdl-venv/bin/python
```

If a new environment is necessary, create Python3.12 venv and install torch2.5.1 from the official cu124 wheel index, then the minimal pins in `requirements.txt`. Do not reinstall the whole repository's optional NLP stack. No API keys, private SSH keys or credentials are needed by the code, and none are included in this branch. Legacy paths inside original checkpoint configs are provenance only; the portable runner never opens them.

**Important performance fact:** encoder inference can use CUDA, but exact `ot.emd`/`ot.emd2` are CPU network-simplex solves. The remote machine's CPU parallelism is the main benefit. Keep exact solvers for maps and primary W2²; do not silently replace them with Sinkhorn, sliced-Wasserstein, a smaller point set, or a smaller geometry subset. Start with8 CPU workers / one BLAS and OT thread per call; use pilot timing and actual CPU quota/memory to choose more workers before the scientific run. CPU and CUDA are float32 for neural encoding, float64 CPU for OT; TF32 disabled. Classical models and probes are CPU.

## 5. Mandatory integration and timing pilot

```sh
PY=/workspace/wdl-venv/bin/python
$PY rebuttal/remote_corruptions/run.py --smoke --device cpu --workers 4
$PY rebuttal/remote_corruptions/run.py --smoke --device cuda --workers 4
$PY rebuttal/remote_corruptions/run.py --pilot --device cuda --workers 8 \
  --families iid --sigmas .075 --seeds 0 \
  --out /workspace/wdl-results/noise-integration-UNIQUE
```

Smoke checks: all15 CPU code reconstructions within atol2e-5/rtol2e-4; **every q1..30 probe prediction identical** to saved reference; CUDA codes close and all probe labels identical to CPU; exact W2² on two samples within1e-7; exact map reconstruction; generator determinism and matched-RMS depth controls. If CUDA predictions differ, preserve failure and investigate; use CPU encoding for the scientific run if required to retain exact probe behavior, recording that choice. Do not relax numerical thresholds silently. Pilot uses six shapes (one/class), labeled integration-only; do not claim performance from it. Native checkpoints and model parameter counts are frozen; no gradients or optimizer are used.

## 6. Stage A / EXP-007: stronger isotropic Gaussian noise

Sigma `.075,.10,.15,.20`, corruption draws0/1/2, full540, all15 models. For each clean cloud p compute its fixed clean centroid c and radius r=max||p-c||; u=(p-c)/r. Add sigma*N(0,I3) independently per point in unit-sphere coordinates; return p'=c+r*u'. No recentering, renormalization, clipping, registration, or resampling after corruption. RNG is keyed by mesh SHA, sigma level and draw in namespace20261202, matching the declared local experiment. These are **noise standard deviations per coordinate** in the sphere frame, not raw cube-frame units.

```sh
$PY -u rebuttal/remote_corruptions/run.py --device cuda --workers 8 \
  --families iid --sigmas .075 .10 .15 .20 --seeds 0 1 2 \
  --max-hours 12 --out /workspace/wdl-results/exp007-iid-UNIQUE
```

This includes clean plus12 noise conditions. Read the local status/ownership notes before launching: the Mac's former idle EXP-007 queue is retired by this handoff to avoid duplicate independent execution. Existing EXP-005 on the Mac is a different lower-noise/seven-type experiment and continues untouched.

## 7. Stage B / EXP-008: correlated depth noise and scan stripes

Execute after Stage A, **without selecting the levels/families based on which method won**. Same four sigma values, three draws and full540; compare these three families:

1. `depth_iid`: control with independent scalar Gaussian depth errors.
2. `depth_smooth`: spatially correlated depth field,32 random Fourier features with transverse frequencies N(0,ell^-2 I2), ell=.25 sphere units; coefficients N(0,1), phases U(0,2pi).
3. `scan_stripes`: sinusoidal depth stripes with wavelength .25 sphere units and random phase.

A mesh/draw-specific random unit view vector v is shared across all depth families. Build deterministic orthonormal transverse axes a,b; s=(u·a,u·b). Draw f(s) for the selected family, subtract its **empirical mean**, normalize to unit empirical RMS and apply delta(u)=sqrt(3)*sigma*f(s)*v. Thus each depth family has **identical 3D displacement RMS sqrt(3)*sigma**, matching the expected RMS of isotropic sigma noise; sigma is an isotropic-equivalent amplitude, not the scalar depth standard deviation. This separates correlation from noise direction/amplitude. Fields/view axes are fixed across amplitudes within each mesh/draw; all methods see identical clouds. New namespace20261204. Store field parameters/view axes and empirical normalization; fail on a degenerate field rather than dropping a shape. This tests zero-mean structured error, not arbitrary global translation.

```sh
$PY -u rebuttal/remote_corruptions/run.py --device cuda --workers 8 \
  --families depth_iid depth_smooth scan_stripes \
  --sigmas .075 .10 .15 .20 --seeds 0 1 2 --max-hours 12 \
  --out /workspace/wdl-results/exp008-depth-UNIQUE
```

This includes clean plus36 noise conditions. Sigma/ell/wavelength are fixed before outcomes; no cherry-picking orientation, object class or best corruption seed. They are explicit synthetic acquisition stress tests, not official PointCloud-C severities. Mark the severe regime if shape/class information collapses.

## 8. Models, metrics, success criteria and uncertainty

Frozen models: selected contribution-weighted EOT grid9 eta=.001/free eta=.01 and hard-k3 EOT/free (each seeds0/1/2); PCA30, signed SC30, positive SC30 (one dictionary each). All use the original1000 source points and fixed training-mean baseline. Full reconstruction B+sum_j z_j D_j, D_j=A_j-B. Encoder input <map-B,D_j>_rho; neural ReLU/LISTA20 steps, optional final hard-k3; classical transforms exactly as serialized. Do not regenerate atoms differently, normalize their norms, constrain coefficients to a simplex, or retrain a probe. Such changes would be separate experiments.

For each noisy raw cloud recompute exact EMD barycentric embedding onto the same source. Target is the **original clean raw cloud**, not the noisy observation. Primary reconstruction: full/native exact uniform W2² (not sqrt, regularized transport, or rescaled XYZ cost). Also report symmetric unsquared Chamfer and clean map half-L2rho at q1/3/5/30. Full-code accuracy and linear-probe accuracy for each retained q1..30: select per-example largest |z_j|sqrt(mean_source||D_j||²), preserve original coordinate IDs. Each q has its own frozen clean-trained scaler/classifier; original C selected on clean validation. q active entries are not a fixed q-dimensional feature space, and hard-k3 plateaus after3.

Preserve per-shape predictions, codes, error vectors, mesh IDs, corruption fields, all stages/seeds and warnings. Plot separately for **each family and sigma**; do not make a pooled corruption average the primary result. Average the3 corruption draws within each trained dictionary first, then mean/sample SD over the3 neural dictionaries. Classical dictionaries have n=1: draws do not become training seeds.

Primary contrasts: selected constrained minus PCA and positive SC, native W2² at every new sigma. Stage A has8 contrasts; Stage B has24, with separate prespecified families. Report conditional class-stratified paired shape-bootstrap95% intervals (20k replicates; keep all model/seed/draw predictions for each shape together), plus familywise Bonferroni percentile bounds for each stage. Report the seed SD separately; intervals do not estimate uncertainty over an initialization population. Every type/sigma outcome remains visible. Classifier/q endpoints and tuned-free contrasts are companions, not a substitute primary endpoint selected after seeing results.

Call a useful reconstruction crossover only with lower absolute error against **both** required baselines at the same level; provide adjusted bounds and clean-adjusted degradation. Also compare with raw corrupted-input W2² and common mean-shape floor, show classifier accuracy and obvious visual collapse: regression to an average at chance classification is not recovery of structure. If no crossover, report that. Better error growth is not lower absolute error, a clean advantage is not automatically extra robustness, and performance of these recipes is not a universal constraint theorem. Include tuned-free and both SC signs; low-activity or hard-k3 wins against a weak control do not establish general superiority.

## 9. Long-running execution, resumption, outputs and return report

Use Supervisor on Vast (read its guide); do not rely on an interactive SSH connection. Example config `supervisor.conf.example` runs `stages.py`, which executes Stage A, then Stage B only after A succeeds, then numerical audit/reporting; set the absolute worktree/output/log paths, check the program name is unique, `supervisorctl reread`, `supervisorctl update`, inspect its log. No web port or Caddy change is needed for a batch experiment. `stages.py` records stage/worker/exit status in its output root; add `--resume` only after inspecting an interrupted run. Both stages keep their separate protocols/deadlines. No stopping other agents' jobs or changing instance billing.

The output path must be new unless `--resume`. Each condition/model has a JSON completion marker and NPZ. Resume with the identical command plus `--resume`; source, assets, device/workers, family/seed/sigma settings must match. Do not reset the deadline or overwrite completed data. Back up any interrupted partial NPZ before resumption; a failed/partial run is not completed. `--max-hours` is a hard watchdog per run; ≥2GiB free storage required while running. Plan12h/≤8GiB output per stage initially; adjust a declared budget from pilot BEFORE launching, not silently when it expires. GPU idle percentages do not measure CPU OT progress.

Expected artifacts: protocol with commit/diff/environment/asset hashes/identities; condition inputs/maps+field metadata; per-model q metrics and per-shape NPZ; raw-input/mean floors; stdout/stderr; status, completion or failure/budget markers. The portable runner supplies these. **The supplied `report.py` (called by `stages.py`) performs aggregation, bootstraps, per-sigma plots and saved-array auditing. The agent must review and visually inspect those outputs**, and provide fixed-example reconstruction checks as described above: group models within the same family/sigma only, validate JSON means against NPZ, confirm all expected case/model markers, inspect per-sigma atom-budget and native error/accuracy curves and fixed-example reconstructions, retain negative results. Do not copy the old Mac analysis scripts without adapting paths, actual sample sizes, sigma/family counts and interval families.

Return a concise REPORT.md with exact commands/commit/versions/device/workers/data hashes; start/end/timezone and stage exits; all-sigma tables for reconstruction/native classification/q-curves; paired contrasts/uncertainty; floor/collapse checks; numerical and visual QA; limitations; output paths/checksums. Mark scientific versus integration-only, numeric versus visually reviewed. Copy raw artifacts off the ephemeral instance and return small report/CSV/figures to the main research record. Update its RESULTS_SUMMARY only with measured findings; proposed classes or expected wins are not results.
