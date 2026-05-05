# Timed MNIST Comparison

This harness compares:

- `heitz_wdl`: the C++ implementation from `matthieuheitz/WassersteinDictionaryLearning`
- `mnist_ot_sae`: this repo's MNIST OT-map SAE pipeline

The run output is a directory with a manifest, stdout logs, method summaries,
and JSONL histories.  For `mnist_ot_sae`, clock time includes OT-map
preparation before training.  For `heitz_wdl`, loss events are parsed from the
C++ binary stdout.

Quick local smoke test:

```bash
python experiments/timed_comparison/run_mnist_comparison.py \
  --preset smoke \
  --device mps
```

Larger local run:

```bash
python experiments/timed_comparison/run_mnist_comparison.py \
  --preset local \
  --device mps
```

CUDA run on a remote machine:

```bash
python experiments/timed_comparison/run_mnist_comparison.py \
  --preset local \
  --device cuda \
  --heitz-avx on
```

Heitz parameter sweep with CPU and CUDA OT-SAE runs:

```bash
python experiments/timed_comparison/run_mnist_heitz_sweep.py \
  --heitz-avx on
```

The sweep runs four Heitz settings:

- `gamma=0.5, sinkhorn_iters=5, max_optim_iter=500`
- `gamma=0.5, sinkhorn_iters=25, max_optim_iter=500`
- `gamma=2.0, sinkhorn_iters=5, max_optim_iter=500`
- `gamma=2.0, sinkhorn_iters=25, max_optim_iter=500`

It also runs `mnist_ot_sae` on CPU and CUDA for up to 500 epochs.  Each method
stops when either wall-clock time exceeds one hour or the moving average loss
over 10 iterations/epochs fails to improve by more than `1e-5`.  The final
sweep output is `summary_table.csv`, `summary_table.pkl`, and
`summary_table.png`; the old sweep loss-curve PNG is no longer generated.

Useful outputs:

- `manifest.json`: common benchmark parameters
- `heitz_wdl/history.jsonl`: baseline loss evaluations vs wall clock
- `mnist_ot_sae/history.jsonl`: map-prep event plus epoch losses vs wall clock
- `shared_ot_reconstruction.json`: common OT reconstruction error for both methods
- `*/summary.json`: method-specific metadata, command lines, and elapsed time

The shared reconstruction file reports `w2_squared = ot.emd2(...)` and `w2`
between each original MNIST image measure and each method's final reconstructed
measure, using squared Euclidean cost on normalized `[0,1]^2` coordinates.
Heitz reconstructions are read from final fitting PNGs; MNIST OT-SAE
reconstructions are the uniform pushforwards of the reconstructed transport maps.

On Apple Silicon, the Heitz wrapper defaults to `--heitz-avx off` and applies a
small scalar `dotp_full` fallback in the cloned external checkout.  That is fine
for local integration tests; use an AVX-capable x86 remote for final timing.
