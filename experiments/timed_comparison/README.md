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

Useful outputs:

- `manifest.json`: common benchmark parameters
- `heitz_wdl/history.jsonl`: baseline loss evaluations vs wall clock
- `mnist_ot_sae/history.jsonl`: map-prep event plus epoch losses vs wall clock
- `*/summary.json`: method-specific metadata, command lines, and elapsed time

On Apple Silicon, the Heitz wrapper defaults to `--heitz-avx off` and applies a
small scalar `dotp_full` fallback in the cloned external checkout.  That is fine
for local integration tests; use an AVX-capable x86 remote for final timing.

