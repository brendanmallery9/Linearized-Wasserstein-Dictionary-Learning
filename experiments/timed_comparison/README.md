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

## Three-Experiment Timing Suite

`run_timing_suite.py` runs the newer comparison suite:

- Pavia 1D subset: HSI 1D OT-map SAE, EBCM, potential SAE, and Heitz
- MNIST digits: EBCM, potential SAE, and Heitz
- synthetic centered Gaussian measures: EBCM vs potential SAE over dimensions
  `1, 5, 10, 20, 40`

In plain terms, the suite does the following.

### Experiment 1: Pavia 1D Spectra

This experiment uses a small random subset of pixels from the Pavia
hyperspectral image.  Each pixel is treated as a 1D distribution over spectral
bands: bright bands get more mass, dim bands get less.

Steps:

1. Pick a repeatable subset of Pavia pixels using the random seed.
2. Turn each selected spectrum into a probability distribution.
3. Build several embedded versions of the same spectra:
   - the HSI 1D method computes ordinary 1D optimal transport maps,
   - EBCM computes entropic transport maps for each requested epsilon,
   - the potential method computes OT potential vectors.
4. Train the matching autoencoder/dictionary model on each embedding.
5. For HSI 1D, EBCM, and Heitz, reconstruct measures and report mean
   `W_2^2` reconstruction error.
6. For the potential method, report reconstruction MSE in potential space,
   because decoded potentials are not directly measures.
7. Record embedding time, training time, stopping epoch/iteration, and error.

### Experiment 2: MNIST Digits

This experiment uses MNIST digit images as 2D probability distributions.  Each
nonzero pixel is a support point, and brighter pixels carry more mass.

Steps:

1. Pick a repeatable subset of MNIST digit images.
2. Save the same images as PNGs for Heitz.
3. Sample a fixed base measure with the requested number of support points.
4. Build EBCM embeddings by computing entropic transport maps from the base
   measure to each digit, once for each requested epsilon.
5. Build potential embeddings by computing an OT potential vector for each
   digit.
6. Train EBCM and potential autoencoders on their respective embeddings.
7. Run Heitz on the same digit PNGs, for the requested gamma values.
8. Report mean `W_2^2` reconstruction error for EBCM and Heitz.
9. Report potential-space reconstruction MSE for the potential method.
10. Record embedding time, training time, stopping epoch/iteration, and error.

### Experiment 3: Centered Gaussian Measures

This experiment is mainly a dimension-scaling test.  It generates synthetic
Gaussian point-cloud measures in dimensions `1, 5, 10, 20, 40`.  The Gaussians
are centered at zero, but their covariance shapes vary.

Steps:

1. For each dimension, generate and cache a repeatable synthetic dataset.
2. Use one fixed Gaussian point cloud as the source/base measure.
3. Generate several centered Gaussian target measures with random covariance
   matrices normalized to variance scale 1.
4. For EBCM, compute entropic transport maps for epsilons `0.1`, `0.05`, and
   `0.2 / dimension`.
5. For the potential method, compute OT potential vectors for the same target
   measures.
6. Train the EBCM and potential autoencoders.
7. Compare wall-clock time as dimension increases.
8. Report embedded reconstruction losses, not `W_2^2`, because this experiment
   is about timing as a function of dimension.

Quick Python-side smoke test, without building/running Heitz:

```bash
python experiments/timed_comparison/run_timing_suite.py \
  --preset smoke \
  --skip-heitz
```

Remote CUDA run with Heitz included:

```bash
python experiments/timed_comparison/run_timing_suite.py \
  --preset local \
  --device cuda \
  --heitz-avx on
```

The `local` preset keeps the Heitz-facing Pavia and MNIST datasets near the
old timed-comparison scale: `100` Pavia spectra and `10` MNIST images per digit
(`100` MNIST images total).  The Gaussian dimension sweep still uses `100`
synthetic measures per dimension because it does not run Heitz.

The same preset also keeps the neural model size close to the prior timing
comparison: `10` atoms, LISTA depth `20`, and batch size `256`.

The Pavia 1D HSI row uses the older HSI SAE training recipe except for the
suite batch size: learning rate `1e-4`, sparsity coefficient `5e-5`,
AdamW weight decay `1e-4`, cosine LR decay, gradient clipping at `100`, and the
old mean-activation sparsity penalty.

The default stopping rule matches the prior Heitz/OT-SAE sweep: neural methods
run for at most `500` epochs, Heitz runs for at most `500` optimization
iterations, all methods have a one-hour wall-clock cap, and all methods stop
early if the moving-average loss over `10` epochs/iterations fails to improve by
more than `1e-5`.

Run one experiment at a time:

```bash
python experiments/timed_comparison/run_timing_suite.py \
  --experiment mnist \
  --preset local \
  --device cuda \
  --heitz-avx on
```

The suite writes one table per experiment plus a combined table:

- `pavia1d/timing_table_pavia1d.csv`
- `mnist/timing_table_mnist.csv`
- `gaussian/timing_table_gaussian.csv`
- `timing_table_all.csv`
- `timing_table_all.html`

Each CSV table is also saved as a browser-friendly `.html` table in the same
directory.

Embeddings and synthetic Gaussian data are cached under
`experiments/results/timing_cache` by default.  Use `--force-cache` to recompute
embeddings and refresh their recorded embedding times.

The table reports `mean_w2_squared` for measure-reconstruction methods in the
Pavia and MNIST experiments.  Potential-method rows intentionally leave
`mean_w2_squared` blank and report `embedded_recon_mse`, the reconstruction MSE
in the potential embedding space.  The Gaussian dimension sweep is a timing
comparison, so it reports embedded reconstruction losses rather than W2.
