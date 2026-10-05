# WDL Experiments

This repository contains experiments for sparse autoencoder and Wasserstein/Brenier
embedding pipelines across several settings:

- hyperspectral imagery (HSI)
- MNIST transport maps
- ModelNet point clouds
- language model activations (LLM)
- runtime and reconstruction comparisons against nonlinear WDL

Each experiment family has two layers:

- shell scripts that generate data, embeddings, transport maps, SAE checkpoints,
  corruption sweeps, and other intermediate outputs
- notebooks that consume those outputs to generate figures and visualize results

Run the relevant script first, then open the matching notebook. For
HSI, `run_hsi_sae_pipeline.sh` feeds the first part of
`hyperspectral_SAE_analysis.ipynb`, while `full_corruption_sweep_hyperspec.sh`
feeds the corruption-sweep figures in the second part. For MNIST,
`run_mnist_sae_pipeline.sh` feeds `visualize_mnist_LWDL.ipynb`. For LLMs,
`run_noised_luther_pipeline.sh` feeds `corrupted_text_analysis.ipynb`, and
`run_pile100k_pipeline.sh` feeds `the_pile_analysis.ipynb`. The paper-specific
notebooks for planted MNIST recovery, clean Pavia reconstruction, ModelNet point
clouds, and timing comparisons are described together under
[Analysis Notebooks](#analysis-notebooks).

## Setup

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the Python packages used by the reproduction workflows.
Python 3.12.14 is the recorded interpreter version. The timing comparison also
needs CMake and the ImageMagick `magick` command. On Apple Silicon, the legacy
Heitz WDL binary is built and run under Rosetta/x86-64 by the timing launcher;
the remaining local workflows use native PyTorch MPS.

## Repository Layout

```text
hsi/
  scripts/                  HSI pipeline launch scripts
  pipeline/                 HSI data, transport map embedding, SAE, and sweep code
  analysis/notebooks/       HSI result notebooks

llm/
  scripts/                  LLM pipeline launch scripts
  pipeline/                 Text corruption, latent space embedding, Brenier potential embedding, and SAE code
  analysis/notebooks/       LLM result notebooks

mnist/
  scripts/                  MNIST pipeline launch scripts
  pipeline/                 MNIST OT map embedding and SAE training code
  analysis/notebooks/       MNIST visualization notebook

pointcloud/
  scripts/                  ModelNet preparation and experiment launch scripts
  pipeline/                 Point-cloud OT map and SAE code
  analysis/                 Point-cloud result notebooks

experiments/timed_comparison/
  run_timing_suite_mark2.py  WDL/LWDL benchmark entry point
  *.ipynb                    Timing result and diagnostic notebooks

datasets/                   Generated or downloaded data and model outputs
logs/                       Pipeline logs
```

## Quick Start

The notebooks are set up to work as-is with the
pretrained weights under `datasets/`; the pipeline scripts
download data, compute transport maps/potentials, and regenerate intermediate outputs. 
The pipelines can also train additional models from scratch but skip this by default.

```bash
# HSI
for dataset in salinas_a pavia botswana; do
  bash hsi/scripts/run_hsi_sae_pipeline.sh --dataset "$dataset" --sae-mode both --seeds "0 1 2 3 4"
done
jupyter notebook hsi/analysis/notebooks/hyperspectral_SAE_analysis.ipynb

# MNIST
bash mnist/scripts/run_mnist_sae_pipeline.sh
jupyter notebook mnist/analysis/notebooks/visualize_mnist_LWDL.ipynb

# LLM, noised Luther
bash llm/scripts/run_noised_luther_pipeline.sh --device cpu
jupyter notebook llm/analysis/notebooks/corrupted_text_analysis.ipynb

# LLM, Pile-100k
bash llm/scripts/run_pile100k_pipeline.sh --device cpu
jupyter notebook llm/analysis/notebooks/the_pile_analysis.ipynb
```

## Analysis Notebooks

The notebooks below consume prepared datasets, checkpoints, and result tables;
they do not generally regenerate the expensive upstream artifacts themselves.
Run the corresponding pipeline first, or use the retained artifacts under
`datasets/` and `experiments/results/`.

### MNIST

- `mnist/analysis/notebooks/visualize_mnist_LWDL.ipynb` analyzes the standard
  MNIST transport-map SAE. It loads trained checkpoints, visualizes learned
  atoms and reconstructions, and examines the learned codes with class-aware
  plots and probes. Run `mnist/scripts/run_mnist_sae_pipeline.sh` first when the
  required maps or checkpoints are not already available.
- `mnist/analysis/notebooks/convex_combos_02378.ipynb` analyzes the planted
  dictionary experiment on digits 0, 2, 3, 7, and 8. It loads the selected
  train-mean-centered checkpoint, matches learned atoms to the five planted
  atoms, measures sparse-coefficient recovery, visualizes held-out mixture
  reconstructions and transport vector fields, and finds nearest held-out MNIST
  maps for each learned atom. It expects the prepared `convex_combos_02378`
  dataset, the corresponding MNIST OT maps, and the selected seed-sweep
  checkpoint.

### Hyperspectral imagery

- `hsi/analysis/notebooks/hyperspectral_SAE_analysis.ipynb` is the general HSI
  analysis notebook. Its first part inspects trained HSI SAEs, atoms, and
  reconstructions; its second part consumes the corruption-sweep tables and
  plots produced by `full_corruption_sweep_hyperspec.sh`. Run
  `hsi/scripts/run_hsi_sae_pipeline.sh` before the model analysis and the full
  corruption sweep before the robustness plots.
- `hsi/analysis/notebooks/reconstruction_comparison.ipynb` performs the clean
  Pavia comparison between transport-map SAEs and rank-10 NMF. It reconstructs
  spectra in the two representations, compares spectral-space and transport-map
  errors, summarizes the methods, and plots example spectra, transport maps,
  and learned atoms. It expects the Pavia cube, precomputed transport maps, and
  the selected Pavia SAE checkpoints.

### Point clouds

- `pointcloud/analysis/pointcloud_WDL_analysis.ipynb` analyzes the six-class
  ModelNet experiment. It displays the base measure and example clouds, loads a
  trained LWDL-EOT checkpoint, visualizes learned 3-D atoms and
  reconstructions, reports code sparsity and linear-probe accuracy, studies the
  codes with PCA and UMAP, runs the principal-component ablation, and generates
  the paper-style reconstruction and atom figures. It expects a dataset from
  `prepare_pointcloud_ot.py` and a checkpoint from
  `pointcloud_run_experiments.py` or `run_pointcloud_sae_pipeline.sh`.

### Language-model activations

- `llm/analysis/notebooks/corrupted_text_analysis.ipynb` analyzes the four
  synthetic-text SAE configurations. It compares atom usage across corruption
  types and strengths, visualizes codes with PCA and UMAP, and runs linear
  probes as the number of code principal components changes. Run
  `llm/scripts/run_noised_luther_pipeline.sh` first when the potentials or SAE
  checkpoints are not already available.
- `llm/analysis/notebooks/the_pile_analysis.ipynb` analyzes the 512-atom Pile
  model. It loads potentials together with document text and source metadata,
  computes codes and basic statistics, clusters and visualizes features,
  retrieves top documents by feature and cluster, ranks features by activation
  and source purity, exports source-specific document views, and includes an
  optional unigram/bigram association analysis. Run
  `llm/scripts/run_pile100k_pipeline.sh` first when the Pile potentials, PCA,
  Gaussian source, or SAE checkpoint are not already available.

### Timing comparison

- `experiments/timed_comparison/wdl_lwdl_timing_results.ipynb` loads the newest
  completed Mark 2 timing run, summarizes the Pavia and MNIST trial table, and
  compares WDL and LWDL using wall-clock time, completed iterations or epochs,
  termination status, numerical failures, and mean squared Wasserstein
  reconstruction error. It also formats the paper-style timing tables. Generate
  its inputs with `experiments/timed_comparison/run_timing_suite_mark2.py`.
- `experiments/timed_comparison/augmented_timing_results.ipynb` is the companion
  diagnostic notebook for augmented Mark 2 runs. It visualizes wall-clock
  timing, Heitz algorithm segment timing and percentages, timing versus
  reconstruction error, and optional loss histories, and can save the timing
  figures to the selected run directory.

## HSI

Use the HSI scripts to download and prepare hyperspectral data, compute transport
maps, optionally train SAEs, and run corruption robustness sweeps. The main HSI
pipeline does not train by default; pass `--train` only when you intentionally
want to write new SAE outputs. If the target SAE directory already has files,
the script refuses to train unless you also pass `--force-train`. The HSI
notebook can use the pretrained SAE checkpoints already under `datasets/hsi_data`.
The Pavia cube is downloaded on demand rather than bundled in the submission
archive. The download script tries the canonical source and a checksum-pinned
fallback, converts the source MAT file to `pavia_cube.pt`, and removes the raw
MAT file unless `--keep-raw-mat` is supplied.

`run_hsi_sae_pipeline.sh` is also the transport-map generator: unless
`--skip-maps` is supplied and unless complete maps already exist, it calls
`hsi/pipeline/compute_transport_maps.py` and writes the result under the
dataset's `transport_maps/` directory.

```bash
# Run the main HSI SAE pipeline for the datasets used in the analysis.
for dataset in salinas_a pavia botswana; do
  bash hsi/scripts/run_hsi_sae_pipeline.sh \
    --dataset "$dataset" \
    --sae-mode both \
    --seeds "0 1 2 3 4"
done
```

HSI datasets discussed here:

- `salinas_a`
- `pavia`
- `botswana`

Useful options:

```bash
bash hsi/scripts/run_hsi_sae_pipeline.sh --dataset pavia --epochs 1000 --seeds "0 1 2"
bash hsi/scripts/run_hsi_sae_pipeline.sh --dataset pavia --skip-download --skip-maps
bash hsi/scripts/run_hsi_sae_pipeline.sh --dataset pavia --sae-mode transport_maps
bash hsi/scripts/run_hsi_sae_pipeline.sh --dataset pavia --sae-mode both --train
bash hsi/scripts/run_hsi_sae_pipeline.sh --dataset pavia --sae-mode both --train --force-train
```

The paper-scale, from-scratch model launcher records the exact two SAE
configurations per method and dataset, prepares the three reported datasets,
computes their transport maps, and trains seeds 0--4 for 400 epochs. It writes
to a separate root by default so it cannot overwrite the retained artifacts:

```bash
bash hsi/scripts/run_paper_hsi_models.sh
```

The full corruption sweep evaluates those transport-map SAEs, linear/raw-space
SAEs, and NMF baselines. `full_corruption_sweep_hyperspec.sh` is restricted by
default to `pavia botswana salinas_a`, averages training and corruption seeds
`0 1 2 3 4`, uses dropout levels `0.1 0.2 0.3 0.4 0.5`, and uses the distinct
paper log-warp levels `1 10 100 500 1000`. The clean baseline is added
automatically. It expects all corresponding SAE checkpoints to exist.

Run the full HSI corruption sweep after the needed SAE models exist:

```bash
bash hsi/scripts/full_corruption_sweep_hyperspec.sh \
  --root datasets/hsi_paper_reproduction \
  --sae-mode both
```

The retained `datasets/hsi_data` tree is sufficient for the clean Pavia
reconstruction notebook and contains seed 0 for the principal robustness
configurations. It is not a complete five-training-seed archive; use the
paper-scale launcher above if the full averaged robustness study must be
regenerated. The code and retained checkpoints use `m=7` for Salinas A's
linear SAE, resolving the manuscript's isolated `m=15` table entry in favor of
the configuration used by the other Salinas results.

Notebook:

```text
hsi/analysis/notebooks/hyperspectral_SAE_analysis.ipynb
```

Dependencies:

- Run `hsi/scripts/run_hsi_sae_pipeline.sh` before using the first half of the
  notebook.
- Run `hsi/scripts/full_corruption_sweep_hyperspec.sh` before using the second
  half of the notebook.

## MNIST

Use the MNIST script to prepare OT maps, train the MNIST SAE, and optionally run
evaluation/visualization outputs. The MNIST notebook can use the pretrained
weights under `datasets/mnist_ot/SAE_params`; use `--skip-train` if you only
want to reuse them.

```bash
bash mnist/scripts/run_mnist_sae_pipeline.sh
```

Useful options:

```bash
# Use Apple Silicon or CUDA if available.
bash mnist/scripts/run_mnist_sae_pipeline.sh --device mps
bash mnist/scripts/run_mnist_sae_pipeline.sh --device cuda

# Reuse existing maps but retrain into a new output directory.
bash mnist/scripts/run_mnist_sae_pipeline.sh --skip-maps --output-dir datasets/mnist_ot/SAE_params_run2

# Train and run evaluation outputs.
bash mnist/scripts/run_mnist_sae_pipeline.sh --eval
```

Notebook:

```text
mnist/analysis/notebooks/visualize_mnist_LWDL.ipynb
```

Dependency:

- Run `mnist/scripts/run_mnist_sae_pipeline.sh` before
  `visualize_mnist_LWDL.ipynb`.

## Point clouds

The six-class ModelNet chair-base experiment has a complete data-to-checkpoint
launcher. This command reuses retained maps when they pass the metadata check,
otherwise prepares them, and writes a fresh checkpoint without touching the
historical result directory:

```bash
bash pointcloud/scripts/run_pointcloud_sae_pipeline.sh \
  --data-dir datasets/modelnet10_6cls_chair_ot \
  --output-dir pointcloud/results/modelnet10_6cls_chair_reproduction \
  --device mps
```

The launcher defaults encode the recorded run: six classes, chair mesh 0 as
the 1,000-point base, seed 42, 30 atoms, 20 LISTA steps, 2,000 epochs,
`epsilon=0.025`, and `c=1e-4`. The exact historical checkpoint is not included
in the repository. The command above recreates the run with the recorded seed
and parameters; no manual directory population is required.

Notebook:

```text
pointcloud/analysis/pointcloud_WDL_analysis.ipynb
```

## LLM

The LLM experiments are split into two separate pipelines: noised Luther text
and Pile-100k activations. Both pipelines embed text with
`EleutherAI/pythia-410m-deduped` by default, compute Brenier potentials, and
feed downstream notebooks. The LLM notebooks can use the pretrained SAE weights
under `datasets/noised_luther` and `datasets/pile-100k`; pipeline runs reuse
those weights unless `--train` is passed.

### Noised Luther

This pipeline generates corrupted versions of the base Luther text, embeds the
base and corrupted documents, computes Brenier potentials using the base
activation as the source, and prepares outputs for the corrupted-text notebook.
It reuses existing `SAE_params` by default; pass `--train` to train new SAEs.
If `SAE_params` already contains files, add `--force-train` or use a new
`ROOT`.

```bash
bash llm/scripts/run_noised_luther_pipeline.sh --device cpu
```

Common options:

```bash
# Standard noised-Luther run size.
bash llm/scripts/run_noised_luther_pipeline.sh --max-docs 20 --device cpu

# Paper run size (400 examples for each corruption/intensity pair).
bash llm/scripts/run_noised_luther_pipeline.sh --max-docs 400 --device cpu

# Apple Silicon.
bash llm/scripts/run_noised_luther_pipeline.sh --device mps

# Train new SAEs.
bash llm/scripts/run_noised_luther_pipeline.sh --device cpu --train

# Intentionally write into an existing SAE_params directory.
bash llm/scripts/run_noised_luther_pipeline.sh --device cpu --train --force-train
```

Notebook:

```text
llm/analysis/notebooks/corrupted_text_analysis.ipynb
```

Dependency:

- Run `llm/scripts/run_noised_luther_pipeline.sh` before
  `corrupted_text_analysis.ipynb`.

For a from-scratch paper reproduction, preserve the bundled base document in a
new run root and train only the four reported models for 2,000 epochs:

```bash
run_root=datasets/noised_luther_paper_reproduction
mkdir -p "$run_root/txt/base"
cp datasets/noised_luther/txt/base/luther.txt "$run_root/txt/base/luther.txt"
ROOT="$run_root" bash llm/scripts/run_noised_luther_pipeline.sh \
  --max-docs 400 --epochs 2000 --device cuda --train
```

The four trials and their TopK/JumpReLU parameters are selected by the pipeline
itself. The checkpoints currently retained under
`datasets/noised_luther/SAE_params` record a five-epoch MPS smoke run, so use
the command above—not those retained checkpoints—for a claimed paper rerun.

### Pile-100k

This pipeline embeds Pile-100k documents, fits a Gaussian source and PCA
transform, computes Brenier potentials, and prepares outputs for the Pile
analysis notebook. It reuses existing `SAE_params` by default; pass `--train`
to train new SAEs. If `SAE_params` already contains files, add
`--force-train` or use a new `ROOT`.

```bash
bash llm/scripts/run_pile100k_pipeline.sh --device cpu
```

The defaults encode the retained paper checkpoint: all available documents
(`N_DOCS=-1`), seed 42, 1,000 Gaussian source samples, 150 PCA components, one
unnormalized 512-atom JumpReLU SAE with `c=1e-1`, 3,000 epochs, learning rate
`5e-4`, and batch size 1,024. For a fresh run without touching retained files:

```bash
ROOT=datasets/pile100k_paper_reproduction \
  bash llm/scripts/run_pile100k_pipeline.sh --device cuda --train
```

Common options:

```bash
# Apple Silicon.
bash llm/scripts/run_pile100k_pipeline.sh --device mps

# CUDA multi-process embedding.
bash llm/scripts/run_pile100k_pipeline.sh --device cuda --num-gpus 4

# Train new SAEs.
bash llm/scripts/run_pile100k_pipeline.sh --device cpu --train

# Intentionally write into an existing SAE_params directory.
bash llm/scripts/run_pile100k_pipeline.sh --device cpu --train --force-train
```

Notebook:

```text
llm/analysis/notebooks/the_pile_analysis.ipynb
```

Dependency:

- Run `llm/scripts/run_pile100k_pipeline.sh` before `the_pile_analysis.ipynb`.

The LLM scripts are configured to reuse some expensive existing outputs by
default. Check the configuration block printed at the start of each run, and
use flags such as `--skip-embed`, `--skip-gaussian`, `--skip-brenier`, and
`--skip-train` when you want to explicitly reuse existing intermediate files.
Use `--train` only when you intentionally want to write new SAE outputs, and
`--force-train` only when replacing or extending an existing SAE directory is
intentional.

## Outputs

Most generated artifacts are written under `datasets/`:

- `datasets/hsi_data/<dataset>/data/`: hyperspectral cubes
- `datasets/hsi_data/<dataset>/transport_maps/`: HSI OT transport-map outputs
- `datasets/hsi_data/<dataset>/SAE_params/`: trained HSI SAE checkpoints
- `datasets/noised_luther/`: noised text, activations, potentials, SAE outputs, and the included base `luther.txt`
- `datasets/pile-100k/`: Pile activations, potentials, included `source.pt`/`pca.pt`, and SAE outputs
- `datasets/mnist_ot/`: MNIST OT maps and SAE outputs

Pipeline logs are written to `logs/` unless a script-specific results directory
is provided.

## Notes

- Run commands from the repository root.
- To keep the submission archive small, the retained PyTorch checkpoints omit
  deterministic geometry buffers (`X`, the target grid, and the source-to-grid
  cost matrix) from their model state. Repository loaders reconstruct these
  buffers from the checkpoint metadata or prepared dataset and strictly verify
  that no learned parameter is missing.
- Many scripts accept environment variable overrides in addition to command-line
  flags. Use `-h` or `--help` on a script to see the available options.
- The notebooks assume the generated data paths used by the scripts above.
