# WDL Experiments

This repository contains experiments for sparse autoencoder and Wasserstein/Brenier
embedding pipelines across three settings:

- hyperspectral imagery (HSI)
- MNIST transport maps
- language model activations (LLM)

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
`run_pile100k_pipeline.sh` feeds `the_pile_analysis.ipynb`.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU or Apple Silicon runs, install the PyTorch build appropriate for your
machine before running the pipelines.

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
bash hsi/scripts/full_corruption_sweep_hyperspec.sh --root datasets/hsi_data --sae-mode both
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

## HSI

Use the HSI scripts to download and prepare hyperspectral data, compute transport
maps, optionally train SAEs, and run corruption robustness sweeps. The main HSI
pipeline does not train by default; pass `--train` only when you intentionally
want to write new SAE outputs. If the target SAE directory already has files,
the script refuses to train unless you also pass `--force-train`. The HSI
notebook can use the pretrained SAE checkpoints already under `datasets/hsi_data`.

```bash
# Run the main HSI SAE pipeline for the datasets used in the analysis.
for dataset in salinas_a pavia botswana; do
  bash hsi/scripts/run_hsi_sae_pipeline.sh \
    --dataset "$dataset" \
    --sae-mode both \
    --seeds "0 1 2 3 4"
done

# Run the full HSI corruption sweep after the needed SAE models exist.
bash hsi/scripts/full_corruption_sweep_hyperspec.sh --root datasets/hsi_data --sae-mode both
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

The full corruption sweep evaluates transport-map SAEs, linear/raw-space
SAEs, and NMF baselines across corruption types and severities.
`full_corruption_sweep_hyperspec.sh` averages results over seeds. By default,
the corruption sweep uses corruption seeds `0 1 2 3 4` and training seeds
`0 1 2 3 4`. It expects the corresponding SAE checkpoints to already exist.

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

# Thesis run size.
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

### Pile-100k

This pipeline embeds Pile-100k documents, fits a Gaussian source and PCA
transform, computes Brenier potentials, and prepares outputs for the Pile
analysis notebook. It reuses existing `SAE_params` by default; pass `--train`
to train new SAEs. If `SAE_params` already contains files, add
`--force-train` or use a new `ROOT`.

```bash
bash llm/scripts/run_pile100k_pipeline.sh --device cpu
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
- Many scripts accept environment variable overrides in addition to command-line
  flags. Use `-h` or `--help` on a script to see the available options.
- The notebooks assume the generated data paths used by the scripts above.
