# WDL Experiments

This repository accompanies experiments on sparse dictionary learning in
linearized Wasserstein and Brenier-potential spaces. The same basic workflow is
used across images, hyperspectral spectra, point clouds, and language-model
activations: prepare the data, embed it, train a sparse autoencoder, and inspect
the learned atoms and codes.

Pretrained checkpoints and selected results are included where practical. The
scripts can also rebuild the intermediate data and train fresh models.

## Setup

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12.14 is the recorded interpreter version. Apple Silicon workflows use
PyTorch MPS where supported. The timing experiment additionally needs CMake and
ImageMagick's `magick` command; its legacy WDL baseline runs under Rosetta on
Apple Silicon.

## Start here

Each experiment has a launcher and a notebook. The launcher prepares anything
that is missing; the notebook turns the resulting artifacts into the analyses
and figures used in the paper.

| Experiment | Launcher | Main analysis |
| --- | --- | --- |
| HSI | `hsi/scripts/run_hsi_sae_pipeline.sh` | `hsi/analysis/notebooks/hyperspectral_SAE_analysis.ipynb` |
| MNIST | `mnist/scripts/run_mnist_sae_pipeline.sh` | `mnist/analysis/notebooks/visualize_mnist_LWDL.ipynb` |
| Planted MNIST mixtures | `mnist/scripts/run_convex_combos_02378_best.sh` | `mnist/analysis/notebooks/convex_combos_02378.ipynb` |
| ModelNet point clouds | `pointcloud/scripts/run_pointcloud_sae_pipeline.sh` | `pointcloud/analysis/pointcloud_WDL_analysis.ipynb` |
| Noised text | `llm/scripts/run_noised_luther_pipeline.sh` | `llm/analysis/notebooks/corrupted_text_analysis.ipynb` |
| Pile-100k | `llm/scripts/run_pile100k_pipeline.sh` | `llm/analysis/notebooks/the_pile_analysis.ipynb` |
| WDL/LWDL timing | `experiments/timed_comparison/run_timing_experiment.py` | `experiments/timed_comparison/wdl_lwdl_timing_results.ipynb` |

Run scripts from the repository root. Use `--help` to see all options.

## Hyperspectral imagery

The HSI workflow covers Pavia, Botswana, and Salinas A. It downloads the source
cubes, computes transport maps, and can train both transport-map and linear
SAEs. Existing checkpoints are reused unless training is explicitly requested.

```bash
for dataset in salinas_a pavia botswana; do
  bash hsi/scripts/run_hsi_sae_pipeline.sh \
    --dataset "$dataset" \
    --sae-mode both \
    --seeds "0 1 2 3 4"
done
```

The general HSI notebook examines atoms, codes, and corruption robustness. The
clean Pavia comparison against rank-10 NMF lives in
`hsi/analysis/notebooks/reconstruction_comparison.ipynb`.

For a full paper-scale rerun, train the five seeds and both representations with:

```bash
bash hsi/scripts/run_paper_hsi_models.sh
bash hsi/scripts/full_corruption_sweep_hyperspec.sh \
  --root datasets/hsi_paper_reproduction \
  --sae-mode both
```

The retained HSI artifacts are enough for the clean Pavia notebook and the
principal seed-0 robustness configurations, but not the entire five-seed sweep.
The Pavia cube itself is downloaded on demand and checked against a pinned
SHA-256 checksum.

## MNIST

The standard pipeline downloads MNIST, constructs OT maps, and trains the
transport-map SAE:

```bash
bash mnist/scripts/run_mnist_sae_pipeline.sh --device mps
```

Use `--device cuda` on a CUDA machine, `--skip-maps` to reuse existing maps, or
`--eval` to generate the evaluation outputs after training. The pretrained
checkpoint under `datasets/mnist_ot/SAE_params` can be used directly by the
notebook.

The planted-mixture experiment on digits 0, 2, 3, 7, and 8 has its own analysis
notebook. It matches learned atoms to planted atoms, measures coefficient
recovery, plots reconstructions and vector fields, and searches held-out MNIST
maps for nearest neighbors. Its selected checkpoint is under
`mnist/results/convex_combos_02378_train_mean_seed_sweep/final/seed_000`.

## Point clouds

The point-cloud experiment samples six ModelNet10 classes, uses a fixed
1,000-point chair as the base measure, computes transport maps, and trains the
LWDL model:

```bash
bash pointcloud/scripts/run_pointcloud_sae_pipeline.sh \
  --data-dir datasets/modelnet10_6cls_chair_ot \
  --output-dir pointcloud/results/modelnet10_6cls_chair_reproduction \
  --device mps
```

The launcher defaults reproduce the recorded configuration: seed 42, 30 atoms,
20 LISTA steps, 2,000 epochs, `epsilon=0.025`, and `c=1e-4`. The notebook covers
atoms, reconstructions, sparsity, linear probes, PCA/UMAP views, and the
principal-component ablation.

## Language-model activations

Both text workflows use `EleutherAI/pythia-410m-deduped` by default, embed its
residual activations into Brenier-potential space, and train sparse
autoencoders. Existing SAE directories are reused unless `--train` is passed;
use `--force-train` only when intentionally writing into a populated directory.

For the noised-text experiment:

```bash
bash llm/scripts/run_noised_luther_pipeline.sh --max-docs 400 --device cuda --train
```

For Pile-100k:

```bash
ROOT=datasets/pile100k_paper_reproduction \
  bash llm/scripts/run_pile100k_pipeline.sh --device cuda --train
```

The Pile defaults match the retained paper model: all available documents,
seed 42, 1,000 Gaussian source samples, 150 PCA components, one unnormalized
512-atom JumpReLU SAE with `c=1e-1`, 3,000 epochs, learning rate `5e-4`, and
batch size 1,024.

The noised-text checkpoints retained in this repository came from a short MPS
smoke run. Use a fresh 2,000-epoch run for a paper-scale reproduction.

## Timing experiment

The fixed-duration timing experiment compares WDL and LWDL on Pavia spectra and
MNIST at sample sizes 100 and 1,000. Every trial receives the same 1,000-second
budget, with no plateau stopping. The output records runtime, completed
iterations or epochs, termination status, numerical failures, reconstruction
error, and the major timing segments of the legacy WDL implementation.

```bash
python experiments/timed_comparison/run_timing_experiment.py \
  --wdl-lwdl-timing-run \
  --device cuda \
  --heitz-avx on
```

For a small Python-side smoke test that skips the legacy C++ baseline:

```bash
python experiments/timed_comparison/run_timing_experiment.py \
  --sample-sizes 10 \
  --duration-seconds 2 \
  --epochs 1 \
  --batch-size 8 \
  --base-supp-size 32 \
  --atoms 3 \
  --top-k 2 \
  --lista-steps 2 \
  --device cpu \
  --skip-heitz
```

The retained trial table is
`experiments/results/timing_experiment_latest/timing_table.csv`. The main
notebook formats the paper tables and plots runtime and Wasserstein
reconstruction error; `augmented_timing_results.ipynb` contains the more
detailed timing diagnostics.

## Data and model citations

Please cite the original datasets and pretrained model in work that uses these
experiments.

- **MNIST:** Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner,
  “Gradient-Based Learning Applied to Document Recognition,” *Proceedings of
  the IEEE*, 1998. [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791)
- **Pavia, Botswana, and Salinas A:** the scenes and ground truth are distributed
  by the University of the Basque Country's
  [Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
  collection. The page also records the original ROSIS, Hyperion, and AVIRIS
  acquisition details.
- **ModelNet10:** Zhirong Wu et al., “3D ShapeNets: A Deep Representation for
  Volumetric Shapes,” *CVPR*, 2015.
  [Paper](https://openaccess.thecvf.com/content_cvpr_2015/html/Wu_3D_ShapeNets_A_2015_CVPR_paper.html)
- **The Pile:** Leo Gao et al., “The Pile: An 800GB Dataset of Diverse Text for
  Language Modeling,” 2020. [arXiv:2101.00027](https://arxiv.org/abs/2101.00027)
  This repository uses the
  [Pile-100k subset](https://huggingface.co/datasets/jannikbrinkmann/pile-100k).
- **Pythia:** Stella Biderman et al., “Pythia: A Suite for Analyzing Large
  Language Models Across Training and Scaling,” 2023.
  [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)

## Repository notes

- Generated data and checkpoints live under `datasets/`; paper figures and
  tables are generally produced by the analysis notebooks.
- Pipeline logs go to `logs/` unless an output directory is supplied.
- Retained PyTorch checkpoints omit deterministic geometry buffers such as the
  source support, target grid, and source-to-grid cost matrix. Repository
  loaders reconstruct these buffers and still require every learned parameter.
- Some workflows are expensive. Reuse flags such as `--skip-maps`,
  `--skip-embed`, `--skip-brenier`, and `--skip-train` when the corresponding
  artifacts already exist.
