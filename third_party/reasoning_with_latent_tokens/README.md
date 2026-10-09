# Anonymized Code Release

Code accompanying our submission. This codebase builds on the open-source
[Eso-LMs](https://arxiv.org/abs/2506.01928) codebase (Sahoo et al., 2025); the
upstream Apache 2.0 license is retained in `LICENSE`.

## Setup

```bash
conda create -n esolm python=3.9
conda activate esolm
pip install -r requirements.txt
```

## Code organization

- `main.py` — unified entry point. `config.mode` selects: `train` (default), `sample_eval` (unconditional generation), `completions` (conditional / puzzle completion), `ppl_eval` (perplexity).
- `trainer_base.py` — base trainer class.
- `algo.py` — re-exports the algorithms: `AR` (`ar.py`), `DiffLM` (`difflm.py`, main diffusion model), `DiffuParallel` (`diffuparallel.py`), `MDLM` (`mdlm.py`).
- `models/` — denoising network architectures.
- `configs/` — Hydra configs (`algo/`, `data/`, `model/`, `sampling/`, ...).
- `synthetic_data/` — dataset generation and evaluation for sudoku, game-of-24, zebra, repeat, and the diffusion-vs-ar task suite.
- `scripts/` — SLURM launch scripts for the experiments (`scripts/icml/`) and evaluation utilities.

## Data

Synthetic datasets (sudoku, game-of-24, zebra, repeat) are generated on the fly
and cached.

The `dvar-*` datasets (sudoku, countdown, 3-SAT, path) come from the public
diffusion-vs-ar repository (Ye et al., "Beyond Autoregression: Discrete
Diffusion for Complex Reasoning and Planning"). Clone it into the repository
root as `diffusion-vs-ar/` so the paths in `configs/data/dvar-*.yaml` resolve.

Inspect any dvar dataset with `python scripts/inspect_dvar.py`.

## Training

Local (no SLURM):

```bash
python main.py --config-name=experiment_base \
  data=sudoku-small \
  algo=difflm \
  mode=train
```

On a SLURM cluster:

```bash
sbatch scripts/icml/train_master.sh --method=<method> --data=<data> [--resume|--no-resume]
# e.g.
sbatch scripts/icml/train_master.sh -m ar -d sudoku-small
sbatch scripts/icml/train_master.sh -m diffu-causal-output-sminy -d game-of-24 --no-resume
```

Method names follow `{noise}-{attention}[-{model_size}][-{extra}]`, e.g.
`diffu-causal-output-sminy` = diffusion with `causal_output` attention and the
`sminy` model size.

## Evaluation / sampling

```bash
sbatch scripts/icml/gen_master.sh --method=<method> --data=<data> [--steps=N] [--batches=N] [--ckpt=<name>]
```

Sweeps over sampling steps and latent tokens:

```bash
sbatch scripts/icml/sweep_latent_steps.sh --method=<method> --data=<data> \
  [--steps-list=32,64,128,256] [--latent-list=4,8,16,32]
```

## Key configuration options

- `algo.diffusion_attn_mode` — attention mode for masked diffusion:
  `full` (bidirectional), `causal`, `causal_context`, or `causal_output`
  (clean tokens attend to all tokens; masked tokens attend only to clean tokens).
- `sampling.n_latent_tokens` — number of additional masked positions included
  in the forward pass during inference (inference-time scaling parameter).
- `training.train_on_all_tokens` — compute the loss on all tokens instead of
  only solution tokens.

## Logging

Logging uses Weights & Biases. Set `wandb.entity` and `wandb.project` in
`configs/config.yaml` (entity is `null` by default; set `wandb.mode=disabled`
to turn logging off).

## Cluster environment

The SLURM scripts read the following environment variables (with sensible
defaults): `CONDA_PROFILE` (path to `conda.sh`), `HF_HOME`, `DATADIR` (base
directory for caches, runs, and checkpoints), and optionally `GCS_DIR`
(a `gs://` bucket for checkpoint mirroring; leave unset to disable).
