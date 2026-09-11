# DCache: recurrent computation for masked denoising

Research experiments built on BD3-LM/MDLM. DCache preserves token-aligned
computation at completed transformer depths between denoising forwards,
optionally alongside previous-final-hidden feedback.

We ask which internal computations are useful to preserve, how to train their
recurrence, and whether they can be revised when evidence changes. Current
pilots demonstrate sample-specific recurrence benefits; stronger workspace,
convergence and efficiency claims remain under investigation.

## Documentation

Two maintained documentation files serve this training repository:

- **This README:** navigation and common entry points.
- **[Research and experiment log](RESEARCH_EXPERIMENT_LOG.md):** exact methods,
  dated experiments/decisions, measured findings, open tasks, commands, literature
  review and complete mathematical derivations.

Start with [current design](RESEARCH_EXPERIMENT_LOG.md#current-design),
[results](RESEARCH_EXPERIMENT_LOG.md#measured-results-and-their-limits), or
[operating commands](RESEARCH_EXPERIMENT_LOG.md#operating-the-repository).
Append future ideas/day logs there instead of creating more progress files.

## Trials

| Launcher variant | What it tests |
|---|---|
| `vanilla` | Original one-state BD3/MDLM pretraining |
| `objective` | Same vanilla model with the five-state objective |
| `dcache-v2` | Five-state layerwise denoising memory |
| `final-state` | V2 plus detached previous-final-hidden feedback |
| `two-forward` | Two-state dual memory, connected DCache and detached final feedback |
| `final-state-adjacent` | Five-state dual memory with strictly one-hop DCache credit |

Main pilots train from scratch on OpenWebText, length/block size 1024, global
batch 512 and 5,000 optimizer updates. Equal updates are **not** equal compute.
Five-state losses are normalized; conditional NLL is not generative perplexity.

## Common commands

Run from this repository with the existing `dcache` Conda environment.
For a new server use [the environment installer](scripts/setup_dcache_environment.sh);
do not update an environment while training uses it.

### Continue the connected trial on one H100

For Lightning AI use [the portable H100 launcher](scripts/cloud/lightning_h100.sh),
not the old two-GPU installer. The verified transfer point is optimizer step
**1500** of the five-forward `final-state-adjacent` run; the target remains 5000.
Copy the full checkpoint **and both prepared OpenWebText cache directories**.
Exact locations, transfer instructions, storage requirements and resume caveats
are in [the H100 migration log](RESEARCH_EXPERIMENT_LOG.md#h100-cloud-continuation).

On the cloud machine, from the cloned repository on persistent storage:

```bash
bash scripts/cloud/lightning_h100.sh setup
# Manually transfer the checkpoint and prepared data before continuing.
bash scripts/cloud/lightning_h100.sh check
bash scripts/cloud/lightning_h100.sh validate  # optional step-1500 reference
bash scripts/cloud/lightning_h100.sh smoke    # one update, isolated test output
bash scripts/cloud/lightning_h100.sh tmux     # resume the real run to 5000
bash scripts/cloud/lightning_h100.sh plot
```

Setup uses the Studio's **existing active Python environment** (CPython
3.9–3.12); it never calls `conda create`. It installs pinned training packages
from [requirements-h100.txt](requirements-h100.txt), without the local notebook
tools. Do not run setup while a job uses that environment. Set `DCACHE_PYTHON`
to an existing interpreter if needed; all launcher actions honor it.

The launcher manages project-local caches and its tmux socket. It preserves
global batch 512, microbatch 2, the one-hop DCache gradient and detached
final-state feedback. CPU tests and transfer checks pass; package installation
and the H100 smoke test must still pass on the destination machine.

### Local plots and trials

For a space-limited cloud continuation, use the **compact 1500→5000 bundle**;
see [export and transfer instructions](RESEARCH_EXPERIMENT_LOG.md#compact-data-continuation).
It stores only required original packed rows and full validation, preserves the
original logical dataset length/permutation, and requires its own transfer
manifest. It is not a new smaller dataset to reshuffle or retokenize.

Refresh the registry-driven dashboard without running evaluation:

```bash
conda activate dcache
MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  python scripts/results/refresh_canonical_results.py
```

It covers vanilla, objective, V2, five-forward final state, and the current
**connected one-hop** run, with smoothing 60 and a 5,000-step cap. Runs without
validation contribute training curves only; new validation appears on refresh.
Two-forward is still outside this training dashboard.

Outputs: [training + validation](results/generated/figures/training/canonical_5k_smooth60.png)
and [validation only](results/generated/figures/training/canonical_validation_nll.png).

Refresh figures from the completed recurrence audit:

```bash
bash scripts/eval/run_urgent_recurrence_audit.sh --plot-only
```

Before training, follow the log's
[tmux/runtime setup outside /tmp](RESEARCH_EXPERIMENT_LOG.md#tmux-and-managed-runtime-paths-outside-tmp).
Choose a distinct run directory and check that GPUs 2,3 are available:

```bash
DCACHE_CUDA_VISIBLE_DEVICES=2,3 DCACHE_DEVICES=2 \
  bash scripts/train/run_canonical_trial.sh \
  final-state-adjacent outputs/NEW_DISTINCT_TRIAL
```

Select the desired variant from the table. The same recipe/directory resumes
its own `last.ckpt`; `max_steps` is an absolute target. Defaults retain the
latest three checkpoints every 500 updates, plus a `last.ckpt` symlink.
Do not reuse another variant's directory. The log includes validation,
comparison evaluation, safe resume, GPU smoke-test and storage instructions.

## Results and paper

- [Canonical run status](results/generated/tables/training/canonical_status.csv)
  and [training figures](results/generated/figures/training/).
- [Recurrence audit report](outputs/eval-urgent-recurrence-5k/comparison/REPORT.md)
  and [same-input figure](outputs/eval-urgent-recurrence-5k/comparison/same_state_recurrence.png).
- [Paper source](iclr2027-paper/main.tex), in an independent nested Git repository.
  Its operational Markdown, official style package, figures and frozen
  data/provenance remain there.

Prepare the local Overleaf upload package:

```bash
make -C iclr2027-paper overleaf
```

This checks/builds the paper and source ZIP; it does not upload or synchronize
remotely. See the log before refreshing frozen experiment data.

## Layout and archives

Training code remains in `models/`, `diffusion.py` and `configs/`; launchers and
evaluation tools in `scripts/`. Raw runs stay in `outputs/`; curated figures and
tables stay in `results/generated/`.

Old notes are preserved in [a dated source ZIP](legacy/documentation/2026-09-09/source_notes.zip),
with [hashes and a source map](legacy/documentation/2026-09-09/manifest.json).
Original BD3 figures, paper PDF and license text are in
[legacy/upstream/bd3](legacy/upstream/bd3/).
Generated reports and existing result archives keep their paths: they are
evidence artifacts, not additional maintained guides.

## Attribution and license

Derived primarily from the official
[kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms)
implementation of *Block Diffusion: Interpolating Between Autoregressive and
Diffusion Language Models*, by Marianne Arriola, Aaron Gokaslan, Justin T Chiu,
Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo and Volodymyr Kuleshov.
This is an independent experiment, not an official BD3-LM release.

The original Apache 2.0 text remains accessible at [LICENSE](LICENSE), linked
to its archived copy. Source notices are preserved. The ZIP retains the
complete pre-consolidation README, including upstream documentation, and the
SSD-LM README; it is a historical snapshot, not a pristine upstream checkout.
