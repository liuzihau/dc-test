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

For explicit H100 microbatch trials, set `export DCACHE_MICRO_BATCH=4`, `8` or `16`
before smoke/train/tmux. Global batch remains 512; accumulation becomes 128,
64 or 32 respectively. Validation remains microbatch 2 × 200 batches. Training's
shuffled-cache identity comparison group changes with the training microbatch.
Separate default output directories ending in `-h100-mb4`, `-h100-mb8` or `-h100-mb16` keep
these trials distinct. Run smoke on the target GPU before full training.

### Restart-safe cloud training

Keep the same checkpoint/data/manifest/microbatch/**run directory** exports as
your current cloud trial. After syncing all new source files and stopping any
previous GPU job, enable local-disk recovery:

```bash
export DCACHE_RECOVERY_ENABLED=1
export DCACHE_RECOVERY_SECONDS=1200
bash scripts/cloud/lightning_h100.sh smoke &&
bash scripts/cloud/lightning_h100.sh arm &&
bash scripts/cloud/vast_onstart.sh
```

`arm` persists the launch settings; `vast_onstart.sh` starts the guarded
background supervisor. **Also add** `bash /workspace/dc-test/scripts/cloud/vast_onstart.sh`
to your Vast template's existing startup/onstart hook (adjust the repo path).
Do not replace the template's other SSH/Jupyter startup commands. This manual
registration is required for reboot recovery; running it once is not registration.

Recovery saves at the next completed optimizer update after roughly 20 minutes,
also at 500-step boundaries, and after final validation. It rotates three new
recovery checkpoints, verifies restart candidates and falls back if needed.
Old periodic/imported checkpoints are not deleted. Validation stays every 500
optimizer steps. Retries are bounded; disk-full, OOM and configuration errors
stop for inspection. The host/GPU must actually return, and its disk must survive;
no off-machine backup or automatic instance rental is configured.

Logs: `logs/recovery-supervisor.log`, and `<RUN_DIR>/supervisor_status.json`.
Plot as usual with `bash scripts/cloud/lightning_h100.sh plot`. Restart-aware
plots discard abandoned loss tails using per-attempt resume metadata, retaining
raw CSVs. Merged CSVs are also written under `results/generated/tables/training/`.

See [the recovery work note](RESEARCH_EXPERIMENT_LOG.md#restart-safe-cloud-recovery)
for guarantees, limitations and verification.

### Local plots and trials

Current-preserving merged trial (2026-09-15): disables previous-V attenuation
and the 20% cache-only query route. At t2/t3, masked queries use 95% joint
current/previous attention and 5% current-only attention. Normal AdaLN gates,
2D RoPE, final-state dropout, identity loss, trajectory weights and one-hop
gradients remain unchanged. This is a **fresh experiment**, not a fix to apply
while resuming the old run. It has not yet demonstrated improved quality.

After activating `dcache` and making sure CUDA 2,3 are free:

```bash
unset DCACHE_RESUME_CKPT DCACHE_PAIR_RUN_DIR
bash scripts/train/train_owt_dcache_merged_pair.sh 3090 off smoke --attention-policy current-preserving && \
  bash scripts/train/train_owt_dcache_merged_pair.sh 3090 off tmux --attention-policy current-preserving
```

Use `h100` instead of `3090` for cloud GPU 0, and `on` instead of `off` for the
neighbor auxiliary. The new output folder and tmux session end in
`-current-preserving`. Its own compatible checkpoint resumes automatically;
cross-policy resumes are rejected. Logs, temporary files and the tmux socket
stay under this project. See [exact settings and verification](RESEARCH_EXPERIMENT_LOG.md#current-preserving-merged-attention).

Historical single-attention ablation (legacy gate/dropout, full prepared data):
`bash scripts/train/train_owt_dcache_merged_adjacent_5k.sh`.
This retains the connected five-forward + detached final-state recipe and uses
one shared-QKV attention with spatial/iteration-age 2D RoPE. It is not compatible
with old two-attention checkpoints or the compact 1500–5000 continuation bundle.
See [scope and comparison caveats](RESEARCH_EXPERIMENT_LOG.md#single-attention-2d-rope-ablation).

Masked-neighbor auxiliary trial (merged attention only):
`bash scripts/train/train_owt_dcache_merged_neighbors_5k.sh`.
Adds two independent final-hidden LM heads for previous/next masked targets,
with `0.5 * (previous CE + next CE) / 2`; a clean source position is allowed.
Defaults to adjacent-only DCache credit; set `DCACHE_GRADIENT_MODE=detached`
for the disconnected variant. Both use detached final-state feedback, distinct
output directories and unchanged primary NLL records. This is fresh training,
not continuation from an old checkpoint. See the [exact recipe, smoke commands
and workspace experiment plan](RESEARCH_EXPERIMENT_LOG.md#masked-neighbor-auxiliary-trial).

Historical matched auxiliary **off/on**, on local 2×3090 or cloud 1×H100
(omitting `--attention-policy` intentionally keeps the legacy experiment):

```bash
# Run after activating the training environment and transferring FULL prepared data.
bash scripts/train/train_owt_dcache_merged_pair.sh 3090 off smoke && \
  bash scripts/train/train_owt_dcache_merged_pair.sh 3090 off tmux
bash scripts/train/train_owt_dcache_merged_pair.sh h100 on smoke && \
  bash scripts/train/train_owt_dcache_merged_pair.sh h100 on tmux
```

Run the first line on the local server and the second on the H100 server.
Either hardware supports `off` or `on`. Both use merged 2D-RoPE attention,
five-forward adjacent-only DCache gradients and detached final feedback.
Defaults are CUDA `2,3` / `0`, microbatch 2, global batch 512, 5,000 updates,
and 400 validation examples every 500 updates. Each variant has its own output
folder, training lock and project-local tmux socket; smoke uses a separate
temporary folder and does not advance the main run. The launcher prints the
attach command. It enables the new explicit full-data cursor restore for its
own checkpoints; historical runs keep their previous default behavior.
This is **not** the old Vast automatic-recovery launcher: tmux survives an SSH
disconnect, not an instance shutdown. See [complete environment setup, resume
limits and cross-hardware comparison caveats](RESEARCH_EXPERIMENT_LOG.md#merged-auxiliary-off-on-launch-pair).

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

It covers vanilla, objective, V2, five-forward final state, the **connected
one-hop** run, and **merged one-hop + final state without auxiliary heads**,
with smoothing 60 and a 5,000-step cap. Runs without
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

### Sudoku, Zebra and Countdown pilots

The isolated [reasoning pipeline](scripts/reasoning/run_reasoning.py) supports
all three tasks with protected clues, task-specific solving metrics, and matched
merged-attention controls. It reuses our transformer; **it is not an exact
reproduction of the unreleased Latent Tokens implementation**. Synthetic data
is labelled pilot data; normalized external JSONL and preserved splits can also
be imported. Read the [method and limitations](RESEARCH_EXPERIMENT_LOG.md#2026-09-14--sudoku-zebra-and-countdown-reasoning-pipeline)
before spending a full training budget.

```bash
conda activate dcache
# Small CPU plumbing checks, isolated from real data/checkpoints and all GPUs:
for task in sudoku zebra countdown; do
  bash scripts/train/train_reasoning.sh cpu "$task" both_aux smoke
done

# Prepare a shared pilot dataset ONCE. Default is only 1,000 training examples.
# Choose larger pilot sizes explicitly; these are still not the paper's splits.
REASONING_TRAIN_EXAMPLES=20000 REASONING_VALID_EXAMPLES=1000 \
REASONING_TEST_EXAMPLES=1000 \
  bash scripts/train/train_reasoning.sh 3090 sudoku mdm prepare

# Matched no-memory + auxiliary vs dual-memory + auxiliary (run sequentially):
bash scripts/train/train_reasoning.sh 3090 sudoku mdm_aux tmux
# After that run finishes:
bash scripts/train/train_reasoning.sh 3090 sudoku both_aux tmux

# Full model-generated solving, or latest training/validation plot:
bash scripts/train/train_reasoning.sh 3090 sudoku both_aux evaluate
bash scripts/train/train_reasoning.sh 3090 sudoku both_aux plot
```

Replace `sudoku` with `zebra`/`countdown`; prepare each task separately. Replace
`3090` with `h100` for one cloud GPU. Defaults are CUDA `2,3` / `0`, global batch
128, microbatch 8, 5,000 optimizer updates, validation and periodic checkpoints
every 500 updates, and 20-minute checkpoint saves at completed updates. Only the
latest three full checkpoints are retained. A rerun resumes that variant's own
`last.pt`; do not point this launcher at an OWT run. Tmux uses a project-local
socket, and the attach command is printed. No automatic instance-reboot hook is
installed. CPU smoke uses a **debug-size model**, not a full-model VRAM test.

#### First H100 trial: objective-matched MDM + neighbor prediction

Use the dedicated no-memory recipe first; the choice of merged versus separate
attention **for the later recurrent model is deferred**. This baseline already
works without the legacy memory gate/source-dropout mechanisms. It uses one
current-only attention per block, the existing 2D RoPE layout, five independent
teacher-forced states, and neighbor weight 0.5. No DCache or final feedback is
created or consumed. No OpenWebText data/checkpoint is needed.

On the cloud machine, with the repository synced and its Python environment
installed:

```bash
cd /workspace/dc-test
export DCACHE_PYTHON="$(command -v python)"

# Prepare once: 20,000 training / 1,000 validation / 1,000 test pilot examples.
bash scripts/train/train_reasoning_mdm_aux_h100.sh sudoku prepare

# Isolated debug-size GPU check, followed by the actual 5,000-update run.
bash scripts/train/train_reasoning_mdm_aux_h100.sh sudoku smoke &&
bash scripts/train/train_reasoning_mdm_aux_h100.sh sudoku tmux

# Refresh the figure; evaluate runs model-generated solving (100 test examples).
bash scripts/train/train_reasoning_mdm_aux_h100.sh sudoku plot
bash scripts/train/train_reasoning_mdm_aux_h100.sh sudoku evaluate
```

Replace `sudoku` with `zebra` or `countdown`. Run trials sequentially on the same
GPU. The default data suffix is `pilot-v1-n20000-v1000-t1000`, shared across
future variants; the baseline run is
`outputs/reasoning/sudoku/mdm_aux-h100-pilot-v1-n20000-v1000-t1000-seed1/`.
Existing smaller demo datasets are not reused accidentally. Custom paths/counts
are supported through the `REASONING_*` variables printed by `--help`; retain
those overrides for plotting, evaluation and resuming. The actual training
dataset hashes/settings are recorded in `contract.json` and `launch.json`.
Validation during training uses a fixed 128-example subset of the prepared
validation split at all four mask ratios, every 500 updates; the remaining
examples are available for later larger evaluations. Auxiliary heads are not
used to generate tokens. Resume by rerunning the same `tmux` command after the
old process exits. This does not install an automatic machine-reboot hook.

The larger dataset is still **synthetic pilot data**, not the author's benchmark.
If the later model uses two attention sublayers, this one-attention baseline
will not replace the extra-current-attention/parameter-matched controls.

Variants: `vanilla` (one-state, ordinary 1D RoPE), `mdm` / `mdm_aux` (matched
five-state, current-only 2D RoPE), `final`, `dcache`, `both` / `both_aux`.
Main causal comparison: `mdm_aux` versus `both_aux`, then `mdm` versus `both`.
All memory controls are trained models, not merely inference-time cache removal.
Memory-specific robustness losses remain a separate ingredient; use
`REASONING_NO_ROBUSTNESS=1` in a **new run directory** to ablate them.

Training CSVs are under `outputs/reasoning/<task>/<variant>-<profile>-seed1/logs/`.
Validation NLL averages fixed 10/30/50/70% answer corruptions **without warmup
memory**. Content-only NLL is also recorded; full solving requires the separate
generation command. This is conditional masked-token CE, not a diffusion ELBO
or generative perplexity. See the log for nested-memory evaluation and
multi-run plotting commands.

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
