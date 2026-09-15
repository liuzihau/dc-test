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

The general launcher above preserves the **legacy** memory recipe for old
checkpoints. For new corrected merged-attention memory trials, use the dedicated
H100 memory queue below; it explicitly selects `current_preserving`.

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

#### Sequential H100 queue: microbatch 128, no memory models

The queue runs **Sudoku and Zebra**, each with `vanilla`, `mdm`, and `mdm_aux`,
one job at a time. It excludes every recurrent model and Countdown. Microbatch
128 and global batch 128 mean **one batch per optimizer update**. All six runs
use 5,000 updates and new `mb128-gb128-seed1`-labelled directories; the previous
microbatch-8 experiment is preserved and is **not** silently resumed at a new
batch size. Changing grouping changes corruption RNG and auxiliary averaging;
equal global batch does not imply exact replay.

After syncing the code to the cloud and finishing/stopping the old GPU job:

```bash
cd /workspace/dc-test
export DCACHE_PYTHON="$(command -v python)"

# Prints the six jobs and paths without running them.
bash scripts/train/train_reasoning_baselines_h100.sh plan --micro-batch 128 --global-batch 128

# One tmux session handles preparation, preflight, all training, plots and evaluation.
bash scripts/train/train_reasoning_baselines_h100.sh tmux --micro-batch 128 --global-batch 128

# The launcher prints the exact project-local tmux attach command.
bash scripts/train/train_reasoning_baselines_h100.sh status --micro-batch 128 --global-batch 128
```

Use `run` instead of `tmux` inside a session you already manage. The `smoke`
action runs **only the full-model preflight**: for each task, `mini` (6 layers,
width 512), BF16, microbatch 128, two complete AdamW updates and validation with
the heaviest `mdm_aux` variant. This is different from the old debug-size smoke.
It prints update time and peak PyTorch allocated VRAM. Production `run` performs
this preflight automatically. An OOM or occupied GPU stops the queue; it never
silently lowers the microbatch, alters the model, kills another job, or proceeds
past a failed stage. A low instantaneous `nvidia-smi` reading alone is not proof
that the full five-forward batch fits.

Existing seed-17 20k/1k/1k pilot splits are reused after count/metadata/checksum
verification. Missing splits are generated without overwriting a partial or
different dataset. Every training run validates every 500 updates and retains
the existing latest-three/20-minute checkpoint policy. After each run, the queue
updates that task's comparison plot and generates solutions on **1,000 test
examples** from the final/latest checkpoint (not validation-best), with the same
candidate-8 top-prob policy and evaluation batch 8 across variants. Auxiliary
heads do not generate tokens. Plots show `train/base_loss` separately from shared
fixed-corruption validation NLL, not the auxiliary-inflated total training loss.
Even base training CE uses a different corruption distribution for `vanilla`
versus the five-state methods; use common validation/solving for their comparison.

Queue status, console logs, smoke measurements and `figures/sudoku.png` /
`figures/zebra.png` are under
`outputs/reasoning/queues/sudoku-zebra-baselines-h100-pilot-v1-n20000-v1000-t1000-mb128-gb128-seed1/`.
On an interruption, rerun the **same** command: the trainer verifies each run's
own checkpoint/contract, completed training is skipped, and matching completed
generation results are reused. Smoke checks rerun. No automatic instance-reboot
hook is installed; tmux itself does not survive machine shutdown. Do not change
batch geometry or repurpose an old run directory during resume.

#### H100 memory queue: corrected merged `both` and `both_aux`

This opt-in queue runs **Sudoku `both` → Sudoku `both_aux` → Zebra `both` →
Zebra `both_aux`**, sequentially. The existing six-run no-memory queue is
unchanged. Each new run uses the same 20k/1k/1k seed-17 pilot splits, model size
`mini` (6 layers, width 512, 8 heads), model/data-order seed 1, BF16,
microbatch 128/global batch 128 and 5,000 optimizer updates. It needs **no OpenWebText
dataset or checkpoint**.

| Variant | Five-state objective | Recurrent memory | Neighbor heads |
| --- | --- | --- | --- |
| `mdm` (existing control) | Yes | None | Off |
| `mdm_aux` (existing control) | Yes | None | On, weight 0.5 |
| `both` | Yes | DCache + detached final state | Off |
| `both_aux` | Yes | DCache + detached final state | On, weight 0.5 |

Both new variants use **one merged attention**, current+previous KV, joint
softmax and 2D RoPE (sequence position plus previous/current iteration identity,
not continuous noise level). The corrected policy keeps current KV available:
previous-V gate **off**, cache-only dropout **0%**, current-only **5%** of eligible
masked queries at the last two training states. Final-state dropout is **10%**
per trajectory. Identity-reference probability is **25%**, weight 0.1/margin 0.05,
with a 50:50 DCache/final shuffle choice when final feedback is available.
DCache gradients cross each of the four adjacent boundaries but **never two
boundaries from one state's loss**; final-state feedback always stays detached.
The five base losses retain weights `(0.05, 0.10, 0.20, 1.00, 0.70) / 2.05`.
`both_aux` adds 0.5 times the trajectory-weighted mean of prev/next CE, only
where the **target neighbor** is masked and both endpoints are valid content.
The source may be revealed. No gold auxiliary target is injected into a forward.

Sync the updated repository to the H100 first. **Finish or stop your existing
GPU queue before starting this one**; neither launcher kills other processes.

```bash
cd /workspace/dc-test
export DCACHE_PYTHON="$(command -v python)"

# Inspect the four jobs without launching GPU work.
bash scripts/train/train_reasoning_memory_h100.sh plan --micro-batch 128 --global-batch 128

# A single tmux session: verify/prepare data, preflight, train, plot and evaluate.
bash scripts/train/train_reasoning_memory_h100.sh tmux --micro-batch 128 --global-batch 128

# Prints the current stage and status; the launcher also prints its attach command.
bash scripts/train/train_reasoning_memory_h100.sh status --micro-batch 128 --global-batch 128
```

Use `run` instead of `tmux` inside an existing tmux session. Use `smoke` to run
only the preflight. Production launch automatically performs the full-model
preflight, so a separate `smoke && tmux` is unnecessary. Preflight uses
`both_aux` for **both full task lengths**, microbatch 128, two AdamW updates,
auxiliary heads and adjacent backward. It **forces an identity-reference forward
on every update**, retains final feedback and tests current-only dropout. Peak
PyTorch VRAM and time/update are recorded in `full_model_smoke.json`. Smoke
contracts/directories are distinct from production; the forced probabilities
are never used for the real trials. GPU fit is not established by CPU tests:
OOM stops the queue, without silently reducing the batch.

Every 500 updates, validation uses the same fixed 128 examples and independent
10/30/50/70% answer masks as the controls: **cold conditional NLL, without
previous memory**, not an ELBO or generative perplexity. After each run, all 1,000
test examples are decoded using **both memories** and the same candidate-8
top-prob policy, evaluation batch 8/seed 2026 and final/latest checkpoint as the
controls. Neighbor heads do not generate tokens. Training figures exclude
auxiliary/identity loss from the `train/base_loss` panel; the validation panel
explicitly says cold NLL. Compatible existing control curves are included after
full training-contract and validation-protocol checks; missing/incompatible
controls are not modified, retrained or silently included.

Runs are under `outputs/reasoning/<task>/`, with prefixes
`both-h100-current-preserving-` and `both_aux-h100-current-preserving-` followed
by the dataset/batch/seed labels.
Queue status, console logs, smoke reports and `figures/sudoku.png` /
`figures/zebra.png` are under
`outputs/reasoning/queues/sudoku-zebra-memory-current-preserving-h100-pilot-v1-n20000-v1000-t1000-mb128-gb128-seed1/`.
Latest-three full checkpoint retention, 500-update and 20-minute saving, strict
full-state resume and project-local `.cache`/tmux sockets are shared with the
baseline queue. Rerun the same command after an interruption; no automatic
machine-reboot hook is installed. Old legacy reasoning, batch-8 and OWT runs
are never imported into these new trials.

Interpretation: `both_aux` versus `both` isolates the added auxiliary objective
within this memory recipe. `both` versus `mdm`, or `both_aux` versus `mdm_aux`,
tests the **complete memory-training package** (including robustness/identity
loss and additional computation), not solely architecture or matched FLOPs.
The task data is still synthetic pilot data, not the authors' benchmark splits.

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
