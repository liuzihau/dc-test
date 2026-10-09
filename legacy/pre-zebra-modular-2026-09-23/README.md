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

### Zebra clue-learning diagnostic (September 21)

The repaired answer-only baseline finished 5,000 updates: no PAD collapse,
but 0/1,000 complete test solves. A paired validation audit finds nearly
unchanged predictions after shuffling clue references; full-mask cell accuracy
is close to a clue-blind baseline. See the
[analysis and paired follow-up](RESEARCH_EXPERIMENT_LOG.md#zebra-hard-start-pair-2026-09-21).

Two full-state continuations use the same frozen checkpoint and data order:
GPU 2 keeps ordinary corruption; GPU 3 forces 50% of examples to have fully
masked answers. Both stop at update 20,013 (three total clean-data epochs).

```bash
# Monitor the already launched jobs; Ctrl-b 0/1 switches trials.
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/tmux \
  -S /share2/home/tliu0205/dc-test/.cache/runtime/reasoning-tfw-hardstart/tmux.sock \
  attach -t zebra-hardstart-pair

# Refresh the validation / clue-sensitivity figure (no GPU needed).
DCACHE_PYTHON=/home/tliu0205/miniconda3/envs/dcache/bin/python \
  bash scripts/reasoning/run_zebra_hardstart_pair.sh report
```

The launcher also has `plan`, `smoke`, `control`, and `hardstart` actions.
`control`/`hardstart` resume their own outputs after interruption; do not launch
a second copy while they are active. Runtime files and the tmux socket are
under the project, not `/tmp`. The comparison figure is
[`status.png`](results/generated/figures/reasoning/zebra-tfw-hardstart/status.png).

### Overnight Zebra encoding tests (September 22)

The full-mask intervention did not materially improve clue-based prediction by
step18,000. Two new **fresh** diagnostics are queued behind those runs:
GPU2 fixes the answer position IDs; GPU3 adds shared public house/attribute/value
features to the same fixed positions. These are representation tests, not an
exact paper reproduction or a new DCache result. Details and safeguards are in
[the overnight log](RESEARCH_EXPERIMENT_LOG.md#zebra-encoding-night-2026-09-22).

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/tmux \
  -S /share2/home/tliu0205/dc-test/.cache/runtime/reasoning-zebra-encoding/tmux.sock \
  attach -t zebra-encoding-night
```

Windows0/1 correspond to GPUs2/3. Queue progress is recorded in
`.cache/runtime/reasoning-zebra-encoding/gpu{2,3}/queue_status.json`;
console logs are `logs/zebra-encoding-answer-relative-gpu2.log` and
`logs/zebra-encoding-typed-coordinates-gpu3.log`. Do not launch duplicate copies.
Each queue waits for current evaluation and GPU release, then runs a real GPU
smoke/resume test and TRAIN-only memorization check before a fresh six-epoch
run. A failed check stops that queue rather than silently starting training.

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

Completed reasoning evaluations can be compared without loading a model or using
a GPU:

```bash
python scripts/reasoning/summarize_reasoning.py \
  --runs outputs/reasoning/sudoku/*h100*mb128-gb128-seed1 \
         outputs/reasoning/zebra/*h100*mb128-gb128-seed1 \
  --output-dir outputs/reasoning/reports/overnight-2026-09-16 --step 5000
```

The report includes whole-puzzle accuracy, Wilson 95% intervals, paired-puzzle
bootstrap differences, CSVs, PNGs and `summary.md`. It rejects mismatched
datasets, steps, decoding protocols and uncontrolled training-contract changes;
unfinished trials are explicitly pending, not treated as failures or zeros.
Intervals describe held-out puzzle uncertainty for one trained seed, not
training-seed variability. A higher masked-token validation accuracy alone is
not evidence of better complete-puzzle solving.

For the authorized 2026-09-16 Vast run, a separate supervisor service
`dcache_reasoning_overnight` now refreshes this report every five minutes while
waiting for the existing training queue. It never terminates that queue. If the
predecessor disappears, it verifies completion or waits for an idle GPU before
resuming the same four memory runs from their own checkpoints. Finished runs
are not trained for extra steps. It starts with supervisor after a container
restart; a failed recovery attempt is left stopped visibly rather than retried
in an uncontrolled OOM loop. No SSH, Jupyter or management services were changed.

On that cloud instance:

```bash
supervisorctl status dcache_reasoning_overnight
cat outputs/reasoning/queues/sudoku-zebra-memory-current-preserving-h100-pilot-v1-n20000-v1000-t1000-mb128-gb128-seed1/overnight_status.json
cat outputs/reasoning/reports/overnight-2026-09-16/summary.md
```

Training/validation plots remain in the queue's `figures/` directory; accuracy,
paired differences and descriptive Zebra clue diagnostics are in the report
directory. The existing tmux session can still be attached with:

```bash
tmux -S /workspace/dc-test/.cache/tmux/reasoning-queue.sock attach -t reasoning-memory-gpu0-5a7efe95
```

The reusable foreground coordinator is `scripts/reasoning/run_overnight_queue.py`.
Its `--wait-pid` and `--wait-pid-starttime` identify the exact predecessor, not
just an occasionally idle GPU. Do not launch another copy while the supervisor
service is active. Experiment caches/temp files stay under the repository's
`.cache`; container recycle/destruction still requires an off-instance backup.

This four-run overnight suite completed successfully at 2026-09-15 20:19 UTC;
the supervisor's `EXITED` state is normal completion. Final local results are in
`imports/reasoning-cloud-20260916/final/overnight-2026-09-16/`, with training and
validation plots in `imports/reasoning-cloud-20260916/final/figures/`.

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

## Zebra follow-up (separate from completed 5k results)

The bounded continuation diagnoses the synthetic Zebra floor without changing
its training recipe or overwriting original checkpoints:

```bash
python scripts/reasoning/run_zebra_continuation.py plan
# Pilot-only launch in its own queue directory (do not duplicate the cloud job):
python scripts/reasoning/run_zebra_continuation.py run --gpu 0 --hours 12 \
  --queue-dir outputs/reasoning/queues/zebra-pilot-only-10k
```

This compares all five variants at 10k updates, with train/validation clue-use
audits at 5k and 10k. It is **not** a paper reproduction. GPU jobs are sequential;
temporary/cache files remain under this repo. Queue status, logs and figures:
`outputs/reasoning/queues/zebra-continuation-5k-10k-v1/`.
On the authorized H100, `dcache_zebra_continuation` is the supervisor service.
Its deployed configuration also imports the released Zebra data and trains a
separate vanilla reference before the pilot continuations. Inspect without
starting a second process:

```bash
supervisorctl status dcache_zebra_continuation
cat outputs/reasoning/queues/zebra-continuation-5k-10k-v1/status.json
```

Data import is CPU-only, so an idle GPU during that stage is expected. The
service configuration is [here](scripts/reasoning/zebra_continuation.supervisor.conf);
it does not stop the cloud instance or its billing. The source-data-aligned
experiment is separate; see the latest research log.

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

## Corrected reasoning benchmark (2026-09-16)

Use the **versioned released-data reconstruction**, not the earlier synthetic
Sudoku/Zebra pilot scores, for the next comparisons. The paper is
[Reasoning with Latent Tokens in Diffusion Language Models](https://arxiv.org/abs/2602.03769v1).
`reasoning/benchmark.py` imports released Sudoku puzzles and migrates the
selected released Zebra puzzles without changing their IDs or split order.
All five methods share immutable 20k train / 1k validation / 1k test subsets.

```bash
python -m reasoning.benchmark zebra \
  --source-dir .cache/reasoning/zebra-official-v1-n20000-v1000-t1000 \
  --output .cache/reasoning/zebra-benchmark-v2-n20000-v1000-t1000
python -m reasoning.benchmark sudoku \
  --train-file imports/official-reasoning/raw/sudoku-train.npy.partial \
  --test-file imports/official-reasoning/raw/sudoku-test.npy \
  --output .cache/reasoning/sudoku-benchmark-v2-n20000-v1000-t1000
bash scripts/reasoning/run_benchmark_aligned.sh plan
# Fresh benchmark training; subsequent launches resume only these new runs.
DCACHE_PYTHON=/venv/main/bin/python \
  bash scripts/reasoning/run_benchmark_aligned.sh run --gpu 0 --hours 10
```

Do not start a second copy while `dcache_benchmark_v2` is running in supervisor.
Status: `outputs/reasoning/benchmark-v2/status.json`; per-stage console logs:
`console/`; accuracy figure/table: `report/`; training figures:
`{zebra,sudoku}-benchmark/training.png`. A persistent time budget includes
downtime. It stops on failure/deadline, and never shuts down the cloud instance.
The supplied cloud service additionally caps runtime at the earlier queue's
deadline. Old pilot checkpoints cannot resume into the new tokenizer/layout.

Generation uses fully masked answers, one sampled token per forward, top-prob
position selection with 8 candidates, and the same test IDs/seed for every model.
Primary accuracy is complete-grid correctness; EOS formatting and strict
grid-plus-format success are separate. There is no oracle repair or ground-truth
token insertion during evaluation. Padding is excluded from attention and loss.

**Not an exact author-verified reproduction.** The paper does not fully specify
token aliases, padding treatment, Zebra flatten order, or format scoring.
Our explicit conventions (Sudoku blank=MASK, Zebra no-BOS clues/SEP/grid/EOS)
produce the reported 14/23 vocabulary sizes and 192/384 sequence limits, but
matching those counts does not prove identical tokenization. The assumptions
are stored in every new manifest and evaluation. The smaller training subsets,
5k updates and final-checkpoint selection are also declared differences. Wait
for author confirmation before claiming exact replication or paper superiority.

## Full-data reasoning: one epoch, five mechanisms (2026-09-16)

The next queue runs **Zebra and Sudoku-Puzzle**, each with MDM, TT, TT + NP,
TT + RM, TT + RM + NP. TT = five-state trajectory training; NP = masked-target
previous/next prediction; RM = layerwise DCache + detached final-state feedback.
RM keeps the corrected current-preserving merged attention and adjacent-gradient
setting. This is ten fresh runs, not continuation of the 20k-subset checkpoints.

Every unique released training puzzle is used once, excluding the frozen
validation puzzles and **all official test puzzle identities**. Validation/test
JSONL files remain byte-identical to benchmark-v2. Packed memory-mapped training
files retain original source row indices and checksums. Actual retained counts
and `ceil(count / 128)` updates appear in the queue's `plan.json`; the final
partial batch is neither dropped nor filled by repeating examples.

```bash
# Inspect the ten jobs (no training).
DCACHE_PYTHON=/venv/main/bin/python bash scripts/reasoning/run_full_epoch.sh plan
# Standalone launch ONLY when no other job owns this GPU:
DCACHE_PYTHON=/venv/main/bin/python bash scripts/reasoning/run_full_epoch.sh run --gpu 0
# Refresh training/validation figures and accuracy bars, including partial results:
DCACHE_PYTHON=/venv/main/bin/python bash scripts/reasoning/run_full_epoch.sh report
```

On the current cloud instance, `dcache_full_epoch` is the intended supervisor
service. Its `--after-subset outputs/reasoning/benchmark-v2` handoff waits for the
current Zebra both_aux **final checkpoint AND complete 1k-test evaluation**,
then retires only `dcache_benchmark_v2` and skips its pending subset Sudoku jobs.
CPU data preparation may overlap the old GPU job. Do not run a duplicate queue.
The old service's autostart must be false before arming this handoff; apply its
updated supervisor configuration only after it finishes (the queue does this).

Global/microbatch128, seed1, mini architecture, AdamW/LR/warmup unchanged.
Validation every500 updates and at the end uses the same frozen1k examples:
`val/conditional_nll` remains **cold**; `val/nested_conditional_nll` additionally
carries memory through teacher-forced ratios0.7→0.5→0.3→0.1. Both are conditional
NLL, not an ELBO or solve accuracy. Final generation uses the unchanged paired1k
test, top-prob candidate8, one sampled token/forward, no teacher forcing/repair.
Keep only the latest3 checkpoints, saved every500 updates or20 minutes.

Outputs: `outputs/reasoning/full-epoch-v1/`; inspect `status.json`, `plan.json`,
`handoff.json`, per-run `logs/`, `validation/`, and `generation.json`.
Figures: each task's `training-cold.png` and `training-nested.png`, plus
`report/accuracy.png` and CSV. Runtime caches stay under `.cache/runtime/`.
After a failure/reboot, restarting the same service resumes verified full-state
checkpoints; a failure is recorded in `failure.json` and is not retried endlessly.
The queue stops after ten runs; it does not power off the instance.

One epoch matches **examples**, not computation: TT evaluates five states per
example. This is still a shorter, paper-informed reconstruction, not an exact
author-verified reproduction or the paper's multi-epoch training budget.

## Second full epoch: Zebra and Sudoku, all five methods (2026-09-17)

The bounded `dcache_second_epoch` cloud service waits for **all ten** jobs in
`full-epoch-v1` (including Sudoku generation evaluations), then continues MDM,
TT, TT + NP, TT + RM, and TT + RM + NP: Zebra first, Sudoku second.
It copies each verified final checkpoint to `outputs/reasoning/second-epoch-v1/`
and retains the original epoch-one results. Model, optimizer, RNG, batch128,
learning rate and loss settings are unchanged. Only the epoch budget becomes2;
each epoch uses a fresh deterministic permutation and its own partial tail.
Zebra continues6671→13342 updates; Sudoku14090→28180. This is **one extra
epoch**, not a new two-epoch run. Validation remains every500 updates and at
completion; keep the latest3 continuation checkpoints.

```bash
# Cloud monitoring: the waiting queue does not reserve GPU memory.
supervisorctl status dcache_full_epoch dcache_second_epoch
cat /workspace/dc-test/outputs/reasoning/second-epoch-v1/status.json

# Preview or regenerate completed comparisons; neither starts training.
DCACHE_PYTHON=/venv/main/bin/python bash scripts/reasoning/run_second_epoch.sh plan
DCACHE_PYTHON=/venv/main/bin/python bash scripts/reasoning/run_second_epoch.sh report
```

Per-task comparisons are written automatically after generation to
`outputs/reasoning/second-epoch-v1/report/{zebra,sudoku}-benchmark/`:
`generation_comparison.png`, `generation_comparison.csv`, `paired_changes.csv`,
and `comparison.json`. They compare the same frozen1000 test examples, seed2026,
candidate8 top-prob policy, one sampled token per forward, batch32, and correct
memory. They include solved counts, whole-puzzle/cell accuracy, and paired
gained/lost puzzles; one training seed does not establish statistical robustness.
Runtime/plot/compiler caches stay under the project `.cache/runtime/` directory.
The queue exits after all ten continuations/evaluations; it does not stop the
cloud instance or alter billing settings. See the experiment log for deployment
status and tests.

## Split-attention reasoning suite: six mechanisms (2026-09-18)

The new suite uses released **Zebra and Sudoku-Puzzle**, one full epoch each,
with the existing frozen 1,000-example validation and test sets. Twelve jobs
run sequentially, with two GPU workers cooperating on each training job.

| Display name | New suite ID | Attention sublayers | Recurrence | NP |
|---|---|---|---|---|
| MDM | `mdm` | 1 | none | off |
| TT | `tt` | 1 | none | off |
| TT + EA | `tt_ea` | 2 | none; extra attention uses current context | off |
| TT + EA + NP | `tt_ea_np` | 2 | none | on |
| TT + EA + RM | `tt_ea_rm` | 2 | DCache + detached final state | off |
| TT + EA + RM + NP | `tt_ea_rm_np` | 2 | DCache + detached final state | on |

EA uses the original separate normalization, QKV, output projection, residual
and gate; the ordinary attention and MLP follow. RM restores split-attention
2D RoPE (sequence position plus previous=0/current=1), shifted cache writer
mapping, one-transition adjacent gradients, and detached final feedback.
TT uses five teacher-forced states and normalized weights
`(0.05, 0.10, 0.20, 1.00, 0.70) / 2.05`. NP retains two independent heads and
`0.5 * mean(prev CE, next CE)`, with only masked neighbor targets supervised;
source positions may be clean. Padding and special boundaries remain excluded.

The full RM recipe restores cache-only/current-only source probabilities
0.20/0.05, source warmup 1,000, final dropout 0.10, and identity probability
0.25 (margin 0.05, weight 0.10, final-source selection 0.50). These have no
meaning in non-memory controls and are disabled there. Add `--no-robustness`
and a NEW `--output` directory for a clean comparison without those auxiliaries.
RM also adds a terminal KV writer and final-state normalization, so EA matches
the extra attention capacity but is not exactly total-parameter/FLOP matched.
Parameter counts are recorded in each run's `launch.json`.

```bash
conda activate dcache
# Preview all twelve commands (does not allocate a GPU).
bash scripts/reasoning/run_split_full_epoch_2x3090.sh plan
# When GPUs 2,3 are free and the full prepared data is present:
bash scripts/reasoning/run_split_full_epoch_2x3090.sh smoke && \
  bash scripts/reasoning/run_split_full_epoch_2x3090.sh run
# Refresh available accuracy and cold/nested validation figures:
bash scripts/reasoning/run_split_full_epoch_2x3090.sh report
```

Defaults: GPUs `2,3`, microbatch **8 per GPU**, global batch **128**, accumulation
**8**, BF16, validation every **500 optimizer updates**, eval batch **8**.
Override `DCACHE_MICRO_BATCH=4` before starting if the memory stress smoke fails;
the global batch stays 128 and accumulation becomes 16. Resume requires the
original batch geometry. Smaller local batches change identity-shuffle groups
and random corruption draws; they do not promise bitwise equivalence to H100.

The smoke command runs one update for every task/variant, including forced
identity/reference forwards for RM, and small cold/nested validation. Its
checkpoints are isolated under a timestamped smoke directory and never used by
the production queue. Run uses fresh split-suite directories and resumes only
its own compatible checkpoints. Latest three checkpoints are retained, with
saves every 500 updates or 20 minutes at optimizer boundaries. Exact epoch
tails are split without dropping or repeating supervised examples; empty ranks
participate using a zero-weight dummy to complete distributed synchronization.

Prepared data defaults to `.cache/reasoning/{zebra,sudoku}-benchmark-full-v1`.
Pass `--data-root /path/to/reasoning` if stored elsewhere. Preparation can reuse
the previous raw sources and frozen subsets; no automatic source download is
performed. Results live in `outputs/reasoning/full-epoch-split-six-2x3090/`.
Application temporary/compiler files stay under the project `.cache/`.

New IDs are activated by `--suite split`, which the launcher always supplies.
Historical `vanilla`, `mdm`, `mdm_aux`, `both`, and `both_aux` checkpoints retain
their original interpretation under the legacy suite; the old merged results
are not renamed into split runs. New plots use the six names above.

## Second epoch: six split controls on GPUs 2,3 (2026-09-19)

The local continuation launcher waits for **all twelve** first-epoch training
and generation jobs in `full-epoch-split-six-2x3090-mb32`. It then runs **Sudoku
first, Zebra second**, all six controls per task, for exactly **one additional
epoch**. It does not interrupt the predecessor or start a concurrent GPU job.

```bash
conda activate dcache
# Preview only; does not use GPUs or copy checkpoints.
bash scripts/reasoning/run_split_second_epoch_2x3090.sh plan
# Start the waiting queue in a project-local tmux server; safe while epoch one runs.
# Do not launch a second copy if this session already exists.
bash scripts/reasoning/run_split_second_epoch_2x3090.sh tmux
# Reconnect to the queue (stage announcements; child log path printed at each stage).
bash scripts/reasoning/run_split_second_epoch_2x3090.sh attach
cat outputs/reasoning/second-epoch-split-six-2x3090-mb32/status.json
# Refresh available epoch-one vs epoch-two accuracy/cell-accuracy comparisons.
bash scripts/reasoning/run_split_second_epoch_2x3090.sh report
```

Settings remain **microbatch32/rank, accumulation2, global128, BF16, GPU2/3**.
Adam state, both ranks' RNG states, data cursor, seed, loss/robustness settings,
and learning rate are retained. The LR is **constant 0.0003 after the original
1,000-update warmup**; neither warmup nor optimizer is restarted. Sudoku
continues **14090 → 28180**; Zebra **6671 → 13342**. The second epoch uses the
next deterministic shuffled data permutation, not the first-epoch order.

Epoch-one results remain unchanged. New checkpoints/logs/validation/evaluations
live in `outputs/reasoning/second-epoch-split-six-2x3090-mb32/`. The source and
destination can be overridden with `DCACHE_FIRST_EPOCH_DIR` and
`DCACHE_SECOND_EPOCH_DIR`; incompatible geometry/recipes are rejected. Validation
remains every500 updates plus final, with unchanged cold/nested ratios; final
generation uses the same frozen1k test, seed2026, candidate8, one sampled token
per forward and batch32. Paired epoch comparisons and plots are in
`report/{sudoku,zebra}-benchmark/`. Latest three continuation checkpoints are
retained, saved every500 updates or20min at optimizer boundaries.

The tmux socket and runtime/compiler/plot caches are under
`.cache/runtime/reasoning-second-split/`, not `/tmp`. Queue announcements are
also in `logs/split-second-epoch-mb32.log`; `status.json` supplies the active
child `console` path for detailed step logs. While waiting, the queue creates
no CUDA context. After a reboot/failure, rerun the same launcher to resume
verified full-state checkpoints; tmux alone does not survive a reboot. The
queue stops after epoch two and its evaluations; it does not start epoch three.

## Third epoch: six split controls on GPUs 2,3 (2026-09-21)

**Revised at the user's request:** finish the active Sudoku `tt_ea_np` training
and evaluation, then **Zebra only**, in this order:
`tt_ea_rm_np → tt_ea_rm → tt_ea_np → tt_ea → tt → mdm`.
The remaining Sudoku RM/RM+NP trials are skipped; already completed Sudoku
results stay intact. `requested_schedule.json` records this selection without
changing any training contract. The old scheduler is paused (not its workers)
until the current training exits; the replacement then takes ownership. Check
`handoff_status.json` while it waits. No settings or LR
restart: microbatch32/rank ×2 GPUs ×accumulation2 = global128, BF16, LR0.0003.
Full optimizer/RNG/data state is preserved; only the epoch budget changes2→3.
The next shuffled epoch consumes every training example exactly once, including
the partial tail. Sudoku advances28180→42270 and Zebra13342→20013 updates.

```bash
conda activate dcache
bash scripts/reasoning/run_split_third_epoch_2x3090.sh plan
# Only start if it is not already queued; tmux action rejects duplicate sessions.
bash scripts/reasoning/run_split_third_epoch_2x3090.sh tmux
bash scripts/reasoning/run_split_third_epoch_2x3090.sh attach
cat outputs/reasoning/third-epoch-split-six-2x3090-mb32/status.json
cat outputs/reasoning/third-epoch-split-six-2x3090-mb32/handoff_status.json
tail -f logs/third-epoch-zebra-handoff.log
# Refresh paired epoch-two vs epoch-three generation plots/tables when available.
bash scripts/reasoning/run_split_third_epoch_2x3090.sh report
```

Outputs: `outputs/reasoning/third-epoch-split-six-2x3090-mb32/`; epoch-one/two
artifacts stay unchanged. Validation remains every500 optimizer updates plus
final, generation uses the same frozen1000 test puzzles and decoding protocol.
Latest3 new-run checkpoints are kept; saves every500 updates or20min. Reports
are written automatically after each final evaluation under
`report/{sudoku,zebra}-benchmark/`. The queue stops after epoch three; no fourth
epoch or instance shutdown is scheduled.

The revised session is `reasoning-epoch3-zebra` (the launcher detects the
schedule file); the old `reasoning-epoch3` controller retires after handoff.
Its socket and all explicitly redirected
temporary/compiler/plot caches are in `.cache/runtime/reasoning-third-split/`,
not `/tmp`. The waiting queue creates no CUDA context. A stopped predecessor
blocks continuation for inspection rather than silently skipping its jobs.
Override source/output only with `DCACHE_SECOND_EPOCH_DIR` /
`DCACHE_THIRD_EPOCH_DIR`. After reboot/failure, rerun the same `tmux` command;
verified checkpoints resume, but tmux itself does not survive a reboot.

## Zebra MDM: independent-mask sampler ablation (2026-09-21)

An **opt-in, fresh plain-MDM trial** changes only corruption: sample
`j ~ Uniform{1,...,64}` per example and independently mask each eligible answer
token with probability `j/64`. Clues and outer padding remain protected. This
matches the referenced `diffusion-vs-ar` corruption rule, **not the entire paper
recipe**; 64 is that repository's default, not a verified Zebra training setting.
All existing rounded-count MDM and five-forward TT runs/queues stay unchanged.

Run in the `dcache` environment **only when GPUs 2,3 are available**:

```bash
bash scripts/reasoning/run_zebra_mdm_bernoulli_2x3090.sh plan
bash scripts/reasoning/run_zebra_mdm_bernoulli_2x3090.sh smoke &&
bash scripts/reasoning/run_zebra_mdm_bernoulli_2x3090.sh run
```

Default: full clean Zebra training set, **3 epochs / 20,013 updates**, global
batch 128, microbatch 32 per GPU, LR .0003, original warmup and fixed validation
every 500 updates. Final generation uses the same frozen 1,000 test puzzles and
candidate-8 sampled decoding. Outputs are isolated in
`outputs/reasoning/zebra-mdm-bernoulli-t64-3ep-2x3090-mb32/`.
Rerunning `run` resumes only this trial's own compatible checkpoint; it does
not switch the sampler of an existing MDM run. Runtime files stay in the repo.
`DCACHE_CORRUPTION_TIMESTEPS`, `DCACHE_EPOCHS`, `DCACHE_DATA_DIR`,
`DCACHE_RUN_DIR`, `DCACHE_MICRO_BATCH`, `DCACHE_EVAL_BATCH`, `DCACHE_PYTHON`,
`DCACHE_DEVICES` and `DCACHE_CUDA_VISIBLE_DEVICES` can be set explicitly.

Direct runner flags: `--suite split --variant mdm --corruption-mode bernoulli
--corruption-timesteps 64`. Zero-mask draws are retained with zero loss (not
resampled); their frequency and the full-mask frequency are logged. Train loss
still averages over **all examples**, including these zeros; use the unchanged
validation protocol/generation for performance comparisons, not a raw train-loss
decrease alone. See the [experiment log](RESEARCH_EXPERIMENT_LOG.md#zebra-mdm-bernoulli-corruption-ablation).

### Paper-informed Zebra GPT-2 control (GPU 2)

**Historical/failed padding control:** the completed shifted and unshifted
runs collapsed toward PAD predictions. Do not extend this recipe unchanged.
Use the answer-only repair below for a new diagnostic run. The commands in
this subsection remain for reproducibility, not as the recommended baseline.

Separate from all historical DiT/TT/RM runs, this control follows the default
model/loss conventions of the code cited by *Train for the Worst, Plan for the
Best*. It is **not an exact reproduction**: author-specific Zebra preprocessing
and overrides are unverified; we retain our leakage-filtered data and held-out IDs.
It uses a19.2M bidirectional GPT-2, LR0.001, global batch128, shifted logits,
inverse-timestep token loss, and visible/denoised padded output slots.

```bash
# In the dcache environment; do not start alongside an already-running copy.
bash scripts/reasoning/run_zebra_tfw_gpu2.sh plan
bash scripts/reasoning/run_zebra_tfw_gpu2.sh smoke &&
bash scripts/reasoning/run_zebra_tfw_gpu2.sh run
```

Default budget is a3epoch probe; the linear LR schedule retains a300epoch
horizon. After the job exits, `DCACHE_TFW_PROBE_EPOCHS=20` extends the same
full-state run to20epochs without changing that horizon. GPU3 is not touched.
Validation every500steps uses the common fixed answer-only NLL. Epoch-end
generation reports compare upstream remasking, paper-informed monotonic
decoding, and our existing candidate8 sampler separately. Do **not** compare
its weighted training objective numerically to the old unweighted train loss.
Full details and remaining reproduction gaps are in `RESEARCH_EXPERIMENT_LOG.md`.

The matched **no-shift GPT-2** control runs independently on GPU3:

```bash
bash scripts/reasoning/run_zebra_tfw_no_shift_gpu3.sh smoke &&
bash scripts/reasoning/run_zebra_tfw_no_shift_gpu3.sh run
```

It starts from the same seed with the same recipe, data order and3epoch budget,
but hidden state at position`i` predicts token`i` directly. Alignment is applied
consistently in training, validation and all generation decoders. Outputs go to
`outputs/reasoning/zebra-tfw-gpt2-no-shift-19m-gpu3/`. Checkpoints enforce the
alignment setting; do not initialize this ablation from the shifted checkpoint.
Old GPT-2 checkpoints without a model-level alignment key retain shifted behavior.

### Zebra GPT-2 answer-only padding repair

The repair keeps the same19.2M model, T64 Bernoulli corruption and
inverse-timestep loss, but restricts corruption/supervision/generation to the
public solution grid plus EOS, and excludes outer padding keys from attention.
It is a separate same-position model, not a resume of collapsed weights.
The launcher uses LR0.0003 after a separate small-subset LR stability check;
this is explicitly not an unchanged-LR reproduction of the paper.
No vocabulary constraints or solver repairs hide invalid generated tokens.

```bash
export DCACHE_PYTHON=/home/tliu0205/miniconda3/envs/dcache/bin/python
# Defaults to physical GPU3. Ensure it is free before starting.
bash scripts/reasoning/run_zebra_tfw_answer_only.sh plan
bash scripts/reasoning/run_zebra_tfw_answer_only.sh smoke
bash scripts/reasoning/run_zebra_tfw_answer_only.sh overfit
bash scripts/reasoning/run_zebra_tfw_answer_only.sh run
```

`overfit` tests32 randomly selected TRAIN puzzles for3000 updates. It is a
memorization diagnostic, not held-out accuracy. `run` refuses to proceed unless
that diagnostic generates at least95% of its full grids correctly with no PAD
in answer cells at each of its last three evaluations. The full-data probe starts from scratch, stops at5000 updates,
validates every500, probes128 validation generations every1000, and evaluates
the same1000 frozen test puzzles at the end with all three labelled decoders.
The300epoch LR horizon is unchanged; this is not a300epoch job. Training
checkpoints keep the latest three and resume only with the same strict contract.

Reports live in `outputs/reasoning/zebra-tfw-answer-only-no-shift-lr3e4-gpu3/`.
Common validation NLL stays comparable to older runs; `full_mask_diagnostic`
separately measures100%-masked answers. Training logs now explicitly show
`train/padding_supervision_fraction` and `train/answer_predicted_pad_fraction`.
Temporary files remain inside project `.cache/runtime`, not `/tmp`.
See `RESEARCH_EXPERIMENT_LOG.md` for evidence and remaining paper-reproduction gaps.

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
