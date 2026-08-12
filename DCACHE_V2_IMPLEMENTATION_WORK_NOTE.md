# DCache-v2 implementation work note

**Date:** 2026-08-12

**Status:** implemented and verified
**Supersedes:** the three-forward pretraining objective in
`FINAL_DENOISING_CACHE_PLAN.md`, `PROJECT_SERVER_HANDOFF.md`, and the older
runbook sections. The shifted layer mapping, separate DCache attention, and 2D
RoPE described in those documents remain in use.

## 1. What changed and why

The first 5k DCache trial learned to use the presence of cache entries, but it
did not learn much dependence on the identity of the cached document:
correct-cache versus no-cache mattered, while correct-cache versus shuffled
cache was almost unchanged. DCache-v2 attacks that failure in three ways:

1. train on four nearby, exact nested denoising states rather than one
   potentially very large `s -> t` jump;
2. sometimes remove one attention source before softmax so masked queries must
   learn both cache-dependent and current-state-dependent behavior;
3. explicitly require the correct cache to beat a batch-shuffled cache.

A learnable residual gate is also added so each layer can control the initial
strength of its DCache branch.

## 2. Exact five-forward training trajectory

For each sequence independently:

```text
k ~ Uniform(0.025, 0.10)
x ~ Uniform(1.5k, 0.9975 - 1.5k)

t0 = x + 1.5k
t1 = x + 0.5k
t2 = x - 0.5k
t3 = x - 1.5k
```

Thus `0 <= t3 < t2 < t1 < t0 <= 0.9975`, and every adjacent sampled mask-ratio
gap is the same sampled `k`. The actual sequence is:

```text
Forward 0: full mask -> write M_full
Forward 1: state t0 reads M_full -> write M_t0
Forward 2: state t1 reads M_t0   -> write M_t1
Forward 3: state t2 reads M_t1   -> write M_t2
Forward 4: state t3 reads M_t2
```

The first/upstream BOS position stays visible. All other valid positions are
randomly permuted once, and the four states use exact prefix counts of that
same permutation. Therefore their masks are strictly nested. Integer boundary
corrections guarantee:

- `t0` is not a second full-mask state;
- every transition reveals at least one token;
- `t3` still contains at least one mask to predict;
- every revealed token is the ground-truth token (100% teacher forcing).

The code uses the realized exact-count ratio, not the unrounded sampled ratio,
when constructing the diffusion loss for a state. Full-sequence training needs
at least five maskable positions; the real length-1024 data easily satisfies
this.

## 3. Loss

The five base loss weights are:

```text
full : t0 : t1 : t2 : t3 = 0.05 : 0.10 : 0.20 : 1.00 : 0.70
```

The base objective is normalized by their sum, `2.05`:

```text
L_base = (0.05 L_full + 0.10 L_t0 + 0.20 L_t1
          + 1.00 L_t2 + 0.70 L_t3) / 2.05
```

The identity term is added separately:

```text
L_total = L_base + 0.10 L_identity
```

Cross-forward gradients remain truncated with
`step_memory.detach_between_steps=true`. Cache projection/writer parameters
still receive learning signal, without retaining the entire previous backbone
graph.

## 4. Query-level source dropout

Source modes are sampled only for still-masked queries in the `t2` and `t3`
forwards. One sampled `[batch, sequence]` mode tensor is shared by all layers
in that forward and is applied before the DCache softmax:

```text
mode 0, joint:        query attends [previous cache, current state]
mode 1, cache-only:   query attends [previous cache]
mode 2, current-only: query attends [current state]
```

Current-only probability is fixed at 5%. Cache-only probability ramps linearly
from 0% to 20% over optimizer steps 0–1000. Consequently, after warmup the
distribution is 75% joint, 20% cache-only, and 5% current-only. Visible queries
remain joint. Source dropout is disabled automatically during validation and
inference.

This is implemented as a boolean SDPA mask over the concatenated previous and
current K/V. Previous and current keys still use one DCache softmax for joint
queries.

## 5. Correct-cache identity loss

On 25% of training batches, the `t3` state is evaluated once more with `M_t2`
rolled across examples in the local GPU batch. It uses the same source mask as
the correct-cache `t3` forward. The auxiliary objective is raw masked-token
NLL, without the MDLM time weight:

```text
L_identity = mean(relu(
  0.05 + NLL(correct M_t2) - stopgrad(NLL(shuffled M_t2))))
```

Current-only queries are excluded because their prediction cannot depend on
the cache. The shuffled forward runs without gradients; gradients act through
the correct-cache prediction. Examples with no eligible query after exclusion
are safely skipped.

The shuffle is local to each DDP worker. Therefore **per-GPU microbatch must be
at least 2** for identity training. The supplied launchers default to
microbatch 2. If microbatch 1 is forced, the five-forward objective still
works, but `identity_applied` stays zero.

## 6. Per-layer residual gate

Every DCache attention sublayer now computes:

```text
hidden += tanh(g_layer) * dcache_attention_output
```

There is one scalar `g_layer` per Transformer layer. It is initialized so its
effective value is exactly `tanh(g_layer) = 0.1`. It is trainable and included
in EMA. The ordinary BD3 attention and MLP residuals are unchanged.

The gate is optional in the base configuration. DCache-v2 launchers enable it;
leaving it disabled preserves the parameter structure needed to load old v1
checkpoints strictly.

## 7. Architecture retained from the shifted DCache implementation

The per-layer order is still:

```text
DCache attention -> DCache residual -> normal attention -> normal MLP
```

Reader layer `l` consumes the preceding forward's shifted `M_(l+1)`. Layers
2–12 naturally write `M2...M12` after completing the preceding layer, and the
single explicit final writer produces `M13` from the last-layer output. The
DCache branch has separate norm, QKV, output projection, and softmax from
normal attention. It uses parameter-free 2D RoPE: spatial position distinguishes
tokens and temporal position distinguishes previous from current K/V.

## 8. Files changed

- `models/dit.py`: per-layer gate; source-mask validation and pre-softmax SDPA
  masking; plumbing through all DIT blocks.
- `diffusion.py`: exact local trajectory, five-forward loss, source-dropout
  schedule, identity loss, diagnostics, and validation component logging.
- `configs/config.yaml`: all DCache-v2 knobs, disabled by default for v1
  compatibility.
- `scripts/train/train_owt_dcache_pretrain_100k.sh`: authoritative DCache-v2
  overrides.
- `scripts/train/train_owt_dcache_pretrain_5k_2x4090.sh`: safe two-GPU 5k
  wrapper and a new output directory.
- `scripts/profile_dcache_pretrain.py`: profiles the full DCache-v2 path,
  including the identity branch.
- `scripts/plot_pretrain_losses.py`: uses `trainer/loss_t2` for v2 and falls
  back to old `trainer/loss_s` logs for v1.
- controlled evaluation scripts: accept `--dcache-gate-enabled`; the shell
  wrapper exposes it as `DCACHE_EVAL_GATE_ENABLED=1`.
- tests: exact/nested trajectory, five-forward gradient flow, source masking,
  gate initialization, validation metrics, and v1/v2 evaluation composition.

## 9. Verification completed on this server

Environment:

```text
conda environment: dcache
PyTorch CUDA training: BF16 mixed precision
GPU profile device: NVIDIA GeForce RTX 3090, 24 GB
model: BD3-small/MDLM, length 1024, SDPA
parameters: 199,128,414
```

Results:

```text
python -m pytest -q
23 passed, including strict gated-checkpoint and EMA reload

microbatch 1, one complete update:
  peak allocated 8.658 GiB
  peak reserved  8.730 GiB
  3.341 seconds

microbatch 2, one complete update with identity probability forced to 1:
  peak allocated 15.420 GiB
  peak reserved  15.924 GiB
  3.657 seconds
```

Both CUDA updates completed without SDPA/source-mask errors. Microbatch 2 is
the recommended 24 GB setting. These short synthetic timings are health-check
measurements, not an end-to-end throughput estimate with real loading, DDP,
validation, and checkpoint I/O.

## 10. Exact 5k launch on the two-4090 server

After pulling the implementation:

```bash
cd /path/to/dc-test
git pull
source /home/jhua0805/miniconda3/etc/profile.d/conda.sh
conda activate dcache

CUDA_VISIBLE_DEVICES=0,1 \
DCACHE_DATA_DIR=/share/home/jhua0805/nick_exp/dc-test/.cache/huggingface \
bash scripts/train/train_owt_dcache_pretrain_5k_2x4090.sh
```

Defaults are 5,000 optimizer steps, global batch 512, per-GPU microbatch 2,
validation every 500 optimizer steps over 100 validation batches, BF16, and a
last checkpoint at:

```text
outputs/owt-dcache-v2-pretrain-5k-2x4090/checkpoints/last.ckpt
```

Logs are below:

```text
outputs/owt-dcache-v2-pretrain-5k-2x4090/run/lightning_logs/version_*/metrics.csv
```

The new output name deliberately prevents automatic resume from the completed
v1 run. If the v2 run itself is interrupted, rerunning the identical command
resumes from its `last.ckpt`. Resume is numerically the normal Lightning resume
path, including optimizer, scheduler, EMA, and global step, although exact
bitwise identity can still depend on data-loader worker state and CUDA kernels.

### Current server: four RTX 3090s

The preferred local allocation is now two RTX 3090s, physical devices 2 and 3:

```bash
cd /home/tliu0205/dc-test
source /home/tliu0205/miniconda3/etc/profile.d/conda.sh
conda activate dcache

bash scripts/train/train_owt_dcache_v2_pretrain_5k_2x3090.sh
```

The wrapper internally sets `CUDA_VISIBLE_DEVICES=2,3`, so physical GPUs 0 and
1 remain untouched. To select a different pair later, set
`DCACHE_CUDA_VISIBLE_DEVICES`, for example
`DCACHE_CUDA_VISIBLE_DEVICES=0,1`. The output is:

```text
outputs/owt-dcache-v2-pretrain-5k-2x3090/
```

Per-GPU microbatch remains 2 so shuffled-cache identity training is active.
With two GPUs and global batch 512, Lightning uses 128 gradient-accumulation
batches per optimizer update.

The wrapper also verifies that PyTorch sees exactly two devices before loading
the dataset. This prevents the opaque dataloader assertion caused when
`trainer.devices=2` but all four physical GPUs remain visible. A real-data
two-rank optimizer update on physical GPUs 2 and 3 passed after this check was
added.

The four-GPU wrapper remains available if the full server is used later.

This checkout already contains both prepared OpenWebText caches. The dedicated
local wrapper uses all four 24 GB GPUs while preserving global batch 512 and
per-GPU microbatch 2:

```bash
cd /home/tliu0205/dc-test
source /home/tliu0205/miniconda3/etc/profile.d/conda.sh
conda activate dcache

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train/train_owt_dcache_v2_pretrain_5k_4x3090.sh
```

Its checkpoint and CSV log are written below:

```text
outputs/owt-dcache-v2-pretrain-5k-4x3090/checkpoints/last.ckpt
outputs/owt-dcache-v2-pretrain-5k-4x3090/run/lightning_logs/version_*/metrics.csv
```

Do not reduce `DCACHE_MICRO_BATCH` below 2 unless the shuffled-cache identity
loss is intentionally being disabled. With four GPUs and microbatch 2, global
batch 512 produces 64 gradient-accumulation batches per optimizer update.

This wrapper was verified on 2026-08-12 with the real cached OpenWebText data:
four NCCL ranks initialized, each rank used local batch 2 at length 1024, one
complete DCache-v2 optimizer update and one validation batch finished, and both
`last.ckpt` and the monitored checkpoint appeared in the expected output tree.
The random-initialization health-run `val/nll` was 10.8249; it is not a quality
measurement. All four GPUs were released after the test.

## 11. Plot during or after training

```bash
python scripts/plot_pretrain_losses.py \
  --vanilla outputs/owt-mdlm-pretrain-5k-2x3090 \
  --dcache outputs/owt-dcache-v2-pretrain-5k-2x4090 \
  --output outputs/dcache-v2-vs-baseline-5k.png
```

Useful raw columns include:

```text
trainer/loss                    total optimized objective
trainer/loss_base               normalized five-state base objective
trainer/loss_full
trainer/loss_t0 ... loss_t3
trainer/mask_ratio_t0 ... mask_ratio_t3
trainer/source_cache_only_probability
trainer/source_cache_only_fraction
trainer/source_current_only_fraction
trainer/identity_applied
trainer/identity_loss
trainer/identity_correct_nll
trainer/identity_shuffled_nll
trainer/identity_gain           shuffled NLL - correct NLL; larger is better
trainer/gate_mean
val/loss_full
val/loss_t0 ... val/loss_t3
val/loss_total
val/nll                         t2 MDLM-weighted validation accumulator
val/gate_mean
```

The training/validation curves are excellent health diagnostics, but the
baseline's ordinary uniformly sampled `val/nll` and DCache-v2's local-trajectory
`val/nll` are not exactly the same state distribution. Do not use that plot
alone as the paper comparison. Use the controlled, identical-mask evaluation
for the scientific conclusion.

## 12. Controlled evaluation of a DCache-v2 checkpoint

For a gated v2 checkpoint, the evaluation architecture flag is mandatory:

```bash
BASELINE_CKPT=/path/to/baseline/checkpoints/last.ckpt \
DCACHE_CKPT=/path/to/owt-dcache-v2-pretrain-5k-2x4090/checkpoints/last.ckpt \
DCACHE_DATA_DIR=/path/to/.cache/huggingface \
DCACHE_EVAL_GATE_ENABLED=1 \
DCACHE_FIXED_CUDA=0 \
DCACHE_TRANSITION_CUDA=1 \
bash scripts/eval/run_5k_teacher_forced_evals.sh parallel
```

Keep `DCACHE_EVAL_GATE_ENABLED=0` or omit it for the old ungated v1
checkpoint. Strict checkpoint loading is intentional: it prevents silently
evaluating a checkpoint with the wrong architecture.

## 13. What to inspect before a longer run

At steps 500, 1000, and 5000, check:

1. `source_cache_only_probability` ramps to 0.20 and the sampled fractions
   roughly track the configured probabilities;
2. `identity_applied` averages near 0.25, not zero;
3. `identity_gain` becomes clearly positive and grows beyond the v1 near-zero
   correct-versus-shuffled difference;
4. `gate_mean` remains finite and does not collapse or explode toward ±1;
5. every component loss is finite, and `loss_t2`/`loss_t3` improve;
6. controlled correct-cache versus shuffled/no-cache results improve without
   sacrificing too much baseline conditional NLL.

The first decision point should be the controlled 5k evaluation. A longer
100k run is justified only if cache identity sensitivity improves. Attention
mass alone is insufficient evidence: the v1 model already assigned substantial
mass to the cache without using its document-specific content.

## 14. Deliberate limitations

- This is still teacher-forced pretraining; exposure to model-sampled reveals
  belongs to the later rollout stage.
- Identity shuffling is within each GPU's local batch, not globally across all
  DDP workers.
- The identity loss adds one no-gradient forward on 25% of batches, so DCache-v2
  is not FLOP-matched to vanilla MDLM.
- No conclusion about quality has been made from the one-step health profiles.
  The 5k controlled evaluation is still required.
