# Dcachehooping Implementation Note

**Run status (2026-08-24 Australia/Sydney):** the full tentative/confidence
variant remains available for later experiments, but it is no longer the
recommended first trial. The current trial is the minimal DCache + detached
final-state architecture described below. A real bf16 two-GPU DDP smoke run
completed three optimizer/EMA updates at sequence length 1024 and per-GPU
microbatch 2 without OOM or unused-parameter errors. The full repository test
suite passes (36 tests). Physical GPUs 2 and 3 were released after the test.

## Recommended core trial: DCache + final-state reuse

The core trial retains the DCache-v2 five-state objective and adds only the
detached final-layer representation from the preceding denoising state:

```text
input = token_embedding + LN(previous_final_hidden)
```

The previous hidden and previous DCache are detached between states. The
final-state LayerNorm scale is initialized to zero, so the new path starts as
a neutral extension of DCache-v2. Ten percent final-state dropout trains a
DCache-only fallback. The existing DCache query-source dropout and the 25%
identity intervention remain enabled; identity batches contaminate DCache or
the final state with equal probability.

The following components are structurally absent, not merely multiplied by a
zero loss weight:

- tentative status embeddings;
- tentative-token correction;
- confidence head and confidence loss;
- latent-mask auxiliary forward;
- editable full-vocabulary log-probability tensors.

Therefore, the comparable base objective is exactly:

```text
(0.05 L_full + 0.10 L_t0 + 0.20 L_t1 + 1.00 L_t2 + 0.70 L_t3) / 2.05
```

plus the same `0.10 L_identity` convention used by DCache-v2. Compare
`trainer/loss_base` and `val/loss_t2`; do not compare raw total losses across
methods with different auxiliary terms.

Launch the 5,000-step core trial on physical GPUs 2 and 3:

```bash
conda activate dcache
cd /share2/home/tliu0205/dc-test
mkdir -p logs
tmux new -s dcache-final
bash scripts/train/train_owt_dcache_final_state_5k_2x3090.sh \
  2>&1 | tee logs/dcache-final-state-5k.log
```

Detach from tmux with `Ctrl-b`, then `d`; reattach with:

```bash
tmux attach -t dcache-final
```

The output directory is:

```text
outputs/owt-dcache-final-state-pretrain-5k-2x3090/
```

## Scope

Dcachehooping is an opt-in extension of DCache-v2. The original
`_step_memory_pretrain_loss` path, its five teacher-forced corruptions, source
dropout, validation aliases, and checkpoint format remain available when
`dcachehooping.enabled=false`.

The extension adds three signals:

1. the detached final-layer hidden from the preceding forward;
2. explicit mask/committed/tentative status embeddings;
3. direct correction and confidence supervision for model-proposed tokens.

It does not add a memory-reliability gate. The existing DCache-v2 residual
gate remains enabled by the launcher so the comparison only changes the new
components.

## Recurrent state

Every regular state uses the existing DCache-v2 cache and, unless final-state
dropout is selected, the preceding detached final hidden:

```text
input = token_embedding + status_embedding + LN(previous_final_hidden)
```

The status IDs are `MASK=0`, `COMMITTED=1`, and `TENTATIVE=2`. Both the status
embedding and the final-hidden LayerNorm scale are initialized to zero. Thus a
DCache-v2 checkpoint loaded with `strict=false` initially preserves its old
logits. The confidence head is also zero-initialized and starts at probability
0.5.

## Base and auxiliary losses

The comparable masked-token base is unchanged:

```text
(0.05 L_full + 0.10 L_t0 + 0.20 L_t1 + 1.00 L_t2 + 0.70 L_t3) / 2.05
```

On 25% of training batches, detached `t2` predictions replace the tokens newly
revealed between `t2` and `t3`. Those positions are marked tentative and a
complete auxiliary `t3` forward learns direct token correction. This uses an
unclamped editable log-probability view because ordinary MDLM substitution
parameterization intentionally copy-clamps visible tokens.

The tentative confidence target is:

- correct committed or tentative token: 1;
- incorrect tentative token: 0;
- mask: detached probability assigned to its clean target.

On 10% of batches, an auxiliary same-state robustness pass replaces eligible
lexical embeddings with `[MASK]`, retains the logical `t2` status, consumes the
correct detached `t2` final hidden, and receives no previous DCache. Its loss
mask remains the original `t2` mask. Prompt/BOS and padding positions are not
synthetically masked.

Tentative correction and latent-mask robustness use one categorical draw:
25% tentative, 10% latent-mask, and 65% neither. Their intended marginal
frequencies are unchanged, while no batch retains both extra trainable
transformer graphs. Final-hidden dropout and the no-gradient identity probe
remain independent because neither adds a second auxiliary gradient graph.

The configured total is:

```text
L_total = L_base
        + 0.10 L_latent_mask
        + 0.10 L_tentative
        + 0.30 L_confidence
        + 0.10 L_identity
```

Auxiliary losses are present only on their selected batches, matching the
existing DCache-v2 identity-loss convention. `trainer/loss_base` and
`val/loss_t2` remain the primary comparable curves; total objectives from
different methods should not be compared directly.

## Identity contamination

The existing identity batch probability is 25%. Each selected batch executes
one complete no-gradient contaminated forward:

- 50% shuffle DCache and keep final hidden correct;
- 50% shuffle final hidden and keep DCache correct.

The original margin objective remains:

```text
relu(0.05 + NLL_correct - stopgrad(NLL_contaminated))
```

## Metrics

Existing DCache-v2 fields are preserved. New CSV fields include:

- `trainer/loss_base`, `trainer/latent_mask_loss`;
- `trainer/tentative_loss`, `trainer/confidence_loss`;
- tentative count, wrong count, before/after accuracy;
- wrong-fix and correct-keep rates;
- confidence Brier score and correct/wrong confidence means;
- latent dropout/mask route indicators;
- identity source, NLLs, and gain.

Validation always runs the tentative correction probe and reports the same
fields under `val/`, while `val/loss_t2` remains the comparable teacher-forced
masked NLL.

## Training on GPUs 2 and 3

From scratch:

```bash
conda activate dcache
bash scripts/train/train_owt_dcachehooping_5k_2x3090.sh
```

The launcher exposes only physical GPUs 2 and 3, which Lightning sees as
logical devices 0 and 1. It saves every 500 steps and writes metrics below:

```text
outputs/owt-dcachehooping-pretrain-5k-2x3090-exclusive/
```

The effective global batch remains 512 with per-GPU microbatch 2 and 128-way
gradient accumulation. Failed pre-fix logs remain separately in
`outputs/owt-dcachehooping-pretrain-5k-2x3090/` and must not be merged with the
clean run.

Warm-start pilot from a DCache-v2 Lightning checkpoint:

```bash
DCACHE_RUN_DIR="$PWD/outputs/owt-dcachehooping-from-v2-5k" \
  bash scripts/train/train_owt_dcachehooping_5k_2x3090.sh \
  training.from_pretrained=/absolute/path/to/dcache-v2.ckpt
```

This is adaptation, not an exact optimizer resume: old model and EMA parameters
are migrated, while new parameters receive their neutral initialization. A
new Dcachehooping `last.ckpt` resumes exactly through the normal launcher.

## Verification completed

- Dcachehooping focused unit tests;
- unchanged DCache-v2 tests;
- shifted writer/source-dropout tests;
- objective-matched control tests;
- DCache-v2 checkpoint and EMA migration;
- one CUDA forward/backward through all new loss paths;
- two-GPU launcher preflight on physical GPUs 2 and 3;
- microbatch-1 worst-case overlapping auxiliaries across multiple optimizer
  updates with Adam and EMA state resident;
- microbatch-2 tentative-only and latent-mask-only training across multiple
  optimizer updates with Adam and EMA state resident.
