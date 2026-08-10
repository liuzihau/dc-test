# Shifted Recurrent Denoising Cache: 100k Pretraining Plan

**Decision date:** 2026-08-10
**Status:** authoritative architecture and first pretraining comparison

This document supersedes conflicting choices in `PROJECT_SERVER_HANDOFF.md`,
`bd3_denoising_kv_research_plan.md`, and the former block-16/50k plan. Those
files remain historical context.

## 1. Experiment goal

Pretrain two BD3-small foundations from scratch on the upstream OpenWebText
pipeline for 100,000 optimizer updates:

1. vanilla MDLM, matching the pretraining stage used before BD3 fine-tuning;
2. the proposed shifted recurrent denoising-cache model.

Both use GPT-2 tokenization, sequence length 1024, effective global batch 512,
the original log-linear MDLM objective, the same optimizer/schedule, and the
same seed. The first comparison is matched by optimizer update and training
tokens, not by FLOPs: the recurrent model performs three forwards per example
and contains a second attention sublayer.

## 2. Two independent cache concepts

- **Completed-prefix cache:** vanilla BD3 inference reuse for clean completed
  blocks. It persists across blocks.
- **Denoising cache:** learned memory from the immediately preceding real
  denoiser forward. It is replaced after every real forward and reset for a new
  block. Reusing already-computed logits does not advance it.

The 1024-token MDLM pretraining stage has no external prompt. The first token
is kept visible under the upstream `ignore_bos` convention; all other valid
positions can be masked.

## 3. Final per-layer computation order

Every non-causal Transformer layer executes:

```text
input hidden H_(l-1)
  -> DCache pre-norm
  -> dedicated DCache QKV
  -> DCache attention over [previous shifted K/V, current K/V]
  -> dedicated DCache output projection
  -> residual add
  -> normal BD3 pre-norm/QKV/attention/output/residual
  -> normal MLP pre-norm/MLP/residual
  -> completed layer output H_l
```

Normal attention and DCache attention have separate norms, QKV weights, output
weights, and softmaxes. Inside DCache attention, previous and current K/V share
one softmax. There is no tanh gate in this first from-scratch version.

## 4. Shifted or diagonal cache mapping

Define the entry projection:

```text
M_l = DCache-QKV_l(H_(l-1))
```

On the next denoising forward, reader layer `l` consumes the preceding
forward's `M_(l+1)`:

```text
layer 1  reads previous M2
layer 2  reads previous M3
...
layer 11 reads previous M12
layer 12 reads previous M13
```

`M2...M12` require no standalone writer. They are produced automatically by
the entry DCache projections of layers 2...12 after the previous forward has
completed the preceding layer:

```text
M2  = DCache-QKV_2(H1)
M3  = DCache-QKV_3(H2)
...
M12 = DCache-QKV_12(H11)
```

There is no layer 13, so one explicit final writer produces raw K/V:

```text
M13 = Final-DCache-KV-Writer(H12)
```

The model API consequently returns exactly `[M2, M3, ..., M13]`. Entry `i`
is read by layer `i+1` on the next denoising forward.

## 5. DCache attention and 2D RoPE

For current reader layer `l`:

```text
Q_current, K_current, V_current = DCache-QKV_l(current H_(l-1))
K_previous, V_previous = previous M_(l+1)

output = Attention(
  Q_current,
  concat[K_previous, K_current],
  concat[V_previous, V_current])
```

Raw unrotated K/V is stored. For 64-dimensional heads:

- 48 dimensions encode spatial token position;
- 16 dimensions encode temporal role;
- previous K uses temporal coordinate 0;
- current Q/K uses temporal coordinate 1;
- V is not rotated.

RoPE is parameter-free. Previous and current occurrences of the same spatial
position are distinguishable through the temporal coordinate.

## 6. Three-pass pretraining example

Every recurrent training example uses nested, fully teacher-forced states:

```text
Forward 0: 100% maskable positions masked
Forward 1: s mask ratio
Forward 2: t mask ratio, with 0 < t < s
```

### Forward 0: full-mask bootstrap

- Keep the first/upstream BOS position visible.
- Mask every other valid position.
- No previous DCache is supplied.
- Compute `L_full` and write `cache_full = [M2...M13]`.

### Forward 1: ordinary MDLM state s

- Sample `s` with the upstream antithetic uniform-time sampler over
  approximately `[0.001, 1)`.
- Mask positions with probability `s`; all visible tokens are clean teacher
  tokens.
- Read `cache_full`, compute the ordinary MDLM loss `L_s`, and write `cache_s`.
- `L_s` is logged separately as the closest comparison with vanilla MDLM.

### Forward 2: recurrent state t

- Sample `t` uniformly from `[0.001, s]`.
- Starting only from positions masked at `s`, retain a mask with probability
  `t/s`; every newly revealed position receives its clean teacher token.
- Read `cache_s` and compute `L_t` on positions still masked.
- Do not remask a token visible at `s`.

Boundary corrections guarantee for every sequence:

```text
masks at s >= 2
masks at t >= 1
newly revealed positions >= 1
mask_t is a strict subset of mask_s
```

The conditional `t` sampler intentionally emphasizes late, low-mask denoising
states. `L_t` is therefore an additional recurrent objective, not a second
unbiased copy of the uniform-time MDLM objective.

## 7. Loss and gradients

Relative loss weights are:

```text
full : s : t = 0.1 : 1.0 : 1.0
```

The optimized loss is normalized to preserve its scale:

```text
L_total = (0.1 * L_full + L_s + L_t) / 2.1
```

Logged values include `loss_full`, `loss_s`, `loss_t`, `loss_total`, sampled
mask ratios, newly revealed token count, and remaining mask count.

With `detach_between_steps=true`, gradients stop at the preceding backbone
hidden state, but the saved cache is re-projected from that detached hidden
state. Therefore the lightweight implicit writers and the explicit `M13`
writer still receive gradients from the next denoising loss. This is truncated
cross-forward credit assignment, not a completely detached writer.

## 8. Inference recurrence

For each active generation block:

```text
previous_cache = None
while masks remain:
    logits, current_cache = model(state, previous_cache)
    reveal token(s) using logits
    previous_cache = current_cache
```

Only the immediately previous real forward is retained. DCache resets at a
new block; the completed-prefix cache keeps its original lifetime.

## 9. First 100k comparison

Launchers:

```text
scripts/train/train_owt_mdlm_pretrain_100k.sh
scripts/train/train_owt_dcache_pretrain_100k.sh
scripts/train/run_owt_pretrain_comparison_100k.sh
```

Both runs use `algo=mdlm`, `block_size=1024`, `model.length=1024`, global batch
512, and `training.from_pretrained=null`. SDPA is used because the current
environment does not include FlashAttention.

Measured model sizes are 169,627,218 parameters for vanilla and 199,128,402
for shifted DCache. Four-GPU health runs with microbatch 4 and 32-way gradient
accumulation took about 15 and 33 seconds per optimizer update respectively;
100k updates are consequently multi-week jobs on the four RTX 3090s.

The comparison must report:

- raw vanilla training loss versus recurrent `loss_s`;
- recurrent `loss_full`, `loss_t`, and normalized total loss;
- validation NELBO under the ordinary one-state MDLM evaluator;
- optimizer updates, sequences/tokens, real forwards, wall time, parameters,
  and peak VRAM.

Later evidence must include cache-disabled and cache-shuffled evaluation plus
a matched-FLOP comparison before claiming an efficiency or quality gain.

## 10. Closest prior work and remaining risks

`dKV-Cache` already caches selected decoded-token K/V across diffusion steps
for training-free inference acceleration. It replaces recomputation for stable
positions; it does not add a separately trained full previous-state attention
branch or the shifted `l -> l+1` mapping. Its one-step-delay result supports
computing a clean-token representation once before trusting it as cache.

The closest architectural precedent is Feedback Transformer, which exposes
higher-level past representations to current lower layers through a learned
shared feedback memory. Our hard one-layer shift remains experimental.

Primary diagnostics after the health run are previous/current attention mass,
Q/K norm by source, DCache residual norm, `M13` gradient norm, cache-disabled
loss, and all-position versus revealed-only cache ablations.

## 11. Deferred work

- model-sampled revealed tokens (begin later with 85% teacher / 15% sampled);
- learned/tanh gates or zero-impact pretrained-checkpoint adaptation;
- separate previous/current softmaxes inside DCache attention;
- learned layer mixtures like Feedback Transformer;
- caching only decoded positions as in dKV-Cache;
- more than one preceding denoising state;
- Hugging Face DIT support;
- block-16 fine-tuning after the pretraining comparison.
