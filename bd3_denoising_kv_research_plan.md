# Recurrent Denoising-Space KV Memory for Block Diffusion Language Models

> **2026-08-08 implementation update:** Read `PROJECT_SERVER_HANDOFF.md` first.
> It records the current code, RTX 3090 feasibility, server procedure, and two
> decisions made after this plan was written: the first smoke test stores the
> ordinary active-block fused QKV rather than using a new post-layer writer,
> and the preferred short training rollout keeps continuous QKV attached so
> future denoising losses can train preceding forwards. Detached recurrence
> remains a memory-saving ablation.

## Purpose of this document

This is an implementation handoff for a research prototype built on [`kuleshov-group/bd3lms`](https://github.com/kuleshov-group/bd3lms). It records the motivation, architectural decisions, training design, curriculum, evaluation plan, known risks, and deferred ideas discussed so far.

The immediate project is **not** to add causal/AR Transformer layers. The first experiment is to add a recurrent, token-aligned KV memory across denoising evaluations while preserving the normal BD3-LM architecture and objective. Causal “motor” layers are a later experiment after the memory mechanism is understood.

## One-paragraph project summary

Vanilla masked/block diffusion language models discard their continuous hidden states after each denoising evaluation. We want each active block to retain the immediately preceding denoising evaluation as a per-layer KV memory. On the next evaluation, normal current-state self-attention and previous-step memory attention use separate softmaxes, while sharing the query and output projections. A zero-initialized per-head gate adds the memory result to normal attention, so the initial model is exactly the vanilla checkpoint. The new memory is written from post-memory, post-layer hidden states so it becomes genuinely recurrent. Training retains the original BD3 loss and adds an on-policy rollout loss for one selected block per sequence. Every rollout starts at the all-mask state—the only state with a well-defined null predecessor—and a curriculum gradually increases both reveal width and recurrent rollout length.

---

## 1. Research motivation

### 1.1 Working hypothesis

The motivating hypothesis comes from the “workspace” interpretation of intermediate Transformer states: early and middle layers may maintain reasoning, semantic, or planning information that is not identical to the next emitted token. Standard diffusion language models repeatedly compute such continuous representations, discretize some token decisions, and then discard the representations.

This creates an information bottleneck:

1. The model computes a rich continuous state at denoising step \(t\).
2. Sampling reduces that state to one or more discrete token decisions.
3. The next denoising evaluation receives only the new hard-token/mask sequence.
4. Uncommitted hypotheses, intermediate computations, and confidence structure are lost.

We want to retain a continuous **denoising workspace** across evaluations.

### 1.2 Core research question

Does access to the previous denoising evaluation's hidden KV state improve:

- likelihood/perplexity;
- generative perplexity at matched entropy;
- robustness to wrong committed tokens;
- sample consistency across a denoising trajectory;
- quality at a fixed number of model evaluations;
- reasoning or long-range coherence?

### 1.3 Why Block Diffusion is a useful test bed

BD3-LM already separates two forms of computation:

- autoregressive conditioning across completed blocks;
- bidirectional diffusion inside the active block.

It also has official small checkpoints, likelihood evaluation, generative evaluation, and an existing inter-block KV cache. The active block gives a bounded setting in which recurrent denoising memory can be implemented and studied.

---

## 2. Reference codebase and baseline facts

### 2.1 Repository

- Code: <https://github.com/kuleshov-group/bd3lms>
- Paper: [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573)
- The repository is based on the MDLM and SEDD codebases.

### 2.2 Official small architecture

From [`configs/model/small.yaml`](https://github.com/kuleshov-group/bd3lms/blob/main/configs/model/small.yaml):

| Setting | Value |
|---|---:|
| Transformer layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Head dimension | 64 |
| Conditioning dimension | 128 |
| Sequence length | 1,024 |
| Dropout | 0.1 |
| Tied token embeddings | Yes |
| Backbone type | `ddit` |

This is very close to the useful part of ELF-B's architecture: ELF-B also uses 12 layers, width 768, and 12 heads, although ELF uses a continuous embedding-flow objective, T5-small contextual embeddings, SwiGLU/RMSNorm/RoPE/qk-norm, and a different training pipeline.

### 2.3 Official BD3 block sizes

The released OpenWebText checkpoints use block sizes:

- 4;
- 8;
- 16.

They use the same model architecture and Python training program, but they are **separately trained checkpoints** because block size changes the attention mask, probabilistic factorization, and training examples. Do not evaluate a block-size-16 checkpoint by merely setting `block_size=4`.

For sequence length \(L=1024\):

| Block size \(B\) | Sequential blocks |
|---:|---:|
| 4 | 256 |
| 8 | 128 |
| 16 | 64 |
| 1,024 | 1 (full-sequence MDLM endpoint) |

### 2.4 Baseline training configuration

Important settings from the repository:

```yaml
algo:
  parameterization: subs
  T: 0                    # continuous-time training
  time_conditioning: false
  cross_attn: true
  var_min: true
  sampler: semi_ar

loader:
  global_batch_size: 512

optim:
  lr: 3e-4
  weight_decay: 0

training:
  ema: 0.9999
  antithetic_sampling: true
  resample: true

trainer:
  max_steps: 1_000_000
```

The official block models initialize from a block-size-1024 MDLM checkpoint trained on OpenWebText. The block-specific script changes `BLOCK_SIZE` and fine-tunes/trains the corresponding run.

### 2.5 Baseline inference and NFE interpretation

The example generation command uses:

```yaml
algo.T: 5000
sampling.first_hitting: true
sampling.kv_cache: true
sampling.nucleus_p: 0.9
```

With first-hitting, one currently masked position is chosen uniformly and one token is committed at each denoising update. For a 1,024-token unconditional sample with one fixed BOS token, the repository reports approximately 1,023 sampling NFEs regardless of block size.

With block size 4:

\[
3 + 255\times 4 = 1023
\]

reported denoising NFEs.

There is a measurement nuance: when `sampling.kv_cache=true`, the implementation performs an extra short forward after completing each block to populate its inter-block KV cache. Those cache-fill calls are not included in the repository's `sampling_steps` counter. For block size 4 and length 1,024, the raw number of `model.forward` calls is approximately (1023+256=1279), although the extra calls operate only on short completed blocks.

---

## 3. Terminology and time convention

### 3.1 Reverse-time convention

In the BD3 sampling code:

- \(t=1\) is maximally noisy/all mask;
- \(t=0\) is clean;
- reverse generation moves from larger to smaller \(t\).

For three consecutive reverse states:

\[
r>t>s.
\]

### 3.2 Two different caches

Keep these strictly separate in code and discussion:

| Cache | Contents | Lifetime | Existing? |
|---|---|---|---|
| `block_kv_cache` | KV from clean, completed earlier blocks | Across blocks | Yes |
| `step_kv_cache` | KV from the immediately preceding denoising evaluation of the active block | Within an active-block trajectory | New |

The existing `sampling.kv_cache` flag refers to inter-block caching. It should eventually be renamed or wrapped clearly so it is not confused with recurrent step memory.

### 3.3 Chosen memory semantics

**Decision:** retain only the immediately preceding denoising KV.

Do not retain the entire history:

```python
# Chosen
previous_step_kv

# Deferred/not chosen initially
all_previous_step_kvs
```

If the new KV is written from a hidden representation that already retrieved the prior KV, it is recurrent:

\[
M_t=f_\theta(x_t,M_r).
\]

It can indirectly carry older information without \(O(TLd)\) storage.

---

## 4. Selected attention architecture

### 4.1 Do not concatenate current and memory KV initially

A single-softmax design would concatenate current and previous keys:

```python
all_k = torch.cat([current_k, previous_k], dim=-2)
all_v = torch.cat([current_v, previous_v], dim=-2)
out = attention(q, all_k, all_v)
```

This is deferred because:

- current and previous states compete under one normalization;
- removing memory changes the normalization over current tokens;
- duplicated token coordinates may create confusing competition;
- memory can dominate unexpectedly;
- exact zero-impact initialization is harder;
- source-specific masking and temporal bias are harder to control.

### 4.2 Chosen design: shared query/output, separate softmaxes

Use one query projection and one output projection, but separate attention normalizations:

\[
A_{\mathrm{cur}}
=\operatorname{softmax}\left(\frac{QK_{\mathrm{cur}}^\top}{\sqrt{d_h}}+B_{\mathrm{cur}}\right),
\]

\[
A_{\mathrm{mem}}
=\operatorname{softmax}\left(\frac{QK_{\mathrm{prev}}^\top}{\sqrt{d_h}}+B_{\mathrm{time}}\right),
\]

\[
Y=W_O\left(A_{\mathrm{cur}}V_{\mathrm{cur}}+\tanh(g)\odot A_{\mathrm{mem}}V_{\mathrm{prev}}\right).
\]

Pseudocode:

```python
q, current_k, current_v = self.qkv(self.attn_norm(hidden))

current_out = scaled_dot_product_attention(
    q,
    current_k,
    current_v,
    attn_mask=current_block_mask,
)

if previous_step_kv is None:
    memory_out = torch.zeros_like(current_out)
else:
    previous_k, previous_v = previous_step_kv
    memory_out = scaled_dot_product_attention(
        q,
        previous_k,
        previous_v,
        attn_mask=None,
        # temporal bias can be added in a later milestone
    )

gate = torch.tanh(self.memory_gate).view(1, num_heads, 1, 1)
attn_out = current_out + gate * memory_out
hidden = hidden + self.out_proj(attn_out)
```

### 4.3 Memory gate

Use a zero-initialized per-head gate at every layer:

```python
self.memory_gate = nn.Parameter(torch.zeros(num_heads))
```

Reasons:

- the augmented model initially equals vanilla BD3;
- loading an official checkpoint is safe;
- heads can specialize differently;
- gates provide an interpretability diagnostic;
- memory can be ablated without changing current-attention normalization.

Log gate values by layer/head during training.

### 4.4 Where to write the next memory

Do **not** cache the ordinary self-attention K/V computed before memory injection. That cache would not contain the result of retrieving the previous memory.

Write the next memory from the post-memory, post-layer representation:

```python
hidden = hidden + self.out_proj(current_out + gate * memory_out)
hidden = hidden + self.mlp(self.mlp_norm(hidden))

memory_source = self.memory_write_norm(hidden)
next_memory_k = self.k_proj(memory_source)
next_memory_v = self.v_proj(memory_source)
```

Initially, reuse the normal K/V projection weights on the post-layer representation. Dedicated memory-writer projections are a later ablation.

### 4.5 Per-layer cache shape

A practical representation is:

```text
K, V: [batch, layers, kv_heads, active_block_length, head_dim]
```

or a Python list of one `(K, V)` tuple per layer to avoid a large stack/copy.

Memory should be restricted to the active block in the first implementation. Completed-block history remains in `block_kv_cache`.

---

## 5. Spatial position and denoising position

### 5.1 Spatial position

Unlike compressed-slot methods such as MetaState, raw KV is token aligned. Memory item \(i\) corresponds to token position \(i\).

Use the same sequence positions for current queries and previous-step keys. For Block Diffusion, use absolute positions within the full sequence rather than resetting every active block to positions `0..B-1` if the backbone's positional scheme expects absolute positions.

### 5.2 Temporal/denoising coordinate

If only the immediately previous state is retained, the separate memory branch already communicates “this KV is from the previous evaluation.” An explicit temporal embedding can be omitted from the first smoke test.

The preferred later temporal signal is denoising progress measured by remaining-mask ratio:

\[
\rho_t=\frac{\#\mathrm{MASK}(x_t)}{B}.
\]

Then add a relative temporal bias:

\[
B_{\mathrm{time}}=f_\theta(\rho_{\mathrm{query}}-\rho_{\mathrm{source}}).
\]

This is preferable to literal two-dimensional RoPE for the initial prototype. It also maps naturally to first-hitting and dense reveal schedules.

### 5.3 MetaState comparison

[MetaState](https://arxiv.org/abs/2603.01331) uses fixed global slots rather than token-aligned memory. It handles:

- slot identity with learned slot embeddings;
- spatial routing with position-aware token queries and cross-attention;
- denoising time with a shared time conditioner and AdaRMSNorm;
- recurrence with a GRU-style updater.

Our approach avoids slot-to-token routing and compression but has memory proportional to active block length.

---

## 6. Why every recurrent rollout starts at all mask

### 6.1 Bootstrap state

The fully masked state is the only state with a natural null predecessor:

\[
x^{(0)}=x_1=[\mathrm{MASK}]^B,
\qquad M^{(0)}=\varnothing.
\]

Every active block encounters this state during inference. Vanilla continuous-time BD3 training samples noise levels and therefore does not guarantee an exactly all-mask input.

### 6.2 Important limitation of the bootstrap memory

For an unconditional block with the same prefix, the all-mask input is identical across sample trajectories. Therefore \(M_1\) is not yet sample-specific. It can encode:

- information from completed blocks;
- positional structure;
- the model's prior over block completions;
- likely continuation structure.

It cannot know which random token samples will be committed. Memory becomes sample-specific only after sampled tokens are fed back through another forward pass.

Therefore, do not use \(M_1\) as a permanent substitute for local previous-step memory.

### 6.3 Local recurrence

The desired inference chain is:

```text
all-mask + null memory
        ↓ forward
      memory_0
        ↓ sample/reveal
state_1 + memory_0
        ↓ forward
      memory_1
        ↓ sample/reveal
state_2 + memory_1
        ↓ ...
```

A direct pair \(x_s+M_1\) with \(s\ll1\) is mismatched because inference would supply \(M_t\) from a nearby preceding state, not the stale bootstrap memory.

---

## 7. Transition sampler and wrong tokens

### 7.1 First-hitting transition

For one-token first-hitting:

1. Find currently masked positions.
2. Choose one uniformly.
3. Sample its token from the model's distribution, using the same top-p and temperature as inference.
4. Commit that sampled token.

```python
masked = x.eq(mask_id)
position = sample_uniform_position(masked)
probs = top_p_filter(logits[:, position].softmax(-1), top_p=0.9)
token = torch.multinomial(probs, 1)

x_next = x.clone()
x_next[:, position] = token
```

### 7.2 Dense transition

For reveal width \(R>1\), select \(R\) masked positions uniformly without replacement and sample their tokens from the model output. This must be matched or mixed with the inference schedule; otherwise training and inference see different transition widths.

### 7.3 Do not always reveal ground-truth tokens

Using:

```python
x_next[:, position] = x0[:, position]
```

is teacher forcing. It is useful during stabilization but creates exposure bias if used exclusively. The rollout should increasingly use model-sampled tokens.

Suggested transition-token curriculum:

| Phase | Ground-truth token | Model-sampled token |
|---|---:|---:|
| Initial stabilization | 80% | 20% |
| Middle | 50% | 50% |
| Mature rollout | 10–20% | 80–90% |

These percentages are tentative and should be configurable.

### 7.4 Wrong committed tokens are intentional training context

Standard absorbing-mask BD3 cannot change an already unmasked token. If the model commits a wrong token, subsequent predictions can adapt to it but cannot repair it.

The memory model should learn to:

- prevent one wrong token from corrupting the rest of the block;
- retrieve a less-corrupted previous representation;
- preserve semantic hypotheses that existed before commitment;
- choose remaining tokens coherently.

Do not compute the rollout loss on already committed positions. Compute it on positions that remain masked.

If actual correction of committed tokens is desired, remasking or replacement is a separate future extension.

---

## 8. Training objective

### 8.1 Preserve the original BD3 loss

Do not replace vanilla BD3 training with all-mask rollouts. Keep the efficient, principled loss over all blocks and sampled noise levels:

\[
\mathcal L_{\mathrm{base}}=\mathcal L_{\mathrm{BD3}}.
\]

### 8.2 Add a recurrent rollout auxiliary loss

Select one active block per sequence (or a small subset) and run a recurrent on-policy rollout starting from all mask:

\[
\mathcal L
=\mathcal L_{\mathrm{BD3}}
+\lambda_{\mathrm{mem}}\mathcal L_{\mathrm{rollout}}.
\]

For \(H\) memory-conditioned states:

\[
\mathcal L_{\mathrm{rollout}}
=\frac{1}{H}\sum_{h=1}^{H}
\mathcal L_{h,\mathrm{remaining\ mask}}.
\]

Average rather than sum so increasing rollout length does not silently increase the auxiliary loss scale.

### 8.3 High-level training pseudocode

```python
base_loss = vanilla_bd3_loss(batch)

target_block_index = sample_one_block_per_sequence(batch)
target_block = batch.block(target_block_index)
clean_prefix = batch.before(target_block_index)

cfg = memory_curriculum(progress, block_size)

x = torch.full_like(target_block, mask_id)
memory = None
rollout_losses = []

# Bootstrap forward: this writes the first memory from a null predecessor.
output = model(
    clean_prefix=clean_prefix,
    active_block=x,
    step_memory=None,
    return_step_kv=True,
)

bootstrap_loss = loss_on_masked_positions(output.logits, target_block, x)
memory = maybe_detach(output.step_kv)

for _ in range(cfg.rollout_steps):
    with torch.no_grad():
        x = reveal_model_sampled_tokens(
            x=x,
            logits=output.logits,
            reveal_count=cfg.reveal_count,
            top_p=0.9,
        )

    output = model(
        clean_prefix=clean_prefix,
        active_block=x,
        step_memory=memory,
        return_step_kv=True,
    )

    rollout_losses.append(
        loss_on_masked_positions(output.logits, target_block, x)
    )
    memory = maybe_detach(output.step_kv)

rollout_loss = torch.stack(rollout_losses).mean()
loss = base_loss + memory_weight * rollout_loss
```

Clarify whether the bootstrap loss is included in the base objective, separately weighted, or omitted from `rollout_loss`. The recommended first implementation logs it and gives it a small auxiliary weight so the all-mask writer is trained even when KV is detached.

### 8.4 Gradient through recurrence

Initial safe implementation:

- discrete token sampling is always stop-gradient;
- detach `step_kv` between denoising evaluations;
- train the reader, writer projections, and gates through local losses.

Later ablation:

- backpropagate through 3–4 continuous KV transitions;
- still detach discrete samples;
- compare against fully detached recurrence.

MetaState reports that detaching its recurrent state hurts, so short BPTT is important to test. It should not be assumed necessary before the detached prototype is stable.

---

## 9. Curriculum

### 9.1 Curriculum parameters

Use a pair:

\[
(\delta,H),
\]

where:

- \(\delta\) is the expected fraction of the **full active block** revealed per transition;
- \(H\) is the number of recurrent transitions after the all-mask bootstrap;
- total forward passes are \(H+1\).

Current intended schedule:

\[
(0.05,1)\rightarrow(0.20,4).
\]

For block size 16:

| Curriculum point | Expected reveals/transition | Recurrent transitions | Total forwards |
|---|---:|---:|---:|
| ((0.05,1)) | 0.8, forced/stochastically rounded to at least 1 | 1 | 2 |
| ((0.10,2)) | 1.6 | 2 | 3 |
| ((0.15,3)) | 2.4 | 3 | 4 |
| ((0.20,4)) | 3.2 | 4 | 5 |

Block size 4 is useful for smoke testing, but this fractional curriculum is nearly degenerate there because both 0.05 and 0.20 reveal fewer than one token and are forced to one.

### 9.2 Suggested scheduler

```python
import math
import torch


def stochastic_round(value: float) -> int:
    lower = math.floor(value)
    return lower + int(torch.rand(()).item() < value - lower)


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def memory_curriculum(
    progress: float,
    block_size: int,
    curriculum_fraction: float = 0.30,
):
    phase = smoothstep(progress / curriculum_fraction)

    reveal_fraction = 0.05 + phase * (0.20 - 0.05)
    rollout_continuous = 1.0 + phase * (4.0 - 1.0)

    reveal_count = max(
        1,
        stochastic_round(reveal_fraction * block_size),
    )
    rollout_steps = max(1, stochastic_round(rollout_continuous))

    return {
        "reveal_fraction": reveal_fraction,
        "reveal_count": reveal_count,
        "rollout_steps": rollout_steps,
        "num_forwards": rollout_steps + 1,
    }
```

### 9.3 Mature-training randomization

After reaching the endpoint, do not make every batch use the same path. Sample around it:

```python
rollout_steps = random.choice([2, 3, 4])
reveal_fraction = random.uniform(0.05, 0.20)
```

Suggested optional mixture:

- 25% of rollout batches force one-token transitions to retain first-hitting compatibility;
- 5% use a complete within-block trajectory;
- the rest use the curriculum/random dense transitions.

The exact percentages are hyperparameters, not settled results.

### 9.4 Train/inference schedule warning

If training reveals 3–4 tokens per transition but inference always uses one-token first-hitting, there is a transition-width mismatch. Either:

- use a matching dense reveal schedule at inference;
- retain a meaningful fraction of one-token rollout training;
- evaluate both samplers.

---

## 10. Proposed implementation map in `bd3lms`

The next coding agent should inspect the current repository revision before editing. Expected touch points follow.

### 10.1 Configuration

Add a dedicated configuration section, for example:

```yaml
step_memory:
  enabled: false
  attention_mode: separate_softmax_shared_qo
  gate_type: per_head
  gate_init: 0.0
  write_from: post_layer
  detach_between_steps: true
  temporal_bias: none
  rollout_weight: 0.1
  rollout_block_count: 1
  curriculum_fraction: 0.30
  reveal_fraction_start: 0.05
  reveal_fraction_end: 0.20
  rollout_steps_start: 1
  rollout_steps_end: 4
  first_hitting_mix: 0.25
  full_rollout_probability: 0.05
  teacher_token_probability_start: 0.80
  teacher_token_probability_end: 0.15
```

Keep existing `sampling.kv_cache` behavior unchanged for inter-block history.

### 10.2 Transformer model

Likely file: `models/dit.py` or the relevant attention implementation under `models/`.

Required API capabilities:

```python
model.forward(
    x,
    sigma,
    previous_step_kv=None,
    return_step_kv=False,
    ...,
)
```

At each layer:

1. Compute normal current attention.
2. If previous-step KV exists, compute memory attention with a separate softmax.
3. Combine under the zero-initialized per-head gate.
4. Complete the layer/MLP.
5. Write next-step K/V from the post-layer hidden state.
6. Return the per-layer memory without mutating the inter-block cache.

Do not overload the existing `store_kv` flag until the semantics are made explicit; it currently refers to completed-block caching.

### 10.3 Diffusion/training module

Primary file: [`diffusion.py`](https://github.com/kuleshov-group/bd3lms/blob/main/diffusion.py).

Keep `_forward_pass_diffusion` and the vanilla loss behavior intact for `base_loss`. Add a separate helper such as:

```python
def _step_memory_rollout_loss(self, x0, attention_mask, progress):
    ...
```

It should:

- select one target block per example;
- build the all-mask active state;
- condition on the appropriate clean prefix in the same way the baseline objective expects;
- run the bootstrap plus curriculum-selected transitions;
- sample model tokens with stop-gradient;
- accumulate loss only on remaining masked positions;
- return rollout metrics as well as loss.

### 10.4 Sampling module

Relevant functions in `diffusion.py` include:

- `_semi_ar_sampler`;
- `_ddpm_caching_update`;
- `_sample`.

Add an explicit local variable for `previous_step_kv` within each active-block loop:

```python
previous_step_kv = None

for denoise_step in active_block_steps:
    logits, next_step_kv = model(
        active_block,
        previous_step_kv=previous_step_kv,
        return_step_kv=True,
    )
    active_block = sampler_update(active_block, logits)
    previous_step_kv = next_step_kv

# Reset when moving to the next block.
previous_step_kv = None
```

The completed-block KV cache should continue to persist across blocks.

### 10.5 Checkpoint compatibility

The model should load an official vanilla checkpoint with missing new parameters initialized as:

- memory gates: exactly zero;
- additional norms/projections: stable standard initialization;
- no effect on output while gates are zero.

Add a test that logits from the augmented model with memory disabled/zero-gated match vanilla logits within numerical tolerance.

---

## 11. Experimental sequence

### Milestone 0: reproduce baseline

- Run official block-size-16 likelihood evaluation.
- Run official sample/generative-PPL evaluation.
- Record exact checkpoint, commit, tokenizer, sampler, top-p, entropy, and NFE reporting.

Do not begin architectural comparisons until baseline reproduction is credible.

### Milestone 1: attention smoke test

- Add memory API and zero gates.
- Verify no-memory equivalence to vanilla.
- Feed synthetic previous-step KV and verify gradients reach memory attention/gates.
- Verify active-block memory resets between blocks.
- Verify completed-block cache behavior is unchanged.

Block size 4 is acceptable here because complete trajectories are cheap.

### Milestone 2: detached recurrent training

- Use block size 16 for the substantive experiment.
- Preserve base BD3 loss.
- Add one selected-block rollout per sequence.
- Start from all mask.
- Use detached KV between steps.
- Use the ((0.05,1)\rightarrow(0.20,4)) curriculum.
- Log base loss, rollout loss, gate norms, memory-attention entropy, and memory-output norm.

### Milestone 3: schedule and recurrence ablations

Compare:

- fixed one-token first-hitting rollouts;
- dense reveal curriculum;
- mixture of one-token and dense rollouts;
- detached KV;
- 3–4-step BPTT through continuous KV;
- no explicit temporal bias;
- relative mask-ratio temporal bias.

### Milestone 4: full comparison

Minimum variants:

| Variant | Recurrent KV | Curriculum | Purpose |
|---|---:|---:|---|
| Vanilla BD3 | No | No | Baseline |
| Zero-gate control | Technically present, gate fixed zero | Same extra compute if possible | Implementation/control |
| Memory, teacher transitions | Yes | Yes | Exposure-bias diagnostic |
| Memory, on-policy transitions | Yes | Yes | Main model |
| Memory with shuffled cache | Yes, wrong example/step | Yes | Tests whether cache carries useful information |
| Memory disabled at inference | Trained yes, evaluated no | — | Measures reliance |

### Milestone 5: causal motor layers (deferred)

Only after memory is evaluated, compare layer masks using the 12-layer architecture:

- all bidirectional: `BBBBBBBBBBBB`;
- terminal causal motor layers: `BBBBBBBBBBCC`;
- periodic causal motor layers: `BBBBBCBBBBBC`.

Here `C` means causal attention within the active block. These are not globally AR layers because their inputs already contain bidirectionally mixed information from lower layers.

Cross causal-layer placement with memory on/off only after the memory mechanism has a stable baseline.

---

## 12. Evaluation

### 12.1 Likelihood perplexity

Use the repository's BD3 likelihood/NELBO evaluation on the OpenWebText validation setup. Keep this distinct from generative perplexity.

Report:

- token NELBO/NLL;
- perplexity derived from it;
- estimator settings and number of samples if applicable.

### 12.2 Generative evaluation

Use a fixed external evaluator, matching the repository where possible:

- GPT-2 Large generative perplexity;
- unigram entropy;
- sample length;
- sampler and top-p;
- denoising updates;
- repository-reported NFE;
- raw `model.forward` call count;
- wall-clock latency and peak memory.

Generative PPL must be interpreted jointly with entropy; repetitive outputs can obtain misleadingly good external-LM perplexity.

### 12.3 Memory-specific diagnostics

Log or evaluate:

- per-layer/per-head gate values;
- memory attention entropy;
- fraction of attention mass on same-position versus cross-position keys;
- cosine similarity of step memories across trajectory;
- memory-output norm relative to current-attention output;
- performance when cache is zeroed;
- performance when cache is shuffled between examples;
- performance when cache comes from the wrong denoising age;
- performance when only selected layers have memory;
- sensitivity to wrong committed tokens.

### 12.4 Closed-loop evaluation is mandatory

Short rollout training must be evaluated on full inference trajectories. Stability across the complete active-block denoising chain is the actual criterion.

---

## 13. Risks and failure modes

### 13.1 Memory is ignored

Symptoms:

- gates remain near zero;
- shuffling memory does not hurt;
- disabling memory at inference changes nothing.

Responses:

- increase rollout loss weight cautiously;
- enable short BPTT;
- inspect whether memory is written after retrieval;
- confirm gradients reach memory writer and gates;
- increase rollout horizon.

### 13.2 Memory copies current state without adding information

Adjacent first-hitting states differ by only one token. The model may learn a trivial near-identity cache.

Responses:

- include dense transitions;
- vary reveal width;
- use temporal-gap/mask-ratio conditioning;
- measure same-position versus cross-position retrieval;
- compare against simply reusing previous logits or embeddings.

### 13.3 Stale memory hurts

Large direct jumps from all mask to a much cleaner state produce stale \(M_1\), which does not match local recurrence. Avoid this by increasing consecutive rollout length, not just jump size.

### 13.4 Error amplification

An incorrect committed token can contaminate both the hard sequence and future memory.

Responses:

- on-policy training with wrong tokens;
- memory dropout;
- gating;
- optional confidence metadata;
- later remasking/replacement extension.

### 13.5 Loss/objective drift

The on-policy rollout state with wrong committed tokens is not an unbiased draw from the original forward corruption process. Keep the original BD3 objective as the primary likelihood objective and treat rollout loss as auxiliary.

### 13.6 Compute blow-up

Do not recursively roll out every block of every 1,024-token sequence. Select one block per sequence or a small subset. Average rollout loss. Track actual forward calls, not only repository NFE.

### 13.7 Train/inference transition mismatch

Dense reveal training and single-token first-hitting inference are different state distributions. Retain one-token rollout batches and evaluate a matching dense sampler.

### 13.8 Positional confusion

Current and previous-step KV contain the same token coordinates. Separate softmaxes and memory source identity reduce ambiguity. If needed, add relative mask-ratio bias before trying 2D RoPE.

---

## 14. Decisions, tentative choices, and deferred ideas

### Decided for the first substantive implementation

- Base repository: `kuleshov-group/bd3lms`.
- Main substantive block size: 16.
- Smoke-test block size: 4 is acceptable.
- Backbone: official 12-layer, 768-wide, 12-head small model.
- Keep vanilla BD3 loss.
- Add one selected-block recurrent rollout loss.
- Every rollout starts from the all-mask state with null memory.
- Retain only immediately previous denoising KV.
- Memory is local to the active block.
- Keep inter-block KV cache separate.
- Separate current and memory softmaxes.
- Share query and output projections.
- Use zero-initialized per-head memory gates.
- Write memory from post-memory/post-layer hidden state.
- Sample rollout tokens from the model increasingly often.
- Causal/AR motor layers are deferred.

### Tentative/configurable

- Curriculum endpoint ((0.20,4)).
- Curriculum duration 30% of training.
- Teacher-to-on-policy transition mixture.
- Rollout loss weight.
- Fraction of one-token and complete-trajectory batches.
- Whether bootstrap loss receives a separate weight.
- Detached recurrence versus short continuous-state BPTT.
- Relative mask-ratio temporal bias.

### Explicitly deferred

- Storing all historical denoising KVs.
- Compressing memory into slots.
- Permanent all-mask anchor memory in addition to local memory.
- Concatenated current/memory KV under one softmax.
- Dedicated memory K/V writer projections.
- Literal two-dimensional RoPE.
- Causal motor layers.
- Remasking or correcting already committed tokens.
- Porting the mechanism to ELF-B continuous embedding flow.

---

## 15. Questions the implementation should answer empirically

1. Do zero-initialized gates open during training, and in which layers/heads?
2. Does shuffled memory hurt, proving that the model uses sample-specific information?
3. Does memory improve true BD3 likelihood, or only sample quality?
4. Does it reduce cascading damage after a wrong committed token?
5. Is one-token first-hitting sufficient, or are dense-gap rollouts required?
6. Does short BPTT improve over detached KV enough to justify its cost?
7. Is explicit denoising-age bias useful when only the previous state is stored?
8. Is memory most useful in early, middle, or late layers?
9. Does block size 16 provide enough workspace to expose benefits that block size 4 cannot?
10. After memory works, do terminal causal motor layers add a complementary benefit?

---

## 16. Suggested first coding task for a new Codex agent

1. Clone/open the exact current revision of `kuleshov-group/bd3lms`.
2. Reproduce the block-size-16 baseline evaluation with the official checkpoint.
3. Inspect the DIT attention implementation and its existing completed-block KV cache API.
4. Propose a minimal patch that adds:
   - `previous_step_kv` input;
   - `return_step_kv` output;
   - separate memory attention softmax;
   - shared Q/output projections;
   - zero-initialized per-head gates;
   - post-layer memory write.
5. Add unit tests for exact zero-gate equivalence and cache reset semantics.
6. Do not implement the rollout curriculum until inference-time recurrent KV works on a short synthetic block.

The implementation should be incremental and preserve the ability to run the unmodified baseline from the same codebase.

---

## 17. Primary references

- [BD3-LM repository](https://github.com/kuleshov-group/bd3lms)
- [Block Diffusion paper](https://arxiv.org/abs/2503.09573)
- [MDLM repository](https://github.com/kuleshov-group/mdlm)
- [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- [ELF repository](https://github.com/lillian039/ELF)
- [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938)
- [MetaState: Persistent Working Memory for Discrete Diffusion Language Models](https://arxiv.org/abs/2603.01331)
- [Scaling Beyond Masked Diffusion Language Models](https://arxiv.org/abs/2602.15014)
