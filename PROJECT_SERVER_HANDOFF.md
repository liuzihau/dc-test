# Recurrent Denoising QKV Memory for BD3-LM

> **2026-09-02 implementation consolidation:** The four canonical training
> variants now share `scripts/train/run_canonical_trial.sh`; aligned validation
> uses `scripts/eval/eval_checkpoint_validation.sh`. DCache + final-state
> recurrence is now propagated by the real sampler, not only by training and
> validation. `scripts/eval/eval_final_state_interventions.py` independently
> tests correct/shuffled/absent DCache and final-state sources. New training
> runs retain the latest three 500-step checkpoints and make `last.ckpt` a
> symlink. The older snapshot below remains design history.
>
> **2026-08-25 canonical project layout:** Current analysis recognizes four
> experiments only: BD3/MDLM vanilla, the objective-aligned no-memory control,
> DCache-v2, and DCache plus detached final-state reuse. Their paths, metrics,
> and status are defined in `experiments/canonical_runs.json`. Run
> `python scripts/results/refresh_canonical_results.py` to regenerate the
> curated figures and tables under `results/generated/`. Failed runs, smoke
> tests, DCache-v1, and superseded plots are recoverably stored under
> `archive/legacy_outputs_2026-08-25/` and must not be used for claims.
>
> The active DCache + final-state trial intentionally disables tentative-token
> correction, token-status embeddings, confidence supervision, and the
> latent-mask auxiliary pass. It is a clean dual-memory test aligned with the
> DCache-v2 five-state objective. See `DCACHEHOOPING_IMPLEMENTATION.md` and
> `results/README.md`.
>
> **2026-08-24 Dcachehooping implementation:** An opt-in DCache-v2 extension
> now propagates the detached final-layer representation, adds zero-initialized
> mask/committed/tentative status embeddings, trains direct tentative-token
> correction with an unclamped editable head, and learns RemeDi-style token
> confidence. DCache-v2 remains available unchanged when the feature is off.
> The exact objective, source contamination, compatibility guarantees,
> metrics, tests, and the GPU-2/3 launcher are in
> `DCACHEHOOPING_IMPLEMENTATION.md`.
>
> **2026-08-18 current research status:** Start with
> `DCACHE_RESEARCH_SUMMARY.md`. It contains the current research question,
> architecture, V1 and V2 controlled results, experiment criteria,
> and task list. For exact V2 equations and launch details, then read
> `DCACHE_V2_IMPLEMENTATION_WORK_NOTE.md`. The older prototype discussion below
> is retained as design history and is not authoritative where it conflicts
> with those two files.
>
> **Completed run:** DCache-v2 finished 5000 optimizer steps on OpenWebText
> using two RTX 3090 GPUs, length 1024, and global batch size 512. Final
> cache-warmed `t2` validation was `NLL=3.6067`, `PPL=36.84`, and mean gate
> `0.1885`. The vanilla final validation was `NLL=3.9329`, but these training
> validation distributions are not exactly matched.
>
> **Most important result so far:** DCache-v1 strongly preferred a present
> cache over no cache, yet a batch-shuffled cache was only about
> `0.0009--0.0017` NLL worse than the correct cache. The model learned cache
> dependence without meaningful document-specific cache identity. V2 directly
> targets this failure using nearby exact nested states, source dropout, an
> explicit correct-versus-shuffled identity loss, and learned residual gates.
>
> **2026-08-19 objective-matched control B:** The repository now has a
> parameter-identical vanilla MDLM control that uses DCache-v2's exact
> five-state nested-mask sampler and normalized token-loss weights, while
> disabling the DCache architecture and every cache treatment. Its objective
> is `(0.05 L_full + 0.10 L_t0 + 0.20 L_t1 + 1.00 L_t2 + 0.70 L_t3) / 2.05`.
> Run `scripts/train/train_owt_mdlm_objective_matched_5k.sh`; exact invariants,
> GPU commands, metrics, and resume behavior are in `DCACHE_RUNBOOK.md`.
> Targeted tests are in `tests/test_objective_matched_pretrain.py`.
>
> **Controlled result:** On the same 800 documents and exact token masks, V2
> correct cache beats vanilla across all six tested transitions by
> `0.1097--0.2055` NLL. Correct cache beats shuffled cache by
> `0.0216--0.0555` NLL, with every paired 95% confidence interval above zero.
> V2 therefore passes the short-run cache-identity and matched-quality gates.
> The next requirements are on-policy generation, source-only/per-layer
> ablations, and compute-normalized evaluation before a large run.
>
> **2026-08-10 shifted-pretraining update:** Read
> `FINAL_DENOISING_CACHE_PLAN.md` first. It now supersedes the same-layer
> architecture below. The implemented proposal runs DCache attention before
> normal attention; layer `l` reads previous `M_(l+1)`; only `M13` needs an
> explicit final writer; and pretraining uses the teacher-forced three-pass
> `100% -> s -> t` objective with relative weights `0.1/1/1`. The matched
> vanilla and recurrent launchers now target 100k full-sequence MDLM
> pretraining updates.
>
> **2026-08-09 final-plan update:** Read `FINAL_DENOISING_CACHE_PLAN.md` first.
> It supersedes this handoff where the documents disagree, including the
> separate denoising QKV/output sublayer, joint previous+current denoising
> softmax, parameter-free 2D RoPE, removal of the tanh gate, from-scratch
> training, and the revised rollout curriculum down to one remaining mask.
> The `dcache` environment, end-to-end CUDA smoke test, full-size VRAM profile,
> and launch/evaluation commands are recorded in `DCACHE_RUNBOOK.md`.

## Historical server handoff and former project state

**Snapshot date:** 2026-08-08  
**Research repository:** `ar-dcache/bd3lms`  
**Upstream:** `kuleshov-group/bd3lms`  
**Upstream revision used:** `1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7`  
**Detailed original research plan:** `bd3_denoising_kv_research_plan.md`

This section records the 2026-08-08 state and is retained for design history.
For the current project state, read `DCACHE_RESEARCH_SUMMARY.md` first.

---

## 1. Project in one paragraph

BD3-LM generates text autoregressively across blocks and denoises bidirectionally inside the active block. Vanilla BD3 recomputes the active block at every denoising evaluation and discards its continuous internal state. This project adds a recurrent per-layer memory: the current active-block queries separately attend to the immediately preceding denoising evaluation's active-block K/V. Normal BD3 attention and memory attention have separate softmaxes, and a zero-initialized per-head gate adds the memory result before the existing output projection. Completed-block cache and previous-denoising memory are independent. The first experiment must study this memory alone; causal/AR layers are deferred.

---

## 2. Motivation and scope

### 2.1 Hypothesis

At a denoising evaluation, the Transformer may represent possible words, semantic plans, confidence, or reasoning information that is richer than the hard tokens selected by the sampler. Vanilla masked diffusion reduces the result to sampled tokens and masks, then discards the continuous state. The next evaluation therefore cannot directly recover hypotheses that were present but not committed.

The hypothesis is that the preceding evaluation's continuous attention state can act as a small recurrent denoising workspace and improve coherence, robustness, likelihood, or sample quality.

### 2.2 Sources of inspiration, stated precisely

- **BD3-LM** is the actual model, objective, block factorization, checkpoint, sampler, and evaluation foundation.
- **J-space/global-workspace work** motivates the belief that internal continuous representations can contain useful, reportable concepts that are not identical to emitted tokens. It does not propose recurrent diffusion KV memory.
- **Nemotron-Labs-Diffusion** motivates the later hypothesis that AR and diffusion objectives provide complementary priors. It does not directly justify changing selected BD3 layers to causal layers.
- **DSpark** motivates introducing intra-block sequential dependency in a parallel model. It is relevant to a later causal-layer experiment, not the current memory mechanism.
- **LLaDA 2.0** was inspected for caching. It uses a fused QKV projection but stores only K/V in a standard cache. Its cache is stable-prefix/completed-block computational reuse, not previous-denoising recurrent memory.

### 2.3 Current stage boundary

Do now:

1. Preserve vanilla BD3 architecture and base loss.
2. Add previous-denoising active-block attention memory.
3. Establish exact zero-gate compatibility.
4. Make inference recurrence work. **Completed for both DCache and the
   detached previous final state on 2026-09-02.**
5. Add a small differentiable rollout objective.
6. Profile memory and speed before scaling the rollout horizon.

Do not do yet:

- replace any Transformer layers with causal/AR layers;
- store the entire denoising history;
- remask committed tokens;
- add compressed memory slots;
- add a dedicated post-layer writer before the simpler cache is evaluated.

---

## 3. Cache terminology

There are two independent states:

| State | Meaning | Lifetime | Gradient role |
|---|---|---|---|
| `block_kv_cache` / existing BD3 `kv_cache` | Clean completed blocks used as stable prefix | Persists across blocks | Primarily inference reuse |
| `previous_step_qkv` | Immediately preceding evaluation of the current active block | Replaced each real denoiser forward; reset at a new block | Recurrent learned memory |

The upstream native BD3 cache happens to store fused QKV, even though prefix queries are unnecessary. LLaDA 2.0 cleanly stores K/V only. The prototype stores fused QKV for the new step memory because this matches BD3's current tensor layout and is easy to implement. Only its K/V slices are read. With block size 16, the extra stored Q is small and can be removed later.

---

## 4. Chosen attention architecture: Option B

For layer `l` and denoising evaluation `t`, compute normal BD3 attention:

```text
Q_current(active block)
    -> K/V(clean completed prefix + current active block)
    -> normal_output
```

Compute a separate memory attention:

```text
Q_current(active block)
    -> K/V(previous evaluation of this active block)
    -> memory_output
```

Combine them before BD3's existing attention output projection:

```text
combined_head_output = normal_output + tanh(per_head_gate) * memory_output
layer_attention_output = existing_output_projection(combined_head_output)
```

The two branches use separate softmax normalizations. The clean prefix is not repeated in the memory branch. This preserves normal-attention probabilities and ensures a zero gate reproduces vanilla BD3.

Do not initially concatenate previous K/V with current K/V under one softmax. That would force current and memory coordinates to compete, change the vanilla normalization, and remove exact zero-impact initialization.

---

## 5. Recurrent rollout and gradient decision

The intended training trajectory for one selected block is:

```text
Forward 0:
  all-mask active block + null step memory
  -> loss_0
  -> step_qkv_0

sample/reveal one or more tokens without differentiating through sampling

Forward 1:
  new active state + step_qkv_0
  -> loss_1
  -> step_qkv_1

sample/reveal again

Forward 2:
  newer active state + step_qkv_1
  -> loss_2
  -> step_qkv_2
```

### 5.1 Do not detach continuous step memory in the main experiment

Discrete token selection remains stop-gradient, but `step_qkv_t` should stay attached for a short rollout. Consequently:

```text
loss_t teaches the current forward to denoise the current state
loss_(t+1) additionally teaches forward t to write useful state for the next evaluation
```

This is analogous to an earlier AR position receiving learning signal because its K/V helps later positions. The future loss does not replace the current denoising loss; it adds a future-use signal.

The gate starts at zero. On the first optimization step, the future gradient mainly opens the gate. Once it is nonzero, gradients can flow through memory into the preceding forward.

Detached recurrence remains a useful memory-saving ablation and may be necessary for long rollouts on 24 GB GPUs, but it is no longer the preferred main configuration.

### 5.2 Current writer versus future writer

The current prototype saves the ordinary QKV produced near the start of each layer. This is the simplest smoke test and reuses checkpoint projections.

It does not give a same-layer post-retrieval write: layer `l` saves QKV before layer `l` consumes memory. Higher-layer QKV can still contain effects introduced by lower memory-enabled layers. A later stronger variant may project new K/V from the post-attention, post-MLP layer output, allowing every layer to write an explicitly updated same-layer recurrent state.

---

## 6. Base loss and future rollout loss

The normal BD3 likelihood objective must remain primary:

```text
total_loss = base_bd3_loss + rollout_weight * recurrent_rollout_loss
```

The auxiliary rollout should:

1. Choose one target block per example or per sequence batch.
2. Use the correct clean prefix.
3. Start the target block at all mask and memory at `None`.
4. Run at least two forwards.
5. Reveal tokens using a configurable mixture of teacher tokens and model samples.
6. Compute each auxiliary loss only on positions still masked in that state.
7. Average losses across rollout states rather than sum them.
8. Keep token sampling stop-gradient.
9. Initially use a very short unrolled graph and increase only after VRAM profiling.

The detailed curriculum proposal remains in `bd3_denoising_kv_research_plan.md`. Its old statement that detached memory trains a dedicated writer through following losses is superseded by this document.

---

## 7. Work already implemented

The following changes currently exist as uncommitted modifications in `bd3lms`.

### 7.1 Configuration

`configs/config.yaml` contains:

```yaml
step_memory:
  enabled: false
  gate_init: 0.0
  detach_between_steps: false
```

`detach_between_steps` records the planned ablation but is not yet wired into a training rollout helper.

### 7.2 Native DIT model

`models/dit.py` now provides:

- a `step_memory_gate` parameter for every non-causal block;
- `previous_step_qkv` input;
- `return_step_qkv` output;
- separate SDPA memory attention;
- current active-block queries only;
- previous active-block K/V only;
- fused active-block QKV output per layer;
- shared existing attention output projection;
- zero-gate vanilla behavior.

The per-layer memory tensor is:

```text
[batch, active_block_length, 3, heads, head_dim]
```

The model returns a Python list with one such tensor per layer.

### 7.3 Diffusion and sampling path

`diffusion.py` now passes step memory through the native BD3 backbone. In `_semi_ar_sampler`:

- `previous_step_qkv` begins as `None` for every new active block;
- it is replaced after each real model evaluation;
- reuse of cached `p_x0` without a forward does not falsely advance memory;
- completed-block cache persists normally;
- the final clean-block cache-fill call remains separate.

The sampler is decorated with `no_grad`, as expected for inference. The differentiable training rollout has not yet been implemented.

### 7.4 Checkpoint and EMA compatibility

- Vanilla checkpoint loading uses `strict=False`, leaving new gates at their configured initialization.
- Old EMA state lists are migrated by inserting current gate values at the corresponding parameter positions.

This migration needs a real official-checkpoint test on the server.

### 7.5 Tests added

`tests/test_step_memory.py` covers:

- exact zero-gate equivalence;
- gradient flow from a second forward into the first forward when the gate is nonzero;
- active-block-only cache shape.

The files pass Python syntax compilation and `git diff --check`. Runtime tests have not run locally because the local Python environment lacks PyTorch.

### 7.6 Known implementation limitations

1. Step memory is implemented only for native `algo.backbone=dit`, not `hf_dit`.
2. The differentiable rollout loss is not implemented.
3. The current writer is pre-attention QKV, not post-layer QKV.
4. `detach_between_steps` is not wired yet.
5. Official checkpoint and EMA migration have not been exercised at runtime.
6. Zero-gate equivalence has not been measured against an official checkpoint on GPU.
7. Peak VRAM and wall-clock cost have not been profiled.
8. No causal/AR layers have been added, intentionally.

---

## 8. Can this run on one or two RTX 3090 GPUs?

### 8.1 Short answer

**Yes for baseline evaluation, inference smoke tests, and likely short-rollout fine-tuning. Not yet guaranteed for the mature `H=4` untruncated rollout. Full reproduction of the official one-million-step training is technically possible but not a practical target on one or two 3090s.**

The RTX 3090 has 24 GB VRAM and supports BF16 tensor operations. The BD3 small model is approximately a 12-layer, width-768 model in the roughly 100M-parameter class, so parameters, gradients, Adam state, and EMA are not the primary problem. Activations dominate because baseline cross-attention training processes a doubled sequence representation and untruncated recurrence retains several forward graphs.

### 8.2 Expected feasibility

| Workload | 1 x 3090 | 2 x 3090 DDP | Confidence |
|---|---|---|---|
| Unit tests and synthetic block test | Yes | Yes | High |
| Official checkpoint sample generation, batch 1 | Yes | Yes | High |
| Baseline likelihood evaluation with reduced eval batch | Yes | Yes | High |
| Vanilla small-model fine-tuning, microbatch 1-2 | Likely | Likely/faster | Medium-high |
| Recurrent inference, block size 16 | Yes | Yes | High |
| Two-forward rollout (`H=1`), microbatch 1 | Likely | Likely/faster | Medium |
| Three-to-five attached forwards | Uncertain; profile first | Still uncertain per GPU under DDP | Low until measured |
| Official global batch 512 via accumulation | Fits if microbatch fits, but slow | Faster, still slow | Medium |
| Full official 1M-step reproduction | Impractical timeline | Still very long | High |

### 8.3 Important two-GPU fact

The repository's default strategy is DDP. DDP places a complete model and its activations on each GPU. Two 24 GB GPUs do not automatically become one 48 GB pool. They approximately double data throughput and reduce the number of accumulation cycles. If one recurrent sample graph exceeds 24 GB, ordinary two-GPU DDP will still OOM.

The repository includes an FSDP configuration, but recurrent training has not been validated with it. FSDP can shard parameters, gradients, and optimizer state, but the main suspected cost here is retained activation graphs, so activation checkpointing or shorter BPTT may matter more.

### 8.4 Recommended initial server configuration

If cloud prices are comparable, one 48 GB GPU (for example, an RTX A6000 or
A40) is safer for this particular research phase than two 24 GB GPUs in DDP.
The experiment's unusual cost is one large retained recurrent graph, so
per-device memory is initially more valuable than aggregate data-parallel
throughput. Two 3090s become attractive after a single rollout sample is known
to fit comfortably below 24 GB.

Start conservatively:

```text
model: small
block_size: 16
backbone: native dit
attention backend: sdpa for the first correctness tests
precision: bf16
per-GPU training microbatch: 1
per-GPU evaluation batch: 1
rollout horizon H: 1
teacher/model transition: simple one-token reveal
step-memory gate: zero initialized
```

Do not begin with the official per-GPU batch 16 or the full `(0.20, 4)` curriculum.

Profile peak allocation after:

1. vanilla base loss only;
2. one memory bootstrap forward only;
3. two attached forwards;
4. three attached forwards.

Record both `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()`.

If two forwards OOM:

1. ensure microbatch is 1;
2. avoid holding unnecessary logits for every rollout state;
3. checkpoint Transformer layers;
4. detach the clean-prefix cache;
5. use truncated BPTT for step memory as a fallback;
6. only then consider FSDP.

---

## 9. Server migration and bootstrap checklist

### 9.1 Preserve the current work

The `bd3lms` tree is dirty and the changes are not committed. Migrate the entire `ar-dcache` directory, including hidden Git metadata, or commit the changes before transfer. Do not clone a fresh upstream repository and assume it contains this work.

Immediately verify on the server:

```bash
cd ar-dcache/bd3lms
git rev-parse HEAD
git status --short
git diff --check
```

Expected upstream HEAD:

```text
1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7
```

Expected modified files include:

```text
configs/config.yaml
diffusion.py
main.py
models/dit.py
tests/test_step_memory.py
```

### 9.2 Environment

Follow the upstream environment first. The current requirements pin PyTorch 2.7.1, Transformers 4.49.0, Lightning 2.5.0, Triton 3.3.1, and related packages. Use a CUDA/PyTorch combination supported by the cloud image and RTX 3090 rather than blindly mixing system CUDA libraries.

After installation:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())"
pytest -q tests/test_step_memory.py
```

### 9.3 First correctness sequence

1. Run the new unit tests.
2. Instantiate the tiny synthetic native DIT on GPU.
3. Load the official block-size-16 checkpoint with step memory disabled.
4. Compare vanilla logits before and after the patch.
5. Enable step memory while gates remain zero and compare again.
6. Set a gate manually nonzero and confirm memory changes logits.
7. Run a short block-size-4 sample and confirm memory reset at block boundaries.
8. Run block-size-16 recurrent inference.
9. Only then implement training rollout loss.

### 9.4 One-GPU launch principle

Use Hydra overrides rather than editing official files. Begin with overrides equivalent to:

```text
trainer.devices=1
loader.batch_size=1
loader.eval_batch_size=1
loader.global_batch_size=16 or 32 for prototype work
model=small
algo=bd3lm
algo.backbone=dit
block_size=16
model.attn_backend=sdpa
trainer.precision=bf16
step_memory.enabled=true
```

The exact training command should be finalized after the rollout helper exists.

### 9.5 Two-GPU launch principle

Keep per-GPU microbatch at 1 initially:

```text
trainer.devices=2
loader.batch_size=1
loader.eval_batch_size=1
strategy=ddp
```

This improves throughput but does not solve a single-sample activation OOM.

---

## 10. Next implementation tasks, in order

### Task 1: runtime validation of the current patch

- Install dependencies.
- Run `tests/test_step_memory.py`.
- Fix any CPU/GPU, RoPE, dtype, or SDPA shape issues.
- Add an official-checkpoint zero-gate equivalence test.
- Validate old EMA migration.

### Task 2: instrument recurrent inference

- Count real model forwards separately from sampler updates.
- Log layer/head gates.
- Assert step-memory list length and tensor shapes.
- Assert memory resets at a block boundary.
- Confirm completed-block cache remains unchanged.

### Task 3: implement the smallest training rollout

- Keep `_loss` and `_forward_pass_diffusion` unchanged for base loss.
- Add a separate helper for one selected block.
- Start all mask with memory `None`.
- Run bootstrap plus exactly one transition (`H=1`).
- Keep QKV attached.
- Stop gradients through sampled token IDs.
- Compute rollout CE only on remaining masks.
- Average bootstrap/conditioned auxiliary losses explicitly.
- Add the auxiliary result under a small configurable weight.

### Task 4: profile one 3090

- Microbatch 1.
- Measure base only, `H=1`, then `H=2`.
- Record allocated/reserved peak VRAM, step duration, and tokens/s.
- Do not expand the curriculum before this table exists.

### Task 5: choose the memory strategy based on profiling

Compare:

- attached QKV recurrence;
- detached recurrence;
- two- or three-step truncated BPTT;
- activation checkpointing;
- pre-attention versus post-layer writer.

### Task 6: substantive experiment

- Use block size 16.
- Fine-tune from the official compatible BD3 checkpoint.
- Preserve base loss.
- Gradually introduce on-policy transitions.
- Evaluate closed-loop full block trajectories.

Only after this stage should causal/AR motor layers be designed.

---

## 11. Required controls and evaluation landscape

Minimum comparisons:

| Variant | Purpose |
|---|---|
| Vanilla BD3 | Main baseline |
| Patched model, gate fixed zero | Implementation control |
| Recurrent memory, teacher transitions | Stabilization/exposure-bias diagnostic |
| Recurrent memory, on-policy transitions | Main experiment |
| Memory disabled at inference | Reliance test |
| Memory shuffled across examples | Information-content test |
| Memory from the wrong denoising age | Temporal-specificity test |
| Detached memory | Credit-assignment ablation |
| Short attached BPTT | Main gradient design |

Evaluate:

- BD3 likelihood/NELBO perplexity;
- external generative perplexity jointly with entropy;
- sample quality at matched real forward count;
- raw model forwards, repository NFE, latency, and peak VRAM;
- robustness after deliberately wrong committed tokens;
- gate values by layer/head;
- memory output norm relative to normal attention;
- memory attention entropy and same-position attention;
- full closed-loop block stability.

Evidence that the mechanism works requires more than a quality improvement. Shuffling, disabling, or aging the cache should hurt if it contains useful sample-specific information.

---

## 12. Central research questions

1. Do zero-initialized gates open, and where?
2. Does previous-step memory improve likelihood, generation, or only robustness?
3. Does it preserve hypotheses lost by discrete token commitment?
4. Does future-loss credit assignment materially outperform detached recurrence?
5. Is ordinary pre-attention QKV sufficient, or is a post-layer writer necessary?
6. How long a recurrent graph is useful before cost and instability dominate?
7. Does shuffled memory prove that the cache is sample-specific rather than a generic bias?
8. Does block size 16 provide enough workspace for the effect?
9. After memory works, are terminal or periodic causal layers complementary?

---

## 13. Instructions to the next agent

Read this file and `bd3_denoising_kv_research_plan.md`, then inspect the actual dirty diff before editing. Treat the current patch as an unvalidated prototype, not a finished training system. Preserve vanilla BD3 behavior and keep the two caches semantically separate. Do not add AR layers yet. Lead with runtime validation, exact equivalence, and one-3090 VRAM profiling. When a result is uncertain, measure it rather than silently changing the research design.
