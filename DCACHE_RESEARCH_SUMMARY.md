# Recurrent Denoising Cache for BD3/MDLM

**Research snapshot:** 2026-09-02
**Current status:** Canonical vanilla, objective-aligned, DCache-v2, and
DCache + detached final-state 5k runs are complete. Current
analysis recognizes only these four training lines; failed runs, smoke tests,
DCache-v1, and full tentative/confidence DCachehooping attempts are archived
and excluded from claims. DCache-v2 EMA controlled evaluations remain the
current mechanism evidence: it uses document-specific cache content, but its
held-out correct-versus-shuffled identity gap plateaued from 5k to 6k.
**Canonical manifest:** `experiments/canonical_runs.json`
**Result index:** `results/README.md`
**Authoritative V2 note:** `DCACHE_V2_IMPLEMENTATION_WORK_NOTE.md`
**Final-state note:** `DCACHEHOOPING_IMPLEMENTATION.md`

![Canonical four-way training health](results/generated/figures/training/four_way_5k_smooth60.png)

The plotted training points use a 60-point rolling mean. The faint raw curve
is intentionally retained because individual logged minibatches span very
different mask ratios and therefore have intrinsically different difficulty.

## Research Aim

Determine whether a masked/block-diffusion language model can use the
continuous state from its immediately preceding denoising step as recurrent
memory. The primary scientific test is whether this memory improves matched
conditional NLL, perplexity, prediction accuracy, and eventually generation
quality or the number of denoising evaluations required, relative to a vanilla
BD3/MDLM model trained on the same data and update budget.

The stronger form of the claim is not merely that a cache branch helps. The
cache must carry useful, example-specific information from the previous
denoising state.

## Motivation

Vanilla masked diffusion repeatedly computes rich hidden representations and
then discards them. Only the newly revealed hard tokens survive into the next
step. That creates an information bottleneck: semantic alternatives,
confidence, and partial plans present in the previous forward pass cannot be
read directly by the next pass.

Ordinary prefix KV caching solves a different problem. It avoids recomputing
already completed prefix blocks, but it does not retain the evolving state of
the active denoising block. Our DCache keeps these two memories separate:

- the normal BD3 prefix cache stores stable completed-block K/V;
- DCache stores one previous denoising state for the active block.

## Current Findings

1. **The original 5k baseline is stronger than DCache-v1.** The vanilla model
   ended at validation NLL `3.9329` (PPL `51.06`), while DCache-v1 ended at
   NLL `4.0802` (PPL `59.16`). Their training objectives are not directly
   comparable, so matched teacher-forced evaluation is the stronger result.

2. **DCache-v1 learned to rely on having a cache branch.** On fixed corruption,
   correct cache substantially beat no cache. At 50% masks, for example, NLL
   was `3.8871` with correct cache versus `5.8976` with no cache.

3. **DCache-v1 did not learn meaningful cache identity.** In controlled
   transitions, shuffling cache entries across documents changed NLL by only
   about `0.0009` to `0.0017`, although zeroing or removing the cache caused a
   large degradation. Previous attention diagnostics also found roughly
   25--35% of DCache attention mass on previous-step keys. Together, these
   observations mean that reading the branch is not evidence that the model is
   reading useful document-specific content.

4. **The V1 DCache checkpoint remained worse than the baseline at all tested
   nontrivial fixed mask ratios.** The correct-cache minus baseline NLL gap was
   approximately `+0.13` to `+0.18` from 5% through 70% masks, then narrowed
   near the full-mask boundary.

5. **V2 completed cleanly and improved throughout the short run.** Final
   cache-warmed `t2` validation NLL/PPL was `3.6067/36.84`, and the mean
   learned DCache gate grew from about `0.10` to `0.1885` rather than
   collapsing to zero. This training-loop validation remains a health metric,
   not the primary matched comparison.

6. **V2 solves the V1 cache-identity failure under held-out evaluation.** In
   six controlled teacher-forced transitions, shuffled-cache minus
   correct-cache NLL is `+0.0216` to `+0.0555`. Every paired 95% bootstrap
   interval is strictly above zero. V1's corresponding difference was only
   about `+0.0009` to `+0.0017`.

7. **V2 beats vanilla under identical corrupted inputs.** In fixed-corruption
   evaluation, correct-cache V2 improves NLL by `0.1023` to `0.1797` at
   5%--70% masks and by `0.0221` at 90% masks. At the 100%-masked boundary it
   is slightly worse by `0.0033`. In transition evaluation, correct-cache V2
   improves NLL over vanilla by `0.1097` to `0.2055`, with all paired 95%
   intervals excluding zero.

8. **The recurrent cache adds a real but smaller part of the total gain.** In
   transition evaluation, removing the cache worsens V2 by `0.0284` to
   `0.0507` NLL. The no-cache V2 model itself still beats vanilla, so the total
   improvement cannot be attributed entirely to recurrent retrieval; the
   multi-state objective and altered architecture/training also contribute.

9. **The noisier raw training curve near step 1900 was not divergence.** Raw
   `t2` loss was strongly correlated with the sampled `t2` mask ratio, while
   fixed validation continued to improve and the run completed normally.
   A logged training point represents a small synchronized minibatch, whereas
   each optimizer update accumulates a global batch of 512 sequences.

10. **The experiment is step-matched, not compute-matched.** V2 uses five
   forwards per update plus occasional shuffled-cache supervision. Its quality
   result justifies further study, but does not yet establish better quality
   per FLOP or wall-clock hour.

11. **Another 1000 steps did not enlarge the held-out cache-identity gap.**
   The macro correct-minus-shuffled top-1 gap is `0.689`, `0.681`, and `0.660`
   percentage points at metric indices 4999, 5499, and 5999. The corresponding
   macro shuffled-minus-correct NLL gaps are `0.04586`, `0.04618`, and
   `0.04615`. The paired 5999-minus-4999 change in macro accuracy gap is
   `-0.0289` points with 95% CI `[-0.0914, +0.0340]`: statistically consistent
   with a plateau, not growth or meaningful regression.

12. **Overall correct-cache quality still improved from 5k to 6k.** Macro
   top-1 accuracy across the six transitions rose from `49.982%` to `51.041%`
   (`+1.060` points; paired 95% CI `[+1.005, +1.117]`), while macro conditional
   NLL fell from `2.7890` to `2.7094` (`-0.0796`; paired 95% CI
   `[-0.0812, -0.0780]`). Every individual transition improved. Thus the model
   continued learning general predictive quality, but not a stronger relative
   dependence on correct versus shuffled cache.

13. **The objective-aligned and dual-memory controls completed.** At the
    final aligned training-validation point, objective-matched no-memory,
    DCache-v2, and DCache + final-state report `val/loss_t2` values of
    `3.6421`, `3.6067`, and `3.5825`, respectively. These are health metrics,
    not a substitute for the fixed-state causal evaluation.

14. **Generation-time recurrence now matches training.** The sampler carries
    both previous-step DCache and the detached previous final-layer state,
    resets both at a new active block, and leaves both unchanged when cached
    logits are reused without a real model forward. A five-condition
    independent DCache/final-state intervention evaluator is implemented; a
    two-document GPU smoke test completed successfully, while the full
    800-document result remains to be run.

Important comparison caveat: V2 validation follows its five-forward trajectory
and reports `val/nll` at `t2`. Its mean `t2` mask ratio is slightly below the
vanilla uniform-mask mean and its cache has already been warmed by earlier
forwards. Therefore the current V2 and vanilla validation curves are useful
health checks, but not a fully matched scientific comparison.

## Proposed Method

### Model structure

For each Transformer layer `l` in the current denoising step:

1. Pre-normalize the current hidden state for DCache attention.
2. Compute a separate DCache Q/K/V projection.
3. Apply parameter-free 2D RoPE: one coordinate represents token position and
   the second distinguishes previous-step keys from current-step queries/keys.
4. Run one DCache softmax over the concatenation of previous and current
   DCache K/V. Source dropout may restrict this to one source during training.
5. Project the DCache output and add it through a learned per-layer
   `tanh`-bounded residual gate.
6. Run the normal BD3 attention sublayer and its residual connection.
7. Run the normal MLP sublayer and its residual connection.

The shifted memory mapping is deliberate. Reader layer `l` reads the previous
forward's `M_(l+1)`, so the remembered state has already passed through layer
`l`. Memories `M2` through `M12` naturally come from the entry to the next
layer. Only `M13`, after the final layer, needs an explicit writer. Stored K/V
is detached between denoising forwards in the present experiments.

### Data feeding and denoising trajectory

Training uses OpenWebText, the BD3 small architecture, sequence length 1024,
and global batch size 512. DCache-v2 constructs five teacher-forced forwards:

```text
k ~ Uniform(0.025, 0.10)
x ~ Uniform(1.5k, 0.9975 - 1.5k)

t0 = x + 1.5k
t1 = x + 0.5k
t2 = x - 0.5k
t3 = x - 1.5k

full mask -> t0 -> t1 -> t2 -> t3
```

The masks are exact, strictly nested token sets constructed from one random
permutation. Every adjacent transition reveals at least one ground-truth token,
`t0` cannot be fully masked, and `t3` retains at least one masked prediction.
All reveals are teacher forced in this pretraining experiment.

### Training objective

The five diffusion losses use normalized weights:

```text
full : t0 : t1 : t2 : t3 = 0.05 : 0.10 : 0.20 : 1.00 : 0.70
```

V2 adds three mechanisms designed specifically around the V1 failure:

- **Nearby states:** adjacent mask-ratio gaps are only `0.025` to `0.10`, so
  the previous cache is relevant rather than separated by a huge state jump.
- **Source dropout:** after a 1000-step warmup, masked queries use joint
  previous+current attention 75% of the time, cache-only attention 20%, and
  current-only attention 5%. This is disabled in validation and inference.
- **Identity loss:** on 25% of local training batches, correct cache must beat
  a batch-shuffled cache by a `0.05` margin at `t3`; its weight is `0.10`.

The complete implementation and precise equations are in
`DCACHE_V2_IMPLEMENTATION_WORK_NOTE.md`.

## Hypotheses

- **H1 — useful recurrence:** correct previous-step cache lowers held-out NLL
  relative to no cache and zero cache.
- **H2 — identity:** correct cache lowers held-out NLL relative to shuffled
  cache by a material amount, not merely numerical noise.
- **H3 — local transitions:** nearby denoising states make cached content more
  useful than the large jumps used by V1.
- **H4 — forced use:** cache-only source dropout teaches the writer to place
  predictive information in memory, while a small current-only probability
  prevents total dependence on the cache.
- **H5 — adaptive strength:** useful layers increase their DCache gates;
  useless layers should keep their gates small.
- **H6 — inference benefit:** if the cache is genuinely useful, the advantage
  should persist under model-generated rollout states and may permit equal
  quality with fewer denoising steps.

A result in which correct cache beats no cache but is indistinguishable from
shuffled cache falsifies the strong identity hypothesis, even if attention
mass on cached keys is high.

## Experiments

### Experiment 1 — 5k vanilla baseline versus DCache-v1

**Question:** Does the first recurrent-cache design outperform a matched BD3
model after short from-scratch pretraining?

**Setup:** Train the BD3 small model on OpenWebText at length 1024 and global
batch 512 for 5000 optimizer steps. Vanilla used the normal MDLM objective;
DCache-v1 used the initial recurrent pretraining trajectory. Evaluate each
run's EMA checkpoint.

**Results:** Vanilla validation NLL/PPL was `3.9329/51.06`; DCache-v1 was
`4.0802/59.16`.

**Conclusion:** V1 did not outperform vanilla at this budget. Because the
training objectives and validation state construction differed, this result
motivated controlled same-mask evaluation rather than an immediate rejection.

### Experiment 2 — fixed-corruption teacher-forced evaluation

**Question:** At the same exact corrupted sequence, does V1 correct cache beat
the vanilla model and the same DCache model without cache?

**Setup:** Evaluate both 5k EMA checkpoints on the same first 800 prepared
OpenWebText validation documents, exact shared masks, and seed `20260812`, at
mask ratios 5%, 10%, 20%, 30%, 50%, 70%, 90%, and 100%.

**Results:** Correct-cache DCache beat its no-cache ablation at every ratio,
often by a large margin. Nevertheless, vanilla beat correct-cache DCache at
every nontrivial ratio; the DCache-minus-baseline NLL gap was roughly `+0.13`
to `+0.18` through 70% masks.

**Conclusion:** V1 learned a useful computational path but not a better model
than vanilla.

### Experiment 3 — transition and cache-content ablation

**Question:** Is the gain caused by information specific to the previous
document and denoising state?

**Setup:** Use teacher-forced `s -> t` transitions on the same 800 documents
and compare correct, batch-shuffled, zero, and absent previous cache.

**Results:** Removing or zeroing cache caused large losses, but shuffled cache
was only about `0.0009` to `0.0017` NLL worse than correct cache.

**Conclusion:** This is the key V1 failure. The model used the branch or its
generic statistics, but extracted almost no example-specific information.

### Experiment 4 — DCache-v2 short pretraining

**Question:** Do local steps, source dropout, and identity supervision make
the cache content-specific without destabilizing optimization?

**Setup:** Train V2 for 5000 steps on two RTX 3090 GPUs, length 1024, global
batch 512, five-forward teacher-forced trajectory, learned gate, source
dropout, and correct-vs-shuffled identity loss.

**Results:** The run completed at step 5000. Final cache-warmed `t2`
validation NLL/PPL was `3.6067/36.84`, and mean gate strength was `0.1885`.
All losses remained finite and the final EMA checkpoint loaded strictly.

**Conclusion:** V2 is stable at this scale and learned a nonzero recurrent
branch. The controlled evaluation below, rather than this unmatched training
validation number, determines quality.

### Experiment 5 — matched V2 evaluation

**Question:** Does V2 solve the V1 identity failure and improve actual
denoising transitions?

**Setup:** At step 5000, run the existing fixed-corruption and transition
evaluators with identical documents, masks, seed, EMA use, and conditions for
the vanilla and V2 checkpoints. Compare correct, shuffled, zero, and absent
cache, and report paired document-level confidence intervals. Cache-only and
current-only controls remain a follow-up experiment.

**Results:** On 800 documents, fixed-corruption correct-cache V2 beats vanilla
at every tested ratio from 5% through 90%, with paired gains of `0.0221` to
`0.1797` NLL; at 100% masks V2 is worse by `0.0033`. Across six nested
transitions, V2 beats vanilla by `0.1097` to `0.2055` NLL. Correct cache beats
shuffled cache by `0.0216` to `0.0555`, no cache by `0.0284` to `0.0507`, and
zero cache by `0.0452` to `0.0559`; all transition intervals exclude zero.

**Conclusion:** The short V2 run passes both predeclared mechanism criteria:
it is competitive with and better than vanilla under matched inputs, and it
uses document-specific cached information. This supports proceeding to
on-policy evaluation and targeted ablations before a large pretraining run.

## Ideas and Open Questions

- Does V2 retain its identity advantage under held-out global cache shuffling,
  not only the local training permutation?
- Which layers learn the largest gates, and do these layers carry more
  cache-identity signal?
- Does the identity gain concentrate on still-masked, newly revealed, or
  high-confidence next-to-reveal positions?
- Should source dropout be weaker, stronger, or curriculum-controlled after
  inspecting held-out cache-only and current-only behavior?
- Is the identity loss margin `0.05` large enough, and should it operate at
  `t2`, `t3`, or several transitions?
- How much of the V2 gain survives real on-policy generation rather than
  teacher forcing? A later curriculum can mix approximately 85% teacher tokens
  with 15% model-sampled tokens.
- Does allowing cross-step gradients improve memory writing enough to justify
  its much larger VRAM cost?
- Would a fully merged normal/DCache attention design be more efficient after
  the separate design establishes whether the idea works?
- Does DCache improve sequence quality, entropy calibration, or quality per
  denoising forward even when token NLL is unchanged?
- What is the compute-matched comparison? V2 uses five forwards per update,
  so optimizer-step matching alone is not a compute-equivalent experiment.

## Tasks

- [x] Implement shifted per-layer DCache and final-layer writer.
- [x] Implement parameter-free 2D RoPE, learned gates, and inference plumbing.
- [x] Implement the exact five-forward V2 trajectory and weighted objective.
- [x] Implement source dropout and correct-versus-shuffled identity loss.
- [x] Verify the full unit test suite and profile 3090 VRAM.
- [x] Train and evaluate the 5k vanilla and DCache-v1 checkpoints.
- [x] Diagnose the V1 cache-identity failure with controlled ablations.
- [x] Finish the current V2 run to step 5000 and preserve its final EMA
  checkpoint and logs.
- [x] Regenerate the health plots and final metric table at step 5000.
- [x] Run fixed-corruption evaluation on V2 with masks exactly matched to the
  vanilla checkpoint.
- [x] Run V2 transition ablations: correct, shuffled, zero, and absent cache.
- [x] Report paired confidence intervals and per-mask-ratio NLL/PPL/accuracy.
- [x] Implement parameter-identical objective-matched vanilla control B with
  the same normalized five-state token objective and no cache mechanisms.
- [x] Train control B for 5000 updates and run the same controlled evaluation.
- [x] Propagate detached final-state recurrence through the real sampler.
- [x] Implement independent correct/shuffled/absent DCache × final-state
  intervention evaluation.
- [ ] Run the full 800-document final-state intervention and report paired
  confidence intervals.
- [ ] Add cache-only/current-only held-out transition controls and report
  per-layer gate values.
- [ ] Run teacher-forced versus on-policy rollout evaluation and measure
  exposure-gap degradation.
- [ ] Compare generation quality, entropy/calibration, throughput, VRAM, and
  quality versus number of denoising forwards.
- [ ] Ablate local step width, source-dropout probabilities, gate, and identity
  loss before committing to a 100k run.
- [ ] Only launch a longer run if held-out cache identity is real and the
  correct-cache model is competitive with vanilla.

## Meeting Notes

### 2026-08-08 to 2026-08-10 — architecture and first pretraining design

**Discussion:** Distinguished ordinary completed-prefix caching from recurrent
active-block denoising memory; examined attention normalization, positional
identity, separate versus merged attention, and the same-layer staleness
problem.

**Decisions:** Use a separate DCache QKV/output sublayer, joint previous/current
DCache softmax, parameter-free 2D RoPE, shifted `M_(l+1)` reads, and only one
explicit final writer. Train from scratch on the same OpenWebText setup as BD3.

**Next steps:** Build the pretraining pipeline, validate CUDA execution, and run
a short vanilla-versus-DCache comparison.

### 2026-08-11 to 2026-08-12 — V1 evaluation and failure diagnosis

**Discussion:** The first 5k DCache model trailed vanilla. Controlled
evaluation showed a large correct-cache versus no-cache effect but essentially
no correct-cache versus shuffled-cache effect.

**Decisions:** Treat cache identity, not cache attention mass, as the central
mechanism criterion. Avoid drawing conclusions from the raw training objective
alone.

**Next steps:** Make adjacent denoising states closer and explicitly force
content-sensitive cache use.

### 2026-08-12 — DCache-v2 design

**Discussion:** Chose a four-transition local trajectory centered on a
uniformly sampled location, a peaked loss distribution around `t2`, source
dropout, and an identity margin loss.

**Decisions:** Use `k` in `[0.025, 0.10]`, five teacher-forced forwards,
`0.05/0.10/0.20/1.00/0.70` loss weights, 20% cache-only and 5% current-only
dropout after warmup, 25% identity batches, and a learnable gate initialized
at effective strength `0.1`.

**Next steps:** Train V2 for 5000 updates on GPUs 2 and 3, monitor fixed
validation and identity gain, then repeat controlled evaluation.

### 2026-08-15 — live V2 health review

**Discussion:** Raw loss became visually noisier after about step 1900.
Inspection showed that mask-ratio variation explains most of this fluctuation;
fixed validation continued downward and gate growth remained smooth.

**Decisions:** Continue the run. Use fixed validation and held-out paired cache
ablations for decisions, not isolated raw minibatch loss spikes.

**Next steps:** Finish step 5000, regenerate this figure, then execute
Experiment 5 before considering a long run.

### 2026-08-18 — final V2 controlled evaluation

**Discussion:** V2 completed 5000 steps. Fixed-corruption and nested-transition
evaluations used the same 800 documents, exact masks, seed, and EMA policy for
vanilla and V2. Correct, absent, shuffled, and zero cache were compared.

**Decisions:** Treat the correct-versus-shuffled result as evidence that V2
solved V1's identity failure. Separate the incremental cache gain from the
larger gain caused by the complete V2 training recipe, and avoid a
compute-efficiency claim until FLOP-matched/on-policy tests are complete.

**Next steps:** Run source-only controls, per-layer diagnostics, and real
on-policy generation; then decide the scope of a longer run and its ablations.

### 2026-08-19 — objective-matched vanilla control B

**Discussion:** The no-cache and shuffled-cache V2 model remained better than
the original vanilla baseline, creating a confound between useful cache
content and the five-state future-supervision objective.

**Decisions:** Train a parameter-identical vanilla MDLM with full/t0/t1/t2/t3
processed independently and loss
`(0.05/0.10/0.20/1.00/0.70) / 2.05`. Disable the DCache architecture, previous
K/V, source dropout, and identity loss. Match data, seed, optimizer, global
batch, trajectory sampler, validation masks, and 5000 optimizer updates.

**Next steps:** Launch B, verify that `trainer/num_forwards=5` and
`trainer/loss_weight_sum=2.05`, then compare A/B/D using the existing
fixed-corruption and teacher-forced transition evaluations.

## References

- [BD3-LM repository](https://github.com/kuleshov-group/bd3lms)
- [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573)
- [MDLM repository](https://github.com/kuleshov-group/mdlm)
- [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- [ELF repository](https://github.com/lillian039/ELF)
- [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938)
- [MetaState: Persistent Working Memory for Discrete Diffusion Language Models](https://arxiv.org/abs/2603.01331)
- `2503.09573v3.pdf` — local BD3 paper copy
- `FINAL_DENOISING_CACHE_PLAN.md` — final shifted-architecture plan
- `DCACHE_V2_IMPLEMENTATION_WORK_NOTE.md` — exact current implementation and
  launch details
- `DCACHE_RUNBOOK.md` — environment, training, plotting, and evaluation commands
- `outputs/eval-5k-teacher-forced/` — V1 controlled evaluation artifacts
- `outputs/eval-v2-5k-teacher-forced/` — final V2 controlled evaluation
  artifacts, paired confidence intervals, and figures
