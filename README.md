# DCache research — active benchmark implementations

The benchmark policy is fixed: **Sudoku and Zebra use the verified author pipeline**
vendored under `third_party/reasoning_with_latent_tokens/`; **OpenWebText uses the
BD3 pretraining pipeline**. Results from these families must not be mixed as if
they shared one data objective or evaluation protocol.

The current Sudoku/Zebra code supports four separate variants: `mdm`, `mdm_np`,
`trajectory_attention`, and `trajectory_recurrent`. The trajectory has three
nested teacher states and normalized weights `(0.25, 1, 0.25) / 1.5`. The two
trajectory arms have matched extra attention; RM adds shifted previous-pass KV
and one-transition gradient credit. Final-hidden feedback and identity loss are
disabled. The NP/RM merged trial is deferred.

For a new four-GPU server, use [the setup and run commands](puzzle_recurrence/USAGE.txt).
`python -m puzzle_recurrence.run_pair --tasks sudoku zebra` runs attention on GPUs
0,1 and recurrence on GPUs 2,3. It evaluates 1,280 completed puzzles every three
epochs and at the final epoch, then moves to the next dataset. Use
`python -m puzzle_recurrence.monitor --tasks sudoku zebra` to refresh live
performance figures from the saved metrics. Existing outputs
are preserved; resume explicitly with `--resume`. Dataset files and checkpoints
are not stored in Git. The OWT adapters and the baseline/NP puzzle adapters are
also included, with their original vendored source and licenses.

Older trajectory, attention, and DCache implementations are retained under
`legacy/`. The sections below also document the earlier NP experiments.

## OpenWebText: BD3 MDM vs MDM + NP (5,000 updates)

The active OWT implementation is `owt/`, using an unchanged BD3 source snapshot
at `third_party/bd3/` (commit `1c3e8f4`, immediately before the DCache prototype).
Both models start from scratch with the same backbone initialization and seed:
12 layers, width 768, 12 heads, length 1,024, GPT-2 tokenization, and the prepared
8,713,822-row OWT training cache. Corruption and main ELBO call the upstream code
directly: antithetic continuous `t` in `[0.001,1]`, independent Bernoulli masking,
the upstream first-position exemption, and token loss weighted by `1/t`.

The NP arm uses two independent linear vocabulary heads on the feature entering
the main LM projection, with coefficients 0.25 per direction. Only masked target
positions contribute; sources may be clean or masked. EOS/special-token boundaries
and sequence edges are excluded. The main ELBO denominator and timestep weighting
are also used by NP, with no division by 1.5. NP projections are chunked and
recomputed during backward to bound memory. TT, EA, and RM are disabled.

The queue runs **MDM to 5,000 → MDM+NP to 5,000**, using physical GPUs **2,3**.
Global batch 512 = 2 GPUs × microbatch 8 × accumulation 32. AdamW uses LR 3e-4,
2,500-step linear warmup then constant LR, EMA 0.9999, clip norm 1.0, BF16 mixed
precision, and the upstream SDPA implementation. Model parameters are 169.6M
(MDM) and 246.9M (NP includes 77.3M training-only auxiliary parameters).

Validation and full-state checkpoints occur every **500 optimizer updates**,
equivalent to 16,000 training microbatches. Validation reports the common main-head
EMA ELBO over 1,024 fixed held-out rows, using identical corruption for both arms
and restoring the training RNG afterward. This is monitoring-subset validation;
it is not the full OWT validation set. Keep the latest three numbered checkpoints
plus a `last.ckpt` symlink. Restarting the queue resumes the full optimizer,
scheduler, EMA and completed data cursor; it does not promise bitwise replay of
dropout/corruption across a process restart.

```bash
bash owt/run.sh check       # baseline/NP numerical and memory checks
bash owt/run.sh smoke       # two-GPU train/validate/save/resume checks
bash owt/run.sh launch      # launch both runs in tmux
TMUX_TMPDIR="$PWD/.tmux" tmux attach -t owt-mdm-np-5k
bash owt/run.sh status      # latest updates/losses and refreshed plot
bash owt/run.sh plot --smooth 128
```

Outputs: `outputs/owt/mdm-np-5k/{mdm,mdm_np}/local_metrics/{train,validation}.csv`,
`checkpoints/`, and `train.log`; comparison plot:
`outputs/owt/mdm-np-5k/loss_vs_step.png`. Runtime caches and tmux sockets stay
inside this repository. No external tracking account or download is required.

### Zero-initialized NP follow-up

`mdm_np_zero_init` repeats the 5,000-update NP experiment from scratch with the
same seed, shared backbone, data order, corruption, 0.25 directional weights,
optimizer, EMA, validation subset, and checkpoint policy. Its only recipe change
is `mechanisms.np.initialization: zero`: both auxiliary vocabulary projections'
weights and biases start at zero, matching BD3's main output projection. BD3's
existing zero-initialized AdaLN residual gates are retained. Other backbone
weights retain their original initialization.

```bash
bash owt/run.sh launch-zero-init
TMUX_TMPDIR="$PWD/.tmux" tmux attach -t owt-np-zero-init-5k
bash owt/run.sh status
```

The follower waits for the live original queue's file lock without allocating a
CUDA context. It verifies that both original arms successfully reached 5,000
updates before starting on physical GPUs 2 and 3. An incomplete or failed
original run prevents launch. Outputs are isolated under
`outputs/owt/mdm-np-5k/mdm_np_zero_init/`; the waiting/running state is recorded
in `zero_init_queue.json`, and the controller log is
`logs/owt-np-zero-init-5k.log`. The comparison plot automatically includes the
zero-init arm when its CSVs appear. Relaunching this follower resumes only its
own checkpoint; completed 5,000-update runs are skipped.

### Ongoing research cycle

The CPU-only controller `owt-research-monitor` checks local progress every
minute and refreshes the living LaTeX/PDF after validation and completion
events. The main document contains the latest theory; the appendix preserves
old theory versions and an append-only activity log. Files are under
`outputs/research-notes/`, including
`neighbor-prediction-research-assessment.{tex,pdf}`, `current_theory.tex`,
`research_events.jsonl`, and `research_monitor.json`.

The conditional controller `owt-np-low-weight-5k` waits for successful 5,000-step
completion of MDM, random-init NP, and zero-init NP. If the final zero-init NP
validation ELBO exceeds MDM by at least 0.03 nats/token, it starts a fresh NP
trial with zero output initialization and directional weights 0.05 instead
of 0.25. Every other setting stays fixed. Otherwise it records `not_launched`
for scientific review. The threshold is a practical scheduling rule, not a
significance test. It uses the same queue lock and only physical GPUs 2 and 3.

```bash
bash owt/run.sh launch-research-monitor
bash owt/run.sh launch-low-weight
bash owt/run.sh status
```

These launch commands reject duplicate controllers. Monitor log:
`logs/owt-research-monitor.log`; next-trial state:
`outputs/owt/mdm-np-5k/low_weight_queue.json`. Future theory revisions should use
`python -m owt.research revise --source NEW_THEORY.tex --rationale 'Evidence and decision'`,
which archives the preceding theory and its observation snapshot before replacement.

## Sudoku: author-aligned MDM vs MDM + NP

Sudoku uses the author `sudoku-puzzle` data path and mini architecture: 1,804,463
training puzzles, 100,000 test puzzles, length 192, vocabulary 14, six layers,
width 512, eight heads, global batch 512, and 20 epochs (3,525 updates per epoch;
70,500 total). The main ELBO is restricted by the author solution loss mask.
Training keeps canonical token order; generation uses the author sampler with
192 denoising steps, candidate window 8, EMA, and 1,280 public test puzzles.
Validation selects an additional best-`val/nll` checkpoint.

```bash
bash sudoku/run.sh check
bash sudoku/run.sh smoke --root "$PWD/.cache/runtime/sudoku-modular/smoke" --microbatch 128
bash sudoku/run.sh launch --microbatch 128

TMUX_TMPDIR="$PWD/.tmux" tmux attach -t sudoku-mdm-np-20ep
tail -f logs/sudoku-mdm-np-20ep.log
bash sudoku/run.sh status
bash sudoku/run.sh plot
```

The queue alternates MDM and MDM+NP in three-epoch blocks, evaluates epochs
3, 6, 9, 12, 15, 18, and 20, and finally evaluates each run’s best-validation
checkpoint. NP predicts masked solution tokens from their immediate left and
right neighbors with weights 0.25 each; source positions may be clean or masked,
and pairs cannot cross `<SEP>`, special tokens, padding, or sequence edges. The
NP heads are training-only and do not alter generation.

## Zebra: run on GPUs 2 and 3

From this repository, with the existing `dcache` environment and prepared author
Zebra data cache:

```bash
bash zebra/run.sh check
bash zebra/run.sh smoke --root "$PWD/.cache/runtime/zebra-modular/smoke" --microbatch 128
bash zebra/run.sh launch --microbatch 128
```

The launcher defaults to `/home/tliu0205/miniconda3/envs/dcache/bin/python`.
Override it with `DCACHE_PYTHON` when needed. Runtime caches and tmux sockets are
inside this repository rather than `/tmp`.

The queue alternates **MDM epochs 1–3 → generation → MDM+NP epochs 1–3 →
generation → MDM epochs 4–6 → ... → both epoch 40**. The final chunk is one
epoch. Each variant has its own model, optimizer, EMA, scheduler, and checkpoint.
Both start from the same backbone initialization, seed and raw dataset. NP heads
are initialized in a separate RNG context. Global batch is 512 (2 GPUs × 128
examples × 2 accumulation steps), or 2 GPUs × 256 with accumulation 1 if selected
and verified. One raw data epoch is 2,930 optimizer updates; 40 epochs is 117,200.

The last partial batch and DDP padding follow Lightning/author behavior.
Resumption preserves model/optimizer/EMA/scheduler/loop state; it does not claim
bitwise-identical random streams after restarting processes. A run contract
rejects changed recipes or batch settings within an existing output directory.

```bash
TMUX_TMPDIR="$PWD/.tmux" tmux attach -t zebra-mdm-np-40ep
tail -f logs/zebra-mdm-np-40ep.log
# Detailed progress for the first segment:
tail -f outputs/zebra/mdm-np-40ep/mdm/train-to-8790.log
# Refresh plots while training:
bash zebra/run.sh plot
# Show active model/steps/latest metrics AND refresh all available plots:
bash zebra/run.sh status
```

Results are under `outputs/zebra/mdm-np-40ep/`:

- `mdm/` and `mdm_np/`: independent checkpoints, resolved configurations and logs.
- `<variant>/local_metrics/train.csv`: total objective, main ELBO, NP components, LR.
- `<variant>/local_metrics/validation.csv`: common author main-head validation ELBO.
- `<variant>/generation/epoch-*/`: author sample outputs and completion records.
- `generation_history.csv`, `accuracy_vs_epoch.png`, `row_cell_accuracy_vs_epoch.png`,
  and `loss_vs_step.png`.

The loss figure starts at optimizer step 5,000 and shows the total training
objective, main training ELBO, and main validation ELBO. Training curves use a
128-update rolling mean; validation points are unsmoothed. Exact-puzzle, row,
and cell accuracy plots use a y-axis 10% above the largest available value
(capped at 100%). The total NP objective includes extra losses and should
not be compared directly to MDM's total as a measure of prediction quality.
Generation points first appear after each model reaches three epochs.

The completed batch-size benchmark measured 256 as only 0.1% faster for MDM and
2.2% faster for MDM+NP, so the run remains at microbatch 128 and global batch
512. The queued policy hook was disabled afterward. Its read-only result remains
in `microbatch_benchmark.json`; `microbatch_policy.disabled.json` records the
historical request.

Checkpointing keeps the latest three numbered snapshots plus a `last.ckpt` link.
Restart the same queue command to continue; successful completed generation
stages are skipped. Failed stages stop the queue visibly.

## Configurable mechanisms and objective

Edit `zebra/configs/mdm_np.yaml` **before a fresh run**, or use a new output root
with `--root outputs/zebra/my-new-trial` after changing a recipe:

```yaml
mechanisms:
  tt: {enabled: false}
  ea: {enabled: false}
  rm: {enabled: false}
  np:
    enabled: true
    offsets: [-1, 1]        # one independent head per offset
    weights: [0.25, 0.25]  # one coefficient per head
    hidden_layers: 0       # 0 = linear projection; >0 adds hidden Linear+GELU layers
    boundary_tokens: [CLUE_END, ANSWER]
objective:
  kind: elbo               # optional masked_ce for separately named trials
  current_weight: 1.0
```

For the first trial, target position j contributes its current-token loss plus
0.25 times prediction from j−1 and 0.25 times prediction from j+1 when eligible.
All contributions use the **same author timestep weighting and denominator**;
there is **no division by 1.5**. NP requires the **target** to be masked; the
source may be clean or masked. Pairs never wrap, cross special/boundary tokens,
or supervise padding. Ground-truth neighbors are loss targets only.

NP reads the same normalized hidden feature as the main LM projection. Its
heads are included in optimization and EMA but are not computed during
generation. Validation always reports only the common current-token ELBO.

Training preserves original token order and bypasses both random shuffling and
clean/masked grouping. Generation retains the author's internal permutation
bookkeeping and original position IDs, with **unrestricted top-prob candidates**,
384 steps, EMA and stochastic token sampling. The same first 1,280 public test
rows are used each time. Full test-file conditional-loss validation runs at data
epoch boundaries. The author `train_on_all_tokens=true` training convention,
including main-loss handling of clues and padding, is preserved.

The verified older author model achieved **95.703%** on those 1,280 puzzles with
unrestricted candidates. Public raw Zebra train/test files overlap; this score
and this first comparison are author-protocol results, not disjoint-test claims.

## Legacy and progress

`legacy/pre-zebra-modular-2026-09-23/` contains a hash-verified snapshot of 285
older source files, including the previous README. Verified original source
directories have been moved into its `original-paths/` subdirectory; the initial
snapshot is also retained. No datasets, checkpoints or results were deleted.
New code imports neither old `reasoning/` nor BD3 root modules. Historical scripts
retain their original path assumptions and are reference code until restored or
ported; use `zebra/run.sh` for the active experiment.

See `zebra/PLAN.md` for the implementation checklist and
`RESEARCH_EXPERIMENT_LOG.md` for research history. The paper remains in
`iclr2027-paper/`. Upstream licenses are retained.
