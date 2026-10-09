# Zebra modular comparison — implementation status

2026-09-23. First trial: independent fresh MDM and MDM+NP, alternated in
three-epoch chunks up to 40 epochs (last chunk is one epoch).

- [x] Author backbone/data/objective/schedule retained in an isolated adapter.
- [x] Canonical training/validation order: no shuffle and no clean/masked grouping.
- [x] Configurable NP offsets, weights, projection depth and objective.
- [x] NP only supervises masked targets; clean or masked source is allowed.
- [x] Special tokens, edges and serialized boundaries excluded from NP pairs.
- [x] Same backbone initialization, no extra backbone forward, NP in optimizer/EMA.
- [x] Common main-head ELBO validation; separate objective and NP metric fields.
- [x] Alternating resumable queue and unrestricted author evaluation.
- [x] Six CPU tests and GPU baseline/gradient/EMA/validation equivalence checks.
- [x] Two-GPU forward/backward/checkpoint/generation smoke tests for both models.
      Global batch512, microbatch128 per GPU; both completed2 updates and
      author generation on4 test examples. NP loss arithmetic verified in CSV.
- [x] Full-state continuation smoke test: NP resumed from update 2 to 3;
      optimizer step, scheduler last_epoch and EMA num_updates all verified as 3.
- [x] Launch the 40-epoch queue after successful checks (2026-09-24).
      GPUs 2,3; tmux `zebra-mdm-np-40ep`; microbatch 128 per GPU.
      Output root: `outputs/zebra/mdm-np-40ep`.
- [x] Hash-verified non-destructive legacy snapshot (285 source files).
- [x] Relocate original legacy source paths under the archive's `original-paths/`.
      Read-only process checks confirmed every old training pane was dead and the
      remaining shell had no children and was idle in the untouched paper folder.
      Approval review then permitted the verified move. No source/data was deleted.

TT, EA and RM have separate configuration namespaces, but their new Zebra
ports are pending. Enabling one raises an explicit error; old implementations
are preserved in the legacy snapshot. All six mechanisms are not yet ported.

Loss: author current-token ELBO + 0.25 times each neighbor ELBO contribution.
All three use the same timestep multiplier and baseline token denominator.
There is no division by 1.5. Optional `masked_ce` changes both current and
auxiliary weighting consistently and requires a new run directory. Evaluation
always reports the common author ELBO.

Author `train_on_all_tokens=true` remains enabled. This includes clue tokens and
padding in the main training objective; NP explicitly excludes special-token
pairs. Validation/generation condition on the author's problem/solution mask.

Training order is canonical; generation retains the author's internal packing
and original position IDs, needed by its adaptive top-prob sampler. Candidate
window min/max are both 0 (unrestricted); token sampling remains stochastic.

Data: same 1,499,933 raw training rows, same first 1,280 public test rows for
generation. Public train/test overlap remains; this is an author-protocol
comparison, not a disjoint generalization claim.

**FINISHED: implementation and smoke/resume verification.** The long experiment
is running, not finished. `bash zebra/run.sh status` reports progress and refreshes
loss/accuracy plots; generation points appear after each three-epoch chunk.

## Scheduled microbatch trial — 2026-09-24

- [x] Current epoch-1–3 MDM+NP training stays at microbatch 128.
- [x] Policy armed after BOTH epoch-3 generation stages complete, before MDM 4–6.
- [x] Automatic two-GPU scratch benchmark of 128 versus 256 for MDM AND MDM+NP.
      This does not update training checkpoints or advance the training sampler.
      Select 256 only if both improve by >3% and retain 2 GiB VRAM headroom;
      failure/OOM/timeout keeps 128. Global batch stays 512; accumulation 2->1.
- [x] Epoch-boundary checkpoint microbatch counters converted; optimizer,
      scheduler, EMA and sampler consumed-row offset are preserved. Originals
      are not overwritten. Generation skips training-counter migration.
- [x] Validation batch remains 128. Twelve CPU tests pass (NP and batch policy).
- [x] Boundary benchmark finished: 256 speedup was 1.001x for MDM and 1.022x
      for MDM+NP, below the 1.03x threshold. Training correctly remains at 128.
- [x] Completed policy hook disabled at user request; report retained.

Historical policy: `outputs/zebra/mdm-np-40ep/microbatch_policy.disabled.json`.
The initial contract still records 128; the benchmark decision explicitly
records any later transition. Keep using the original queue command. The
entrypoint checks the policy at each new training segment, including subprocesses
from the already-running queue, so no process interruption is required.
