# Evaluation Entry Points

## Canonical evidence

- `eval_fixed_corruption.py`: identical corrupted inputs across models.
- `eval_teacher_forced_transitions.py`: matched teacher-forced denoising
  transitions with correct, absent, shuffled, and zero cache.
- `eval_same_state_recurrence.py`: same-input recurrence intervention.
- `eval_checkpoint_validation.sh`: aligned unshuffled validation for any of
  the four canonical training variants.
- `eval_final_state_interventions.py`: independent correct/shuffled/absent
  DCache and final-state causal interventions.
- `plot_three_way_teacher_forced.py`: vanilla, objective-aligned, and
  DCache-v2 comparison.
- `plot_dcache_checkpoint_identity.py`: correct-versus-shuffled cache trend.
- `plot_focused_checkpoint_transitions.py`: late-denoising checkpoint trend.

The retained raw results are registered in
`../../experiments/canonical_runs.json` and curated by:

```bash
python scripts/results/refresh_canonical_results.py
```

For a completed final-state checkpoint:

```bash
bash scripts/eval/eval_final_state_interventions.sh \
  CHECKPOINT OUTPUT_DIR
```

## Historical wrapper

`run_5k_teacher_forced_evals.sh` was written for the rejected DCache-v1
experiment and its default paths are historical. Do not use it without
explicitly overriding both checkpoints, the gate setting, and the output
directory. New experiment-specific launchers should read paths from the
canonical manifest.
