# Canonical Repository Map

The upstream BD3 code remains intact because all four experiments share its
model, data loader, Lightning loop, and checkpoint format. New work should use
the small canonical surface below; other launchers are retained only for
reproducibility of completed historical trials.

## Core implementation

- `diffusion.py`: vanilla objective, five-state objective, DCache/final-state
  training, validation, and generation recurrence.
- `models/dit.py`: normal BD3 blocks, separate DCache attention, shifted
  memory writer, 2D RoPE, gate, and final-state input fusion.
- `configs/config.yaml`: feature switches and objective hyperparameters.
- `configs/callbacks/`: latest-three periodic checkpoint policy.

## Commands to use

- `scripts/train/run_canonical_trial.sh`: train one of the four retained
  variants with consistent GPU, validation, and checkpoint defaults.
- `scripts/eval/eval_checkpoint_validation.sh`: reproduce the unshuffled
  training-objective validation subset.
- `scripts/eval/eval_final_state_interventions.sh`: test the two recurrent
  sources independently.
- `scripts/results/refresh_canonical_results.py`: rebuild compact figures and
  tables from registered raw output.

## Scientific source of truth

- `experiments/canonical_runs.json`: registered runs and evaluations.
- `results/README.md`: comparison rules.
- `DCACHE_RESEARCH_SUMMARY.md`: current claims, caveats, and open tasks.
- `DCACHE_RUNBOOK.md`: exact operational instructions.

## Generated and historical data

`outputs/`, `results/generated/`, `archive/`, `.cache/`, `logs/`, and `.tmp/`
are intentionally outside Git. Existing old checkpoints are preserved because
some support prior analyses; the latest-three rule applies prospectively to
new canonical runs. Failed/superseded results must not be used for claims.
