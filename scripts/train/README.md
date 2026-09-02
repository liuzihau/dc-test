# Training Entry Points

## Canonical experiments

| Experiment | Launcher |
|---|---|
| BD3/MDLM vanilla | `train_owt_mdlm_pretrain_5k_2x3090.sh` |
| Objective-aligned no-memory | `train_owt_mdlm_objective_matched_5k.sh` |
| DCache-v2 | `train_owt_dcache_v2_pretrain_5k_2x3090.sh` |
| DCache + final state | `train_owt_dcache_final_state_5k_2x3090.sh` |

The canonical paths and comparison metrics live in
`../../experiments/canonical_runs.json`.

Preferred interface for new runs:

```bash
bash scripts/train/run_canonical_trial.sh \
  {vanilla|objective|dcache-v2|final-state} outputs/NEW_RUN_NAME
```

It normalizes the GPU defaults and retains only the latest three periodic
500-step checkpoints. Existing completed runs keep their historical files so
published local evidence is not silently deleted.

## Shared infrastructure

- `train_owt_mdlm_pretrain_100k.sh`: common vanilla launcher.
- `train_owt_dcache_pretrain_100k.sh`: common DCache launcher.
- `continue_owt_5k_to_6k_4x3090.sh`: reproduces the completed V2/vanilla
  checkpoint-extension study.
- `run_owt_dcache_then_baseline_5k_to_6k_4x3090.sh`: sequential wrapper for
  that extension study.

## Historical or deferred

The 4090 comparison launchers, 50k prototype launchers, DCache-v1 launchers,
and full tentative/confidence `train_owt_dcachehooping_5k_2x3090.sh` are not
canonical experiments. They remain as implementation history and must use a
distinct output directory if deliberately rerun.
