# Canonical Research Results

This directory is the human-facing result index. Raw Lightning logs,
checkpoints, and batch-level evaluation records remain under `outputs/`; the
small, interpretable figures and tables are regenerated under
`results/generated/`.

The project currently recognizes exactly four training experiments:

| ID | Purpose | Raw run |
|---|---|---|
| `bd3_vanilla` | BD3/MDLM baseline | `outputs/owt-mdlm-pretrain-5k-2x3090` |
| `objective_aligned` | Same five-state objective without recurrent memory | `outputs/owt-mdlm-objective-matched-5k` |
| `dcache_v2` | Layerwise recurrent DCache | `outputs/owt-dcache-v2-pretrain-5k-2x3090` |
| `dcache_final_state` | DCache plus detached final-layer state | `outputs/owt-dcache-final-state-pretrain-5k-2x3090` |

The source of truth is `experiments/canonical_runs.json`. Refresh every
canonical figure and the training-status table with:

```bash
conda activate dcache
python scripts/results/refresh_canonical_results.py
```

Generated content is organized as:

```text
results/generated/
├── figures/
│   ├── training/       # four-way training and validation health
│   └── mechanism/      # DCache-v2 identity and recurrence evidence
└── tables/
    ├── training/       # latest matched metrics
    └── mechanism/      # compact evaluation summaries
```

## Comparison rules

- Truncate the primary budget comparison at 5,000 optimizer steps.
- Use vanilla `trainer/loss` and `val/nll`.
- Use `trainer/loss_t2` and `val/loss_t2` for the three five-state methods.
- Treat training-loop validation as a health metric. Use the controlled
  teacher-forced evaluations for scientific claims.
- Do not compare raw total objectives when auxiliary loss terms differ.
- Do not present archived smoke tests, DCache-v1, failed full-DCachehooping
  runs, or hardware-transfer checks as model-quality results.

## Canonical mechanism evidence

Three evaluation families remain relevant:

1. the 5k three-way teacher-forced comparison;
2. DCache-v2 correct-versus-shuffled cache identity across checkpoints;
3. same-state recurrence with correct, shuffled, and absent cache.

The implemented fourth family, independent DCache × final-state intervention,
becomes canonical only after its full 800-document run is complete and is then
registered in `experiments/canonical_runs.json`.

The archival manifest is `results/archive_manifest.md`. Archived files are
recoverable and are not used by canonical plotting.
