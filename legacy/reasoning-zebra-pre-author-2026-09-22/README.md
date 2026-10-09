# Pre-author-release Zebra archive

This directory preserves the repository's earlier Zebra reproduction and
diagnostic work before the authors of *Reasoning with Latent Tokens in
Diffusion Language Models* released their implementation.

Archived on: 2026-09-22

## Contents

- `code/`: dedicated Zebra modules, launchers, reports, and tests.
- `artifacts/outputs/`: completed or partial Zebra runs. Mixed Sudoku/Zebra
  queues were split: only each `zebra-benchmark` subtree and Zebra-named
  console logs were moved here.
- `artifacts/results/`: Zebra tables, figures, and audits.
- `artifacts/cache/`: derived Zebra datasets produced by our older pipeline.
- `artifacts/logs/`: Zebra-named launch logs.

`artifacts/outputs/author-zebra-aborted-causal-output-empty` is an empty
directory from the corrected pre-launch mapping attempt. No SIDM/SCDM
optimizer step was run.

The raw author-provided `zebra-train-data.pkl` and `zebra-test-data.pkl` files
were deliberately **not** moved. They are inputs to the new author-faithful
reproduction and remain under `.cache/downloads/reasoning-puzzles/`.

## Compatibility files retained in the active tree

`reasoning/zebra_official.py` and `reasoning/zebra_continuation.py` remain in
the active tree because shared Sudoku-capable queue/data modules import them.
They are legacy compatibility dependencies; the new Zebra reproduction does
not use them. Zebra branches embedded in shared modules were also left in
place so this cleanup does not alter any Sudoku behavior.

The replacement Zebra path is isolated under
`third_party/reasoning_with_latent_tokens/` and is launched only through
`scripts/reasoning/run_author_zebra_3ep_2x3090.sh`. The wrapper selects the
authors' `diffu-full-mini-zebra-tat` MDM condition; it does not select their
SIDM/SCDM causal-output condition.
