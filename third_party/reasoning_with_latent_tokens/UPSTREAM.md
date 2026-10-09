# Upstream provenance

This directory is an unmodified vendored snapshot of the anonymous author
release for *Reasoning with Latent Tokens in Diffusion Language Models*.

- Source: <https://anonymous.4open.science/r/anon-code-release-8493/>
- Downloaded: 2026-09-22
- Downloaded ZIP SHA-256:
  `23d324205bd8f8420501db9887fb1826d7c916f31f7924896baba59a5314f0d7`
- License: Apache License 2.0, retained verbatim in `LICENSE`.
- Snapshot contents: 176 files from the release archive.

No research-code file in this directory was modified after extraction. Local
hardware adaptation, path selection, checks, and launch management live in
`../../scripts/reasoning/run_author_zebra_3ep_2x3090.sh`.

The target author configuration is the standard MDM run named
`diffu-full-mini-zebra-tat`:

- `algo=difflm`
- full bidirectional diffusion attention (the MDM condition, not SIDM/SCDM)
- the released `difflm` ordering defaults (clean and masked tokens are
  effectively shuffled while retaining original positional embeddings)
- `model=mini` (width 512, 6 blocks, 8 heads)
- `training.train_on_all_tokens=True`
- raw released Zebra train/test pickle files
- sequence length 384
- global batch size 512
- AdamW at `3e-4`, constant schedule after 2,500 warmup steps

The local wrapper maps the author's global batch of 512 onto two 24 GiB RTX
3090s as 256 examples per rank with no gradient accumulation. The global
batch, optimizer-step semantics, data, model, objective, and schedule remain
the author's settings. Because the requested three-epoch run ends at update
8,790—before the authors' 10,000-update checkpoint interval—the wrapper saves
at each 2,930-update epoch boundary. This changes only checkpoint I/O, not
optimization or model state.

The same wrapper adds a callback from the outer repository (the vendored tree
remains unchanged) that writes synchronized per-update training loss/LR and
per-epoch held-out NLL/PPL/BPD to local CSV files. Held-out validation runs at
the same 2,930-update epoch boundaries. The callback snapshots and restores
Python, NumPy, CPU Torch, and CUDA RNG states around validation so these extra
monitoring passes do not advance the later training-corruption RNG streams.
Task-generation evaluation is kept separate from these loss-only validation
passes.

On the local torch 2.7.1 + RTX 3090 runtime, the release's forced Inductor
CUDA-graph capture fails during long validation in the CUDA graph allocator.
The local entrypoint therefore retains Inductor-compiled flex attention while
disabling only CUDA-graph capture; the vendored source remains unchanged. A
full 196-batch validation passed with this shim. A resumed run is bounded by
the original final optimizer step (8,790), because Lightning may advance an
empty epoch counter when restoring a checkpoint saved immediately before
epoch-end validation.
