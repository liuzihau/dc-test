# BD3 pretraining reference

These files are copied unchanged from this repository's last upstream commit
before the DCache prototype: `1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7`.
Upstream: https://github.com/kuleshov-group/bd3lms (Apache-2.0; LICENSE retained).

The active adapter is `owt/`. It keeps the original small-model architecture,
MDLM corruption/ELBO, optimizer, LR schedule, tokenizer and packed data format.
Operational changes (local logs, fixed validation randomness, data-cursor
restoration and checkpoint retention) and optional NP live outside this snapshot.

The recipe uses PyTorch SDPA, an existing upstream attention backend, and BF16
mixed precision for the local 3090s. The 5,000-step stop is a pilot budget, not
the full paper pretraining budget. Monitoring uses 1,024 fixed validation rows.
