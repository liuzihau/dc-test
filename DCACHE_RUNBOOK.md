# Dcache environment and experiment runbook

## Environment

The tested environment is named `dcache`, uses Python 3.9, and follows the
upstream BD3 dependency pins (PyTorch 2.7.1 with CUDA 12.6 wheels). FlashAttention
is not required for BD3 training; FlexAttention and SDPA are provided by PyTorch.

Create it from scratch:

```bash
cd /home/tliu0205/dc-test
conda env create -f environment.yml
conda activate dcache
python -m pip check
python -m pytest -q
```

The environment already created on this server is:

```text
/home/tliu0205/miniconda3/envs/dcache
```

Activate it with:

```bash
source /home/tliu0205/miniconda3/etc/profile.d/conda.sh
conda activate dcache
```

## Focused tests for the shifted pretraining design

```bash
python -m pytest -q \
  tests/test_step_memory.py \
  tests/test_rollout_curriculum.py \
  tests/test_dcache_pretrain.py
```

These cover DCache-before-normal ordering, `[M2...M13]` cache mapping, 2D RoPE,
the final writer gradient under truncated recurrence, nested `s -> t` masks,
and the complete three-pass loss.

## 100k MDLM pretraining comparison

The new authoritative comparison is full-sequence pretraining, not the older
block-16 trial. Both jobs train from scratch with sequence/block size 1024,
global batch 512, SDPA, and 100,000 optimizer updates.

Vanilla MDLM (the foundation pretraining stage used by BD3):

```bash
bash scripts/train/train_owt_mdlm_pretrain_100k.sh
```

Shifted DCache with three teacher-forced forwards `100% -> s -> t`:

```bash
bash scripts/train/train_owt_dcache_pretrain_100k.sh
```

The scripts use `python` from the active environment. For a detached shell or
job launcher, pin the interpreter explicitly:

```bash
export DCACHE_PYTHON=/home/tliu0205/miniconda3/envs/dcache/bin/python
```

To run them sequentially on the same four GPUs:

```bash
bash scripts/train/run_owt_pretrain_comparison_100k.sh
```

The tested safe default is per-GPU microbatch 4. A five-update full three-pass
profile reserved 21.4 GiB on one RTX 3090. Microbatch 5 reached 22.9 GiB in a
one-update profile and is not the DDP default because it leaves little
operational headroom. Override when a smaller batch is needed:

```bash
DCACHE_MICRO_BATCH=1 \
  bash scripts/train/train_owt_dcache_pretrain_100k.sh trainer.max_steps=10
```

Outputs and resumable checkpoints are written under:

```text
outputs/owt-mdlm-pretrain-100k/
outputs/owt-dcache-pretrain-100k/
```

CSV loss curves are kept inside each stable run directory:

```text
outputs/owt-mdlm-pretrain-100k/run/lightning_logs/version_*/metrics.csv
outputs/owt-dcache-pretrain-100k/run/lightning_logs/version_*/metrics.csv
```

Compare vanilla `trainer/loss` with recurrent `trainer/loss_s`; also inspect
recurrent `loss_full`, `loss_t`, and the normalized total `trainer/loss`.

The recurrent run executes three model forwards per training example and has
extra DCache parameters. A 100k-vs-100k result is matched by updates and tokens,
not wall time or FLOPs.

Measured on this server:

```text
model                         parameters    4-GPU update time
vanilla MDLM                  169,627,218   about 15 seconds
shifted three-pass DCache     199,128,402   about 33 seconds
```

With global batch 512 and microbatch 4, Lightning accumulates 32 batches per
optimizer update. At the measured health-run rate, 100k updates project to
roughly 17 continuous days for vanilla and 38 days for DCache, before full
validation overhead or interruptions. The 100k target is resumable, but use
early checkpoints and loss curves rather than waiting for completion before
checking whether the hypothesis is learning.

## First comparison: 5k steps across two servers

The matched comparison uses two RTX 3090s for vanilla MDLM and two RTX 4090s
for DCache. Both jobs use seed 1, global batch 512, per-GPU micro-batch 4,
sequence length 1024, the same OpenWebText split, and exactly 5,000 optimizer
updates. Different GPU speed changes wall time, not the amount of training.

On this two-3090 server, run the vanilla control:

```bash
source /home/tliu0205/miniconda3/etc/profile.d/conda.sh
conda activate dcache
bash scripts/train/train_owt_mdlm_pretrain_5k_2x3090.sh
```

On the two-4090 server, install/check the environment and run DCache:

```bash
cd /path/to/dc-test
bash scripts/setup_dcache_environment.sh
eval "$(conda shell.bash hook)"
conda activate dcache
bash scripts/train/train_owt_dcache_pretrain_5k_2x4090.sh
```

The current implementation is not yet committed, so `git clone` or `git pull`
alone will not reproduce it on the second server. Copy the working tree first
(replace the host and destination):

```bash
rsync -a --exclude .git --exclude outputs --exclude .cache \
  /home/tliu0205/dc-test/ USER@GPU4090_HOST:/path/to/dc-test/
```

If the machines do not share the prepared data cache, copy it separately:

```bash
rsync -a --info=progress2 /home/tliu0205/dc-test/.cache/huggingface/ \
  USER@GPU4090_HOST:/path/to/dc-test/.cache/huggingface/
```

The setup script creates or updates the `dcache` environment, checks the pinned
dependencies, confirms that two CUDA GPUs and BF16 are available, and reports
whether the prepared OpenWebText cache exists. It does not hide a missing data
cache: the first training run otherwise downloads and preprocesses roughly
243 GiB on the current installation. Copying `.cache/huggingface` to the second
server, or setting `DCACHE_DATA_DIR` to an existing shared copy, avoids that
large preparation step.

Each run validates on 100 fixed batches every 500 optimizer steps. The RNG is
isolated and deterministically seeded, so the vanilla state and DCache `s`
state use matched validation corruptions. In both CSV files, `val/nll` is the
primary comparable validation metric. For training, compare vanilla
`trainer/loss` with DCache `trainer/loss_s`; DCache `trainer/loss` is the full
three-pass objective and is plotted only as extra context.

Outputs are stable and resumable:

```text
outputs/owt-mdlm-pretrain-5k-2x3090/
outputs/owt-dcache-pretrain-5k-2x4090/
```

To compare the separate-server logs, first copy the small DCache output log
directory to the vanilla server. For example:

```bash
mkdir -p outputs/remote-dcache-logs
scp -r USER@GPU4090_HOST:/path/to/dc-test/outputs/owt-dcache-pretrain-5k-2x4090/run/lightning_logs \
  outputs/remote-dcache-logs/
python scripts/plot_pretrain_losses.py \
  --vanilla outputs/owt-mdlm-pretrain-5k-2x3090 \
  --dcache outputs/remote-dcache-logs \
  --output outputs/pretrain-5k-loss-comparison.png
```

Repeat the `scp` and plotting commands at any time during training. The plotter
combines multiple Lightning CSV versions after resume, smooths the training
curve, plots all available validation points, and prints the latest numeric
values. Budget roughly 26--35 hours for vanilla on two 3090s and 51--61 hours
for DCache on two 4090s, pending a short measurement on the actual machines.

## Legacy block-16 end-to-end smoke check

This uses a tiny synthetic model and data, but runs the real recurrent rollout,
Lightning optimizer step, validation likelihood path, checkpoint save/reload,
semi-autoregressive sampler, both cache types, and generative-PPL evaluator:

```bash
python scripts/smoke_test_dcache.py
```

The first run downloads `sshleifer/tiny-gpt2` only for the evaluator. To test
everything except external generative PPL, use:

```bash
python scripts/smoke_test_dcache.py --skip-generative-ppl
```

Success ends with `DCACHE_SMOKE_OK` and a JSON summary.

To profile the real BD3-small shape (sequence 1024 and block 16)
before a long run:

```bash
python scripts/profile_full_dcache.py
```

This profiles the mature five-forward rollout by default, performs one full
recurrent training update and one validation batch, and reports peak
allocated/reserved VRAM. Its random-token loss is only a health check. Use
`--rollout-forwards 2 --final-mask-ratio 0.9375` to profile the early curriculum.
Pass `--micro-batch 2` (or higher) before increasing the real launcher's
`DCACHE_MICRO_BATCH`.

## OpenWebText data

Both comparison scripts use exactly the upstream `openwebtext-split` pipeline
and GPT-2 tokenizer. By default, Hugging Face data is cached under
`.cache/huggingface` (ignored by Git). OpenWebText is large, so point both runs
at the same faster/larger location when appropriate:

```bash
export DCACHE_DATA_DIR=/path/with/enough/space/bd3-data
```

The first invocation downloads and tokenizes the dataset. Reusing the same
cache is important for a matched comparison.

## Legacy 50k block-16 fine-tuning comparison

These scripts record the earlier experiment and are no longer the authoritative
pretraining comparison. The recurrent trial and vanilla control both train
BD3-small from scratch with
block size 16, sequence length 1024, the same seed/config defaults, effective
global batch 512, and four GPUs. The tested default per-GPU microbatch is `2`;
Lightning uses 64-way gradient accumulation to reach the global batch.

```bash
bash scripts/train/train_owt_dcache_50k.sh
bash scripts/train/train_owt_vanilla_50k.sh
```

Useful runtime overrides:

```bash
DCACHE_MICRO_BATCH=1 DCACHE_GLOBAL_BATCH=512 \
  bash scripts/train/train_owt_dcache_50k.sh trainer.max_steps=100
```

The full mature five-forward rollout was measured on this server at 14.06 GiB
peak allocated / 14.11 GiB peak reserved with microbatch 2. Microbatch 1 used
8.96 / 9.00 GiB. Validation used 3.09 / 3.15 GiB. Keep some headroom for DDP
and data-pipeline variation; profile before raising the microbatch above 2.
Extra arguments are forwarded as Hydra overrides, so a stopped run can resume
from its own `last.ckpt`.

Checkpoints are written to:

```text
outputs/owt-dcache-50k/checkpoints/
outputs/owt-vanilla-50k/checkpoints/
```

Override `DCACHE_RUN_DIR` to place either run elsewhere.

## Validation and generation evaluation

Likelihood/NELBO-derived perplexity on the upstream OpenWebText validation
split:

```bash
bash scripts/eval/eval_dcache_checkpoint.sh \
  outputs/owt-dcache-50k/checkpoints/last.ckpt ppl_eval
```

Generation, GPT-2-Large generative perplexity, and entropy:

```bash
bash scripts/eval/eval_dcache_checkpoint.sh \
  outputs/owt-dcache-50k/checkpoints/last.ckpt sample_eval
```

To do a cheap validation check first, cap the number of batches:

```bash
bash scripts/eval/eval_dcache_checkpoint.sh \
  outputs/owt-dcache-50k/checkpoints/last.ckpt ppl_eval \
  trainer.limit_val_batches=10
```

For the vanilla checkpoint, use the upstream scripts or pass
`step_memory.enabled=false` as the final override.

## Important fairness note

The recurrent model has an extra attention sublayer and rollout forwards, so a
50k-vs-50k result is a matched-update comparison, not a matched-FLOP result.
The measured parameter counts are 197,947,986 recurrent versus 169,627,218
vanilla (28,320,768 additional parameters). Disabled mode does not construct
unused denoising modules, which keeps the vanilla control faithful and avoids
unused-parameter failures under DDP.
Always report parameter count, real forward count, wall time, and peak VRAM;
add a matched-compute comparison before making a strong quality claim.
