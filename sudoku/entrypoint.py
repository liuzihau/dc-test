#!/usr/bin/env python3
"""Isolated entrypoint; spawned Lightning ranks use the same module selection."""
import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / 'third_party/reasoning_with_latent_tokens'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTHOR))


def make_config(args):
    import hydra
    from omegaconf import OmegaConf
    import main as upstream  # Registers the upstream resolvers.
    run = args.run.resolve()
    data_root = ROOT / '.cache/downloads/reasoning-puzzles/Reasoning puzzles public data'
    train_candidates = [data_root / 'sudoku-train-data.npy',
                        data_root / 'Sudoku-train-data.npy']
    valid_candidates = [data_root / 'sudoku-test-data.npy',
                        data_root / 'Sudoku-test-data.npy',
                        ROOT / 'imports/official-reasoning/raw/sudoku-test.npy']
    train_data = next((p for p in train_candidates if p.is_file()), None)
    valid_data = next((p for p in valid_candidates if p.is_file()), None)
    if train_data is None or valid_data is None:
        tried = train_candidates + valid_candidates
        raise FileNotFoundError("Author Sudoku data missing; tried:\n  " +
                                "\n  ".join(map(str, tried)))
    with hydra.initialize_config_dir(version_base=None, config_dir=str(AUTHOR/'configs')):
        config = hydra.compose(config_name='experiment_base', overrides=[
            'data=sudoku-puzzle', 'algo=difflm', 'model=mini', 'sampling=synthetic_base'])
    OmegaConf.set_struct(config, False)
    recipe = OmegaConf.load(args.recipe)
    config.mechanisms = recipe.mechanisms
    config.objective = recipe.objective
    config.reasoning_generation_layout = getattr(args, 'generation_layout', 'author')
    config.seed = args.seed
    config.model.length = 192
    config.algo.diffusion_attn_mode = 'full'
    config.algo.diffusion_shuffle = False
    config.algo.shuffle_clean_tokens = False
    config.algo.shuffle_masked_tokens = False
    config.algo.ar_noise = False
    config.algo.next_token_prediction = False
    config.algo.loss_type = 'elbo'
    config.algo.shifted_logits = False
    config.training.train_on_all_tokens = False
    config.data.cache_dir = str(ROOT / '.cache/author-sudoku')
    config.data.train_data_path = str(train_data)
    config.data.valid_data_path = str(valid_data)
    config.checkpointing.save_dir = str(run)
    config.checkpointing.resume_from_ckpt = args.resume is not None
    config.checkpointing.resume_ckpt_path = str(args.resume) if args.resume else None
    config.wandb = None
    config.loader.num_workers = args.workers
    config.loader.global_batch_size = 512
    config.loader.eval_global_batch_size = 512
    config.loader.batch_size = args.microbatch
    # Keep validation batching fixed when changing training microbatch.
    config.loader.eval_batch_size = 128
    if 512 % (args.devices * args.microbatch):
        raise ValueError('512 must be divisible by devices * microbatch')
    config.trainer.devices = args.devices
    config.trainer.max_epochs = 1000
    config.trainer.max_steps = args.target_steps
    config.trainer.log_every_n_steps = 1
    config.trainer.num_sanity_val_steps = 0
    config.trainer.val_check_interval = 1.0
    config.trainer.check_val_every_n_epoch = 1
    # No logger-dependent LearningRateMonitor; local metrics record LR directly.
    del config.callbacks.learning_rate_monitor
    config.callbacks.checkpoint_monitor.update({
        'monitor': 'val/nll', 'mode': 'min', 'save_top_k': 1,
        'save_last': False, 'filename': 'best'})
    config.callbacks.checkpoint_every_n_steps.update({
        'every_n_train_steps': args.checkpoint_interval,
        'save_top_k': 3, 'save_last': 'link', 'monitor': 'step', 'mode': 'max',
        'filename': '{epoch}-{step}'})
    config.callbacks.local_metrics = {
        '_target_': 'zebra.metrics.AuthorLocalMetricsCallback',
        'output_dir': str(run / 'local_metrics'), 'train_every_n_steps': 1,
        'preserve_rng_around_validation': True}
    config.sampling.steps = 192
    config.sampling.num_sample_batches = args.eval_batches
    config.sampling.topk_candidate_min = args.candidate_window
    config.sampling.topk_candidate_max = args.candidate_window
    config.sampling.unmask_policy = 'topp'
    config.sampling.greedy_tokens = False
    config.sampling.kv_cache = False
    config.eval.disable_ema = False
    config.eval.run_task_evaluation = False
    if args.stage == 'evaluate':
        if args.resume is None:
            raise ValueError('Evaluation requires --resume checkpoint')
        config.mode = 'completions'
        config.trainer.devices = 1
        config.loader.batch_size = config.loader.eval_batch_size = args.eval_batch_size
        config.loader.global_batch_size = config.loader.eval_global_batch_size = args.eval_batch_size
        config.eval.checkpoint_path = str(args.resume)
        config.eval.generated_samples_path = str(run/'samples.json')
    else:
        config.mode = 'train'
    if args.smoke:
        # Actual architecture/backward at the requested microbatch; only two updates.
        config.trainer.limit_train_batches = 2 * (512 // (args.devices * args.microbatch))
        config.trainer.limit_val_batches = 2
        config.callbacks.checkpoint_every_n_steps.every_n_train_steps = 1
    return config, upstream


def main():
    sys.argv[0] = str(Path(__file__).resolve())
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stage', choices=['train', 'evaluate'], default='train')
    parser.add_argument('--target-steps', type=int, default=8790)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--devices', type=int, default=2)
    parser.add_argument('--microbatch', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--checkpoint-interval', type=int, default=3525)
    parser.add_argument('--eval-batches', type=int, default=10)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--candidate-window', type=int, default=8,
                        help='0 = unrestricted top-probability candidate set')
    parser.add_argument('--generation-layout', choices=['author', 'canonical'],
                        default='author',
                        help=('author dynamically packs/shuffles physical slots; '
                              'canonical keeps every token in its original slot'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.candidate_window < 0:
        parser.error('--candidate-window must be nonnegative')
    args.run = args.run.resolve()
    args.recipe = args.recipe.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve()
    from zebra.batch_policy import apply_scheduled_batch
    apply_scheduled_batch(args, ROOT)
    # Lightning respawns this argv after we chdir into the run directory.
    for flag, value in [('--run', args.run), ('--recipe', args.recipe), ('--resume', args.resume)]:
        if value is not None and flag in sys.argv:
            sys.argv[sys.argv.index(flag) + 1] = str(value)
    from zebra.runtime import install_no_cudagraph_compile
    install_no_cudagraph_compile()
    import lightning as L
    from omegaconf import OmegaConf
    from zebra.model import ZebraMDM
    config, upstream = make_config(args)
    args.run.mkdir(parents=True, exist_ok=True)
    os.chdir(args.run)
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        OmegaConf.save(config, args.run / 'resolved_config.yaml', resolve=True)
    L.seed_everything(args.seed)
    tokenizer = upstream.dataloader.get_tokenizer(config)
    logger = upstream.utils.get_logger('sudoku')
    fn = upstream._complete if args.stage == 'evaluate' else upstream._train
    fn(ZebraMDM, config, logger, tokenizer)


if __name__ == '__main__':
    main()
