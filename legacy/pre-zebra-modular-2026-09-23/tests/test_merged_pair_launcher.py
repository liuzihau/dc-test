"""Both hardware profiles exercise the same scientific recipe without GPUs."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

from test_neighbor_launcher import ROOT, compose, launch


PAIR = 'train_owt_dcache_merged_pair.sh'


@pytest.fixture
def pair(launch):
    root, _ = launch
    script = root / 'scripts/train' / PAIR
    shutil.copyfile(ROOT / 'scripts/train' / PAIR, script)
    binaries = root / 'bin'
    binaries.mkdir()
    tmux = binaries / 'tmux'
    tmux.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
with open(os.environ['PAIR_TEST_TMUX'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if 'has-session' in sys.argv:
    sys.exit(0 if os.environ.get('PAIR_TEST_EXISTING') == '1' else 1)
''')
    tmux.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith('DCACHE_')}
    env.update(DCACHE_PYTHON=str(root / 'python'),
               PATH=str(binaries) + os.pathsep + env['PATH'],
               NEIGHBOR_TEST_CALLS=str(root / 'calls.jsonl'),
               PAIR_TEST_TMUX=str(root / 'tmux.jsonl'))

    def run(profile='3090', auxiliary='off', action='train', *options, **variables):
        result = subprocess.run(['/bin/bash', str(script), profile, auxiliary, action, *options],
                                env={**env, **variables}, capture_output=True,
                                text=True, timeout=30)
        calls_file = root / 'calls.jsonl'
        calls = [json.loads(line) for line in calls_file.read_text().splitlines()] \
            if calls_file.exists() else []
        return result, calls
    return root, run, env


@pytest.mark.parametrize('profile,devices,batches,accumulation', [
    ('3090', 2, 100, 128), ('h100', 1, 200, 256)])
@pytest.mark.parametrize('auxiliary', ['off', 'on'])
@pytest.mark.parametrize('policy', ['legacy', 'current-preserving'])
def test_fixed_recipe_for_both_hardware_profiles(pair, profile, devices, batches, accumulation, auxiliary, policy):
    root, run, _ = pair
    result, calls = run(profile, auxiliary, 'train', '--attention-policy', policy)
    assert result.returncode == 0, result.stdout + result.stderr
    config = compose(calls[-1])
    assert config.neighbor_prediction.enabled == (auxiliary == 'on')
    assert config.neighbor_prediction.weight == 0.5
    assert config.step_memory.attention_mode == 'merged'
    assert config.step_memory.merged_policy == policy.replace('-', '_')
    assert config.step_memory.gate.enabled == (policy == 'legacy')
    dropout = config.step_memory.pretrain.source_dropout
    assert dropout.enabled and dropout.current_only_probability == 0.05
    assert dropout.cache_only_probability == (0.20 if policy == 'legacy' else 0.0)
    assert config.dcachehooping.enabled and config.dcachehooping.adjacent_grad.enabled
    assert not config.step_memory.detach_between_steps
    assert not config.dcachehooping.two_forward.enabled
    assert config.checkpointing.restore_data_cursor
    assert config.strategy._target_ == 'lightning.pytorch.strategies.DDPStrategy'
    assert config.trainer.num_nodes == 1
    assert config.trainer.devices == devices
    assert config.trainer.accumulate_grad_batches == accumulation
    assert config.trainer.val_check_interval == 500 * accumulation
    assert config.trainer.limit_val_batches == batches
    assert config.loader.batch_size == config.loader.eval_batch_size == 2
    assert config.loader.global_batch_size == 512
    assert config.trainer.max_steps == 5000
    assert config.callbacks.checkpoint_every_n_steps.save_top_k == 3
    assert config.callbacks.checkpoint_every_n_steps.every_n_train_steps == 500
    suffix = '' if policy == 'legacy' else '-current-preserving'
    assert config.checkpointing.save_dir == str(root / 'outputs' /
        f'owt-dcache-merged-final-state-adjacent-neighbors-{auxiliary}-{profile}-5k{suffix}')
    assert calls[-1]['env']['CUDA_VISIBLE_DEVICES'] == ('2,3' if devices == 2 else '0')
    assert calls[-1]['env']['TMPDIR'].startswith(str(root / '.cache'))
    status = Path(config.checkpointing.save_dir) / 'launch_status.txt'
    assert 'exit_code=0' in status.read_text()
    assert len(list((status.parent / 'launch_logs').glob('*.log'))) == 1


@pytest.mark.parametrize('profile', ['3090', 'h100'])
@pytest.mark.parametrize('policy', ['legacy', 'current-preserving'])
def test_off_and_on_differ_only_in_auxiliary_and_output(pair, profile, policy):
    _, run, _ = pair
    configs = []
    for auxiliary in ('off', 'on'):
        result, calls = run(profile, auxiliary, 'train', '--attention-policy', policy)
        assert result.returncode == 0, result.stderr
        config = compose(calls[-1])
        # Normalize the root before resolving interpolated callback directories.
        config.checkpointing.save_dir = '/SAME_RUN'
        config = OmegaConf.to_container(config, resolve=True)
        config['neighbor_prediction'].pop('enabled')
        config['checkpointing'].pop('save_dir')
        config['checkpointing'].pop('resume_ckpt_path')
        configs.append(config)
    assert configs[0] == configs[1]


def test_current_preserving_changes_only_two_mechanisms_and_identity_label(pair):
    _, run, _ = pair
    configs = []
    for policy in ('legacy', 'current-preserving'):
        result, calls = run('3090', 'off', 'train', '--attention-policy', policy)
        assert result.returncode == 0, result.stdout + result.stderr
        config = compose(calls[-1])
        config.checkpointing.save_dir = '/SAME_RUN'
        config.step_memory.merged_policy = 'SAME_POLICY'
        config.step_memory.gate.enabled = False
        config.step_memory.pretrain.source_dropout.cache_only_probability = 0.0
        configs.append(OmegaConf.to_container(config, resolve=True))
    assert configs[0] == configs[1]


def test_stale_shell_cannot_change_pair_recipe(pair):
    root, run, _ = pair
    result, calls = run('h100', 'on', DCACHE_GRADIENT_MODE='detached',
        DCACHE_MERGED_POLICY='current_preserving',
        DCACHE_DEVICES='4', DCACHE_NEIGHBOR_ENABLED='false', DCACHE_MICRO_BATCH='16',
        DCACHE_GLOBAL_BATCH='1024', DCACHE_VAL_INTERVAL='1', DCACHE_VAL_BATCHES='1',
        DCACHE_EVAL_BATCHES='1', DCACHE_CUDA_VISIBLE_DEVICES='2,3',
        DCACHE_RUN_DIR=str(root / 'old-run'))
    assert result.returncode == 0, result.stderr
    config = compose(calls[-1])
    assert config.loader.batch_size == 2
    assert config.loader.global_batch_size == 512
    assert config.trainer.devices == 1
    assert config.trainer.limit_val_batches == 200
    assert config.trainer.val_check_interval == 128000
    assert config.dcachehooping.adjacent_grad.enabled
    assert config.step_memory.merged_policy == 'legacy'
    assert config.neighbor_prediction.enabled
    assert calls[-1]['env']['CUDA_VISIBLE_DEVICES'] == '0'
    assert 'Ignoring old DCACHE_RUN_DIR' in result.stderr
    assert not (root / 'old-run').exists()


@pytest.mark.parametrize('profile', ['3090', 'h100'])
@pytest.mark.parametrize('auxiliary', ['off', 'on'])
def test_smoke_is_fresh_isolated_and_does_not_reuse_main_checkpoint(pair, profile, auxiliary):
    root, run, _ = pair
    main_run = root / 'user-specified-main'
    ckpts = main_run / 'checkpoints'
    ckpts.mkdir(parents=True)
    checkpoint = ckpts / 'last.ckpt'
    checkpoint.write_bytes(b'Never touch this checkpoint during smoke')
    result, calls = run(profile, auxiliary, 'smoke', DCACHE_PAIR_RUN_DIR=str(main_run))
    assert result.returncode == 0, result.stdout + result.stderr
    config = compose(calls[-1])
    assert config.trainer.max_steps == 1
    assert config.trainer.limit_val_batches == 1
    assert config.trainer.val_check_interval == config.trainer.accumulate_grad_batches
    assert config.callbacks.checkpoint_every_n_steps.every_n_train_steps == 1
    assert Path(config.checkpointing.save_dir).is_relative_to(root / '.cache/runtime/merged-pair')
    assert checkpoint.read_bytes() == b'Never touch this checkpoint during smoke'
    assert not (main_run / 'launch_logs').exists()
    assert not any('torch.load' in ' '.join(call['args']) for call in calls)


@pytest.mark.parametrize('policy', ['legacy', 'current-preserving'])
def test_tmux_uses_project_socket_and_propagates_runtime_settings(pair, policy):
    root, run, env = pair
    # Spaces exercise quoting in the complete command sent to tmux.
    directory = root / 'trial with spaces'
    result, calls = run('h100', 'on', 'tmux', '--attention-policy', policy,
                        DCACHE_PAIR_RUN_DIR=str(directory),
                        DCACHE_MAX_STEPS='7500', DCACHE_PAIR_MICRO_BATCH='4')
    assert result.returncode == 0, result.stdout + result.stderr
    tmux_calls = [json.loads(line) for line in (root / 'tmux.jsonl').read_text().splitlines()]
    new = next(args for args in tmux_calls if 'new-session' in args)
    assert new[:2] == ['-S', str(root / '.cache/tmux/merged-pair.sock')]
    expected_session = 'merged-h100-on' + ('' if policy == 'legacy' else '-current-preserving')
    assert new[new.index('-s') + 1] == expected_session
    command = shlex.split(new[-1])
    assert f'DCACHE_PAIR_RUN_DIR={directory}' in command
    assert 'DCACHE_PAIR_MICRO_BATCH=4' in command
    assert f'DCACHE_PYTHON={root / "python"}' in command
    assert command[-2:] == ['--attention-policy', policy]
    # Emulate an old server environment; forwarded argv must recover the intended run.
    result = subprocess.run(command, env={**env, 'DCACHE_RUN_DIR': '/old'},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in (root / 'calls.jsonl').read_text().splitlines()]
    config = compose(calls[-1])
    assert config.trainer.max_steps == 7500
    assert config.loader.batch_size == 4 and config.loader.eval_batch_size == 2
    assert config.checkpointing.save_dir == str(directory)
    assert config.step_memory.merged_policy == policy.replace('-', '_')


@pytest.mark.parametrize('options', [
    ('--attention-policy', 'unknown'), ('--attention-policy',),
    ('--attention-policy', 'legacy', '--attention-policy', 'current-preserving'),
    ('train',),
])
def test_invalid_policy_options_stop_before_launch(pair, options):
    _, run, _ = pair
    result, calls = run('3090', 'off', 'train', *options)
    assert result.returncode == 2
    assert not calls


def test_existing_tmux_is_not_replaced(pair):
    root, run, _ = pair
    result, calls = run('3090', 'off', 'tmux', PAIR_TEST_EXISTING='1')
    assert result.returncode == 0, result.stderr
    tmux_calls = [json.loads(line) for line in (root / 'tmux.jsonl').read_text().splitlines()]
    assert not any('new-session' in args for args in tmux_calls)
    assert not calls


def test_run_lock_prevents_duplicate_writers(pair):
    root, run, _ = pair
    directory = root / 'locked-run'
    directory.mkdir()
    with (directory / '.training.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, calls = run(DCACHE_PAIR_RUN_DIR=str(directory))
        assert result.returncode == 2
        assert 'Another launch' in result.stderr
        assert not calls


@pytest.mark.parametrize('settings', [
    {'DCACHE_RESUME_CKPT': '/old/0-1500.ckpt'},
    {'DCACHE_PAIR_MICRO_BATCH': '3'},
    {'DCACHE_MAX_STEPS': '0'}])
def test_invalid_or_old_continuation_settings_stop(pair, settings):
    _, run, _ = pair
    result, calls = run(**settings)
    assert result.returncode == 2
    assert not calls
