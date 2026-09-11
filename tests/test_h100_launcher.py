"""Exercise cloud command construction without installing packages or using GPUs."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/cloud/lightning_h100.sh'


@pytest.fixture
def cloud(tmp_path):
    script = tmp_path / 'scripts/cloud/lightning_h100.sh'
    script.parent.mkdir(parents=True)
    shutil.copyfile(SOURCE, script)
    checkpoint = tmp_path / 'imports/adjacent/0-1500.ckpt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'only-a-command-construction-test')
    binaries = tmp_path / 'fakebin'
    binaries.mkdir()
    recorder = '''import json, os, sys
from pathlib import Path
kind = Path(sys.argv[0]).name
with open(os.environ['CALLS'], 'a') as handle:
    handle.write(json.dumps({'kind': kind, 'args': sys.argv[1:],
        'env': {k: v for k, v in os.environ.items()
                if k.startswith('DCACHE_') or k in ('TMPDIR', 'CUDA_VISIBLE_DEVICES')}}) + '\\n')
if kind == 'python' and sys.argv[1:2] == ['-c']:
    print(1 if 'torch.cuda.device_count()' in sys.argv[2] else 1500)
if kind == 'tmux' and 'has-session' in sys.argv:
    sys.exit(1)
if kind == 'conda':
    sys.exit(99)
if kind == 'python' and sys.argv[1:] == ['-m', 'pip', 'check']:
    sys.exit(int(os.environ.get('FAKE_PIP_CHECK_EXIT', '0')))
'''
    for name in ('python', 'bash', 'tmux', 'conda'):
        binary = binaries / name
        binary.write_text(f'#!{sys.executable}\n' + recorder)
        binary.chmod(0o755)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('DCACHE_'):
            env.pop(key)
    env.update(DCACHE_PYTHON=str(binaries / 'python'),
               PATH=str(binaries) + os.pathsep + env['PATH'],
               CALLS=str(tmp_path / 'calls.jsonl'))

    def run(*args, **overrides):
        result = subprocess.run(['/bin/bash', str(script), *args],
                                env={**env, **overrides}, text=True,
                                capture_output=True, timeout=20)
        calls = Path(env['CALLS'])
        entries = [json.loads(line) for line in calls.read_text().splitlines()] \
            if calls.exists() else []
        return result, entries

    return tmp_path, run


def test_shell_syntax_and_help(cloud):
    _, run = cloud
    syntax = subprocess.run(['/bin/bash', '-n', str(SOURCE)], capture_output=True)
    assert syntax.returncode == 0
    result, calls = run('--help')
    assert result.returncode == 0
    assert '5000' in result.stdout
    assert not calls


def test_setup_uses_existing_studio_python_without_conda(cloud):
    root, run = cloud
    result, calls = run('setup', DCACHE_PYTHON='')
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(item['kind'] == 'python' for item in calls)
    assert all(item['env']['DCACHE_PYTHON'] == str(root / 'fakebin/python')
               for item in calls)
    assert all(item['env']['TMPDIR'] == str(root / '.cache/runtime/h100/tmp')
               for item in calls)
    commands = [item['args'] for item in calls]
    assert ['-m', 'pip', 'install', '-r', 'requirements-h100.txt'] in commands
    assert ['-m', 'pip', 'check'] in commands
    assert any('https://download.pytorch.org/whl/cu126' in cmd for cmd in commands)
    assert 'Setup complete' in result.stdout
    assert not (root / '.cache/envs/dcache').exists()
    assert not any('main.py' in cmd for cmd in commands)


def test_setup_keeps_explicit_interpreter_override(cloud):
    root, run = cloud
    custom = root / 'chosen environment/python'
    custom.parent.mkdir()
    shutil.copyfile(root / 'fakebin/python', custom)
    custom.chmod(0o755)
    result, calls = run('setup', DCACHE_PYTHON=str(custom))
    assert result.returncode == 0, result.stderr
    assert all(item['env']['DCACHE_PYTHON'] == str(custom) for item in calls)


@pytest.mark.parametrize('action', ['setup', 'train'])
def test_missing_explicit_python_never_creates_environment(cloud, action):
    root, run = cloud
    result, calls = run(action, DCACHE_PYTHON=str(root / 'missing/bin/python'))
    assert result.returncode == 2
    assert not calls
    assert 'No executable Python found' in result.stderr


@pytest.mark.parametrize('version,expected', [((3, 8), 1), ((3, 9), 0),
    ((3, 10), 0), ((3, 11), 0), ((3, 12), 0), ((3, 13), 1)])
def test_setup_python_version_guard_before_any_install(cloud, version, expected):
    _, run = cloud
    result, calls = run('setup')
    assert result.returncode == 0
    # Execute the actual guard, with only sys.version_info replaced. No pip,
    # conda, GPU or package import runs in this subprocess.
    assert calls[0]['args'][0] == '-c'
    guard = calls[0]['args'][1]
    code = f'import sys; sys.version_info = {version!r}\n' + guard
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == expected, result.stdout + result.stderr
    if expected:
        assert 'No packages changed' in result.stderr


def test_setup_stops_on_dependency_conflicts(cloud):
    _, run = cloud
    result, calls = run('setup', FAKE_PIP_CHECK_EXIT='1')
    assert result.returncode == 1
    assert calls[-1]['args'] == ['-m', 'pip', 'check']
    assert 'Setup complete' not in result.stdout


def test_non_setup_actions_also_use_active_python_by_default(cloud):
    root, run = cloud
    result, calls = run('check', '--cpu-only', DCACHE_PYTHON='')
    assert result.returncode == 0
    assert calls[0]['env']['DCACHE_PYTHON'] == str(root / 'fakebin/python')


def test_cloud_runtime_pins_match_local_without_notebook_tools():
    repo = SOURCE.parents[2]
    def pins(path):
        return {line.strip() for line in path.read_text().splitlines()
                if line.strip() and not line.startswith(('#', '-'))}
    assert pins(repo / 'requirements-h100.txt') == \
        pins(repo / 'requirements.txt') - {'notebook==7.2.2', 'git-lfs==1.6', 'nvitop==1.3.2'}
    assert '-c requirements-h100-constraints.txt' in \
        (repo / 'requirements-h100.txt').read_text().splitlines()


def test_cloud_check_is_explicitly_cpu_only_and_strict_on_import(cloud):
    root, run = cloud
    result, calls = run('check', '--cpu-only')
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert '--cpu-only' in calls[0]['args']
    assert '--allow-descendant' not in calls[0]['args']
    assert calls[0]['env']['TMPDIR'] == str(root / '.cache/runtime/h100/tmp')


def test_full_run_keeps_geometry_validation_retention_and_resume(cloud):
    root, run = cloud
    result, calls = run('train')
    assert result.returncode == 0, result.stderr
    launch = next(item for item in calls if item['kind'] == 'bash')
    env = launch['env']
    assert env['CUDA_VISIBLE_DEVICES'] == '0'
    for key, expected in dict(DEVICES='1', MICRO_BATCH='2', GLOBAL_BATCH='512',
                              MAX_STEPS='5000', VAL_BATCHES='200',
                              VAL_INTERVAL='500', CHECKPOINT_SAVE_TOP_K='3').items():
        assert env['DCACHE_' + key] == expected
    assert 'checkpointing.allow_batch_geometry_change=true' in launch['args']
    assert 'strategy=ddp' in launch['args']
    assert 'checkpointing.resume_from_ckpt=true' in launch['args']
    assert f'checkpointing.resume_ckpt_path={root}/imports/adjacent/0-1500.ckpt' in launch['args']


def test_smoke_is_one_optimizer_update_and_isolates_output(cloud):
    root, run = cloud
    result, calls = run('smoke')
    assert result.returncode == 0, result.stderr
    launch = next(item for item in calls if item['kind'] == 'bash')
    assert launch['env']['DCACHE_MAX_STEPS'] == '1501'
    assert launch['env']['DCACHE_VAL_BATCHES'] == '2'
    assert launch['env']['DCACHE_VAL_INTERVAL'] == '1'
    assert launch['env']['DCACHE_CHECKPOINT_SAVE_TOP_K'] == '1'
    assert launch['env']['DCACHE_RUN_DIR'].startswith(str(root / '.cache/runtime/h100/smoke.'))
    assert 'callbacks.checkpoint_every_n_steps.every_n_train_steps=1' in launch['args']


def test_existing_cloud_checkpoint_is_preferred(cloud):
    root, run = cloud
    ckpt_dir = root / 'outputs/owt-dcache-final-state-adjacent-pretrain-5k-2x3090/checkpoints'
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / '0-2000.ckpt').write_bytes(b'mock-descendant')
    (ckpt_dir / 'last.ckpt').symlink_to('0-2000.ckpt')
    result, calls = run('check', '--cpu-only')
    assert result.returncode == 0
    assert '--allow-descendant' in calls[0]['args']
    assert str(ckpt_dir / 'last.ckpt') in calls[0]['args']


def test_broken_last_link_never_falls_back_to_source(cloud):
    root, run = cloud
    ckpt_dir = root / 'outputs/owt-dcache-final-state-adjacent-pretrain-5k-2x3090/checkpoints'
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / 'last.ckpt').symlink_to('missing.ckpt')
    result, calls = run('train')
    assert result.returncode == 2
    assert not calls
    assert 'Missing resume checkpoint' in result.stderr


def test_plot_works_with_partial_cloud_history_and_override(cloud):
    root, run = cloud
    result, calls = run('plot', DCACHE_RUN_DIR=str(root / 'custom-run'))
    assert result.returncode == 0
    assert '--available-only' in calls[0]['args']
    assert '--training-only' in calls[0]['args']
    assert str(root / 'custom-run') in calls[0]['args']


def test_standalone_validation_does_not_pollute_training_csvs(cloud):
    root, run = cloud
    result, calls = run('validate')
    assert result.returncode == 0
    launch = next(item for item in calls if item['kind'] == 'bash')
    output = launch['args'][3]
    assert output.startswith(str(root / 'outputs/eval-adjacent-h100-step1500.'))
    assert 'owt-dcache-final-state' not in output


@pytest.mark.parametrize('action', ['train', 'check', 'smoke'])
def test_missing_source_fails_before_model_or_launch(cloud, action):
    root, run = cloud
    (root / 'imports/adjacent/0-1500.ckpt').unlink()
    result, calls = run(action)
    assert result.returncode == 2
    assert not calls


def test_invalid_workers_fails_before_launch(cloud):
    _, run = cloud
    result, calls = run('train', DCACHE_NUM_WORKERS='0')
    assert result.returncode == 2
    assert not calls


@pytest.mark.parametrize('micro', [2, 4])
def test_actual_launcher_chain_composes_the_verified_scientific_recipe(cloud, micro):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from checkpoint_resume import batch_geometry
    from scripts.cloud.check_h100_resume import scientific_config

    root, run = cloud
    repo = SOURCE.parents[2]
    scripts = root / 'scripts/train'
    scripts.mkdir(parents=True)
    for name in ('train_owt_dcache_final_state_adjacent_5k_2x3090.sh',
                 'train_owt_dcache_final_state_5k_2x3090.sh',
                 'train_owt_dcache_pretrain_100k.sh'):
        shutil.copyfile(repo / 'scripts/train' / name, scripts / name)
    # Run the real shell wrapper chain, but keep all Python/GPU calls mocked.
    (root / 'fakebin/bash').unlink()
    result, calls = run('train', DCACHE_PREFLIGHT_ONLY='0', DCACHE_MICRO_BATCH=str(micro))
    assert result.returncode == 0, result.stdout + result.stderr
    command = next(item['args'] for item in calls if item['args'][:2] == ['-u', 'main.py'])
    for name, resolver in (
        ('cwd', lambda: str(root)), ('eval', eval),
        ('div_up', lambda x, y: (x + y - 1) // y), ('device_count', lambda: 1)):
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
    with initialize_config_dir(version_base=None, config_dir=str(repo / 'configs')):
        config = compose(config_name='config', overrides=command[2:])
    expected = json.loads((repo / 'experiments/h100_transfer_manifest.json').read_text())
    assert scientific_config(OmegaConf.to_container(config, resolve=False)) == \
        expected['checkpoint']['scientific_config']
    assert batch_geometry(config) == {
        'version': 1, 'world_size': 1, 'micro_batch': micro, 'global_batch': 512,
        'accumulation': 512 // micro, 'distributed_sampler': True}
    assert config.trainer.val_check_interval == 500 * (512 // micro)
    assert config.loader.eval_batch_size == 2
    assert config.trainer.limit_val_batches == 200
    assert config.callbacks.checkpoint_every_n_steps.every_n_train_steps == 500
    assert config.callbacks.checkpoint_every_n_steps.save_top_k == 3
    assert config.checkpointing.allow_batch_geometry_change
    assert config.checkpointing.resume_from_ckpt
    assert config.trainer.max_steps == 5000


def test_mb4_tmux(cloud):
    root, run = cloud
    result, calls = run('smoke', DCACHE_MICRO_BATCH='4')
    assert result.returncode == 0, result.stderr
    launch = next(item for item in calls if item['kind'] == 'bash')
    assert launch['env']['DCACHE_MICRO_BATCH'] == '4'
    assert launch['env']['DCACHE_MAX_STEPS'] == '1501'
    assert 'loader.eval_batch_size=2' in launch['args']
    result, calls = run('tmux', DCACHE_MICRO_BATCH='4')
    assert result.returncode == 0, result.stderr
    command = next(item for item in calls if item['kind'] == 'tmux' and 'new-session' in item['args'])['args'][-1]
    assert 'DCACHE_MICRO_BATCH=4' in command
    assert 'pretrain-5k-h100-mb4' in command


@pytest.mark.parametrize('micro', ['0', '3', '8', 'oops'])
def test_unsupported_cloud_microbatch_rejected(cloud, micro):
    _, run = cloud
    result, calls = run('train', DCACHE_MICRO_BATCH=micro)
    assert result.returncode == 2
    assert not calls
