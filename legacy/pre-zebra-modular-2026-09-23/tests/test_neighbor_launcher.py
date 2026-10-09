"""Exercise the complete neighbor launcher chain without starting a GPU job."""

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import hydra
from omegaconf import OmegaConf, open_dict
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
NAME = 'train_owt_dcache_merged_neighbors_5k.sh'


@pytest.fixture
def launch(tmp_path_factory):
    # Keep the fake checkout short enough for project-local UNIX socket paths.
    tmp_path = tmp_path_factory.mktemp('neighbor')
    target = tmp_path / 'scripts/train'
    target.mkdir(parents=True)
    for name in (NAME, 'train_owt_dcache_final_state_5k_2x3090.sh',
                 'train_owt_dcache_pretrain_100k.sh'):
        shutil.copyfile(ROOT / 'scripts/train' / name, target / name)
    data = tmp_path / '.cache/huggingface'
    for dataset in ('openwebtext-train_train_bs1024_wrapped_specialFalse.dat',
                    'openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat'):
        directory = data / dataset
        directory.mkdir(parents=True)
        for name in ('state.json', 'dataset_info.json'):
            (directory / name).write_text('{}')
    calls_path = tmp_path / 'calls.jsonl'
    recorder = tmp_path / 'python'
    recorder.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
with open(os.environ['NEIGHBOR_TEST_CALLS'], 'a') as handle:
    handle.write(json.dumps({'args': sys.argv[1:], 'env': dict(os.environ)}) + '\\n')
if sys.argv[1:2] == ['-c'] and 'torch.cuda.device_count()' in sys.argv[2]:
    print(os.environ['DCACHE_DEVICES'])
''')
    recorder.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('DCACHE_')}
    env.update(DCACHE_PYTHON=str(recorder),
               NEIGHBOR_TEST_CALLS=str(calls_path))

    def run(*args, **variables):
        result = subprocess.run(
            ['/bin/bash', str(target / NAME), *args], env={**env, **variables},
            text=True, capture_output=True, timeout=20)
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()] \
            if calls_path.exists() else []
        return result, calls

    return tmp_path, run


def compose(call):
    for name, resolver in {
        'cwd': lambda: str(ROOT), 'device_count': lambda: 2, 'eval': eval,
        'div_up': lambda x, y: (x + y - 1) // y,
    }.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
    assert call['args'][:2] == ['-u', 'main.py']
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / 'configs')):
        return hydra.compose(config_name='config', overrides=call['args'][2:])


@pytest.mark.parametrize('neighbor', [True, False])
@pytest.mark.parametrize('mode', ['adjacent', 'detached'])
@pytest.mark.parametrize('policy', ['legacy', 'current_preserving'])
def test_five_state_recipe_and_only_requested_gradient_difference(launch, mode, neighbor, policy):
    root, run = launch
    result, calls = run(DCACHE_GRADIENT_MODE=mode,
                        DCACHE_NEIGHBOR_ENABLED=str(neighbor).lower(),
                        DCACHE_MERGED_POLICY=policy)
    assert result.returncode == 0, result.stdout + result.stderr
    config = compose(calls[-1])
    assert config.step_memory.attention_mode == 'merged'
    assert config.step_memory.merged_policy == policy
    assert config.step_memory.enabled and config.step_memory.use_previous_kv
    assert config.dcachehooping.adjacent_grad.enabled == (mode == 'adjacent')
    assert config.step_memory.detach_between_steps == (mode == 'detached')
    assert not config.dcachehooping.two_forward.enabled
    assert config.neighbor_prediction.enabled == neighbor
    assert config.neighbor_prediction.weight == 0.5
    assert config.neighbor_prediction.chunk_size == 128
    assert config.neighbor_prediction.checkpoint_chunks
    assert config.trainer.max_steps == 5000
    assert config.trainer.devices == 2
    assert config.loader.batch_size == 2
    assert config.loader.global_batch_size == 512
    assert config.trainer.accumulate_grad_batches == 128
    assert config.trainer.val_check_interval == 64000
    assert config.trainer.limit_val_batches == 100
    assert config.callbacks.checkpoint_every_n_steps.save_top_k == 3
    assert config.callbacks.checkpoint_every_n_steps.every_n_train_steps == 500
    source = 'neighbors' if neighbor else 'no-neighbors'
    suffix = '-current-preserving' if policy == 'current_preserving' else ''
    assert config.checkpointing.save_dir == str(
        root / f'outputs/owt-dcache-merged-{source}-final-state-{mode}-5k{suffix}')
    assert config.checkpointing.resume_ckpt_path.endswith('/checkpoints/last.ckpt')
    pretrain = config.step_memory.pretrain
    assert [pretrain[key] for key in ('full_loss_weight', 't0_loss_weight',
            't1_loss_weight', 't2_loss_weight', 't3_loss_weight')] == \
        [0.05, 0.10, 0.20, 1.00, 0.70]
    assert pretrain.teacher_token_probability == 1.0
    assert pretrain.source_dropout.enabled
    assert pretrain.source_dropout.cache_only_probability == (0.0 if suffix else 0.20)
    assert pretrain.source_dropout.current_only_probability == 0.05
    assert pretrain.identity.batch_probability == 0.25
    assert config.step_memory.gate.enabled == (policy == 'legacy')
    assert config.step_memory.gate.init == 0.1
    assert config.dcachehooping.latent_dropout_probability == 0.10
    assert config.dcachehooping.identity_final_probability == 0.50
    assert not config.dcachehooping.status_embedding.enabled
    assert not config.dcachehooping.tentative.enabled
    assert not config.dcachehooping.confidence.enabled
    assert config.dcachehooping.latent_mask_probability == 0
    assert calls[-1]['env']['CUDA_VISIBLE_DEVICES'] == '2,3'
    assert calls[-1]['env']['TMPDIR'] == str(root / '.cache/runtime/merged-neighbors/tmp')


def test_runtime_overrides_allow_small_isolated_smoke(launch):
    _, run = launch
    result, calls = run('trainer.limit_train_batches=8', DCACHE_MAX_STEPS='1',
                       DCACHE_DEVICES='1', DCACHE_MICRO_BATCH='8',
                       DCACHE_CUDA_VISIBLE_DEVICES='0', DCACHE_VAL_BATCHES='200')
    assert result.returncode == 0, result.stderr
    config = compose(calls[-1])
    assert config.trainer.max_steps == 1
    assert config.trainer.limit_train_batches == 8
    assert config.trainer.accumulate_grad_batches == 64
    assert config.trainer.val_check_interval == 32000
    assert config.trainer.limit_val_batches == 200
    assert calls[-1]['env']['CUDA_VISIBLE_DEVICES'] == '0'


@pytest.mark.parametrize('argument', [
    'step_memory.attention_mode=separate', '+neighbor_prediction.weight=1.0',
    '++dcachehooping.adjacent_grad.enabled=false', '~neighbor_prediction',
    'checkpointing.resume_ckpt_path=/old.ckpt',
    'checkpointing.save_dir=/old-run', 'model=large',
    'training.from_pretrained=/old.ckpt', 'loader.batch_size=8',
])
def test_recipe_overrides_cannot_bypass_resume_or_architecture_checks(launch, argument):
    _, run = launch
    result, calls = run(argument)
    assert result.returncode == 2
    assert 'Cannot override recipe field' in result.stderr
    assert not calls


@pytest.mark.parametrize('failure', ['old-source', 'compact', 'missing-data',
                                    'broken-last', 'orphan-checkpoint', 'bad-mode', 'bad-policy'])
def test_unsafe_launches_stop_before_importing_torch(launch, failure):
    root, run = launch
    settings = {}
    data = root / '.cache/huggingface'
    ckpts = root / 'outputs/owt-dcache-merged-neighbors-final-state-adjacent-5k/checkpoints'
    if failure == 'old-source':
        settings['DCACHE_RESUME_CKPT'] = '/old/0-1500.ckpt'
    elif failure == 'compact':
        (data / 'compact_train.json').write_text('{}')
    elif failure == 'missing-data':
        (data / 'openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat/state.json').unlink()
    elif failure in ('broken-last', 'orphan-checkpoint'):
        ckpts.mkdir(parents=True)
        if failure == 'broken-last':
            (ckpts / 'last.ckpt').symlink_to('missing.ckpt')
        else:
            (ckpts / '0-500.ckpt').write_bytes(b'mock')
    elif failure == 'bad-mode':
        settings['DCACHE_GRADIENT_MODE'] = 'all-steps'
    elif failure == 'bad-policy':
        settings['DCACHE_MERGED_POLICY'] = 'unknown'
    result, calls = run(**settings)
    assert result.returncode == 2
    assert not calls


@pytest.mark.parametrize('neighbor', [True, False])
@pytest.mark.parametrize('mode', ['adjacent', 'detached'])
@pytest.mark.parametrize('changed', [None, 'step_memory.attention_mode',
    'dcachehooping.adjacent_grad.enabled', 'step_memory.detach_between_steps',
    'neighbor_prediction.enabled', 'neighbor_prediction.weight',
    'step_memory.pretrain.t2_loss_weight', 'loader.batch_size', 'optimizer_states'])
def test_resume_guard_executes_and_rejects_wrong_recipe(launch, monkeypatch, mode, changed, neighbor):
    root, run = launch
    source = 'neighbors' if neighbor else 'no-neighbors'
    ckpts = root / f'outputs/owt-dcache-merged-{source}-final-state-{mode}-5k/checkpoints'
    ckpts.mkdir(parents=True)
    (ckpts / 'last.ckpt').write_bytes(b'mocked-torch-load')
    result, calls = run(DCACHE_GRADIENT_MODE=mode,
                        DCACHE_NEIGHBOR_ENABLED=str(neighbor).lower())
    assert result.returncode == 0, result.stdout + result.stderr
    config = compose(calls[-1])
    guard = calls[0]
    assert guard['args'][0] == '-c'
    assert 'torch.load' in guard['args'][1]
    checkpoint = {'hyper_parameters': {'config': copy.deepcopy(config)},
                  'global_step': 500, 'optimizer_states': [{}],
                  'state_dict': {'mock': torch.ones(1)}}
    if changed == 'optimizer_states':
        checkpoint.pop('optimizer_states')
    elif changed is not None:
        current = OmegaConf.select(checkpoint['hyper_parameters']['config'], changed)
        wrong = not current if isinstance(current, bool) else \
            ('separate' if isinstance(current, str) else current + 0.25)
        OmegaConf.update(checkpoint['hyper_parameters']['config'], changed, wrong)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: checkpoint)
    for key, value in guard['env'].items():
        if key.startswith(('DCACHE_', 'NEIGHBOR_')):
            monkeypatch.setenv(key, value)
    code = compile(guard['args'][1], '<neighbor-checkpoint-guard>', 'exec')
    if changed is None:
        exec(code, {})
    else:
        with pytest.raises(SystemExit, match='Refusing incompatible|full training checkpoint'):
            exec(code, {})


@pytest.mark.parametrize('launch_policy,checkpoint_policy,allowed', [
    ('legacy', None, True),
    ('current_preserving', 'current_preserving', True),
    ('current_preserving', None, False),
    ('current_preserving', 'legacy', False),
    ('legacy', 'current_preserving', False),
])
def test_policy_guard_including_old_unlabelled_checkpoints(
        launch, monkeypatch, launch_policy, checkpoint_policy, allowed):
    root, run = launch
    directory = root / 'chosen-run'
    (directory / 'checkpoints').mkdir(parents=True)
    (directory / 'checkpoints/last.ckpt').write_bytes(b'mocked-torch-load')
    result, calls = run(DCACHE_RUN_DIR=str(directory), DCACHE_MERGED_POLICY=launch_policy)
    assert result.returncode == 0, result.stdout + result.stderr
    config = compose(calls[-1])
    if checkpoint_policy is None:
        # Original checkpoints predate this config field.
        with open_dict(config.step_memory):
            del config.step_memory.merged_policy
    else:
        config.step_memory.merged_policy = checkpoint_policy
    checkpoint = {'hyper_parameters': {'config': config}, 'global_step': 500,
                  'optimizer_states': [{}], 'state_dict': {'mock': torch.ones(1)}}
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: checkpoint)
    guard = calls[0]
    for key, value in guard['env'].items():
        if key.startswith(('DCACHE_', 'NEIGHBOR_')):
            monkeypatch.setenv(key, value)
    code = compile(guard['args'][1], '<merged-policy-checkpoint-guard>', 'exec')
    if allowed:
        exec(code, {})
    else:
        with pytest.raises(SystemExit, match='step_memory.merged_policy'):
            exec(code, {})


@pytest.mark.parametrize('checkpoint_policy,allowed', [
    (None, True), ('legacy', True), ('current_preserving', False)])
def test_historical_merged_launcher_is_explicitly_legacy_only(
        launch, monkeypatch, checkpoint_policy, allowed):
    root, _ = launch
    script_dir = root / 'scripts/train'
    name = 'train_owt_dcache_merged_adjacent_5k.sh'
    for filename in (name, 'train_owt_dcache_final_state_adjacent_5k_2x3090.sh'):
        shutil.copyfile(ROOT / 'scripts/train' / filename, script_dir / filename)
    ckpts = root / 'historical-run/checkpoints'
    ckpts.mkdir(parents=True)
    (ckpts / 'last.ckpt').write_bytes(b'mocked-torch-load')
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('DCACHE_')}
    env.update(DCACHE_RUN_DIR=str(ckpts.parent),
               DCACHE_PYTHON=str(root / 'python'),
               NEIGHBOR_TEST_CALLS=str(root / 'calls.jsonl'))
    result = subprocess.run(['/bin/bash', str(script_dir / name)], env=env,
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in (root / 'calls.jsonl').read_text().splitlines()]
    config = compose(calls[-1])
    assert config.step_memory.merged_policy == 'legacy'
    if checkpoint_policy is None:
        with open_dict(config.step_memory):
            del config.step_memory.merged_policy
    else:
        config.step_memory.merged_policy = checkpoint_policy
    checkpoint = {'hyper_parameters': {'config': config}}
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: checkpoint)
    monkeypatch.setenv('MERGED_CHECKPOINT', str(ckpts / 'last.ckpt'))
    code = compile(calls[0]['args'][1], '<historical-merged-guard>', 'exec')
    if allowed:
        exec(code, {})
    else:
        with pytest.raises(SystemExit, match='current-preserving checkpoint'):
            exec(code, {})
    # Neither environment nor Hydra flags may silently select the new recipe.
    for arguments, settings in [([], {'DCACHE_MERGED_POLICY': 'current_preserving'}),
                               (['step_memory.merged_policy=current_preserving'], {})]:
        failure = subprocess.run(['/bin/bash', str(script_dir / name), *arguments],
            env={**env, **settings}, text=True, capture_output=True, timeout=20)
        assert failure.returncode == 2
