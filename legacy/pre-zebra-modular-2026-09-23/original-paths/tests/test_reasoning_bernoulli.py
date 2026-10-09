"""Corruption-only ablation: law, protected clues, empty draws and resume."""
import copy
import json
import math
import os
from pathlib import Path
import socket
import subprocess

import pytest
import torch

from reasoning import runner
from reasoning.data import ReasoningDataset, prepare_dataset
from reasoning.model import DEFAULTS, ReasoningModel
from test_reasoning_model import batch, config, nonzero_head
from test_reasoning_runner import assert_tree_equal


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')

    def forbidden(*args, **kwargs):
        raise AssertionError('Bernoulli unit tests must not use CUDA or network')

    for key in ('_lazy_init', 'set_device', 'get_rng_state'):
        monkeypatch.setattr(torch.cuda, key, forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def plain_config(**overrides):
    return {**config(attention_mode='vanilla', trajectory='single'),
            'corruption_mode': 'bernoulli', 'corruption_timesteps': 64, **overrides}


def test_exact_upstream_mask_law_clues_padding_eos_and_rng():
    model = ReasoningModel(plain_config())
    b = batch()
    # Include an ineligible outer PAD even when target_mask alone says True.
    b['target_mask'][:, -1] = True
    actual_rng = torch.Generator().manual_seed(17)
    reference_rng = torch.Generator().manual_seed(17)
    eligible = b['target_mask'] & b['attention_mask']
    timestep = torch.randint(1, 65, (2, 1), generator=reference_rng)
    probability = timestep.float() / 64
    expected = (torch.rand(b['input_ids'].shape, generator=reference_rng) < probability) & eligible
    result = model.sample_trajectory(b, actual_rng)
    assert len(result['states']) == 1
    assert torch.equal(result['masks'][0], expected)
    assert torch.equal(result['states'][0], b['input_ids'].masked_fill(expected, model.mask_id))
    assert torch.equal(result['states'][0][~eligible], b['input_ids'][~eligible])
    torch.testing.assert_close(result['sampled_ratios'], probability)
    assert torch.equal(actual_rng.get_state(), reference_rng.get_state())
    assert not model.has_dcache and not model.has_final


def test_timestep_one_always_fully_masks_answers_not_clues():
    model = ReasoningModel(plain_config(corruption_timesteps=1))
    b = batch()
    result = model.sample_trajectory(b, torch.Generator().manual_seed(8))
    assert torch.equal(result['masks'][0], b['target_mask'] & b['attention_mask'])
    loss, metrics = model.compute_loss(b, generator=torch.Generator().manual_seed(8))
    loss.backward()
    assert metrics['full_mask_fraction_single'] == 1
    assert metrics['zero_mask_fraction_single'] == 0
    assert metrics['num_forwards'] == 1
    assert torch.isfinite(loss)


def test_empirical_full_empty_and_count_variance_match_formula():
    model = ReasoningModel(plain_config(corruption_timesteps=8))
    repeats = 100000
    # Four answer slots; no backbone forwards are needed for this distribution test.
    b = {key: value[:1].expand(repeats, -1).clone() for key, value in batch().items()}
    b['target_mask'][:, 8:] = False
    result = model.sample_trajectory(b, torch.Generator().manual_seed(937))
    counts = result['masks'][0].sum(-1)
    probability = result['sampled_ratios'].flatten()
    for event, expected in (
            (counts.eq(4), sum((j/8)**4 for j in range(1, 9))/8),
            (counts.eq(0), sum((1-j/8)**4 for j in range(1, 9))/8)):
        measured = event.float().mean().item()
        assert abs(measured - expected) < 6*math.sqrt(expected*(1-expected)/repeats)
    # Independent token decisions, not rounding: p=.5 does NOT imply count=2.
    middle = counts[probability.eq(.5)].float()
    assert middle.mean().item() == pytest.approx(2, abs=.04)
    assert middle.var().item() == pytest.approx(1, abs=.06)


def test_zero_target_microbatch_has_finite_zero_loss_gradients_and_metrics():
    model = nonzero_head(ReasoningModel(plain_config()))
    b = batch()
    b['target_mask'][:, 5:] = False  # one target per row makes empty draws frequent
    seed = next(seed for seed in range(100) if not model.sample_trajectory(
        b, torch.Generator().manual_seed(seed))['masks'][0].any())
    loss, metrics = model.compute_loss(b, generator=torch.Generator().manual_seed(seed))
    assert loss.requires_grad and loss.item() == 0
    loss.backward()
    assert metrics['zero_mask_fraction_single'] == 1
    assert metrics['accuracy_single'] == 0
    assert all(torch.isfinite(value).all() for value in metrics.values())
    # DDP can reduce zeros even when another rank has supervised examples.
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.grad.isfinite().all() and parameter.grad.count_nonzero() == 0


def test_mixed_zero_and_nonzero_examples_keep_per_example_objective():
    model = nonzero_head(ReasoningModel(plain_config()))
    b = batch()
    b['target_mask'][:, 5:] = False
    seed = next(seed for seed in range(100) if model.sample_trajectory(
        b, torch.Generator().manual_seed(seed))['masks'][0].any(-1).sum() == 1)
    loss, metrics = model.compute_loss(b, generator=torch.Generator().manual_seed(seed))
    assert metrics['zero_mask_fraction_single'] == .5
    torch.testing.assert_close(loss.detach()*2, metrics['nonempty_nll_single'])
    loss.backward()
    assert any(p.grad.abs().sum() > 0 for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize('trajectory', ['single', 'five'])
def test_historical_configs_and_sampler_rng_stay_unchanged(trajectory):
    supplied = config(attention_mode='vanilla', trajectory=trajectory)
    model = ReasoningModel(supplied)
    assert model.config == {**copy.deepcopy(DEFAULTS), **supplied}
    explicit = ReasoningModel({**supplied, 'corruption_mode': 'rounded_count'})
    assert explicit.config == model.config
    b = batch()
    a = torch.Generator().manual_seed(11)
    ref = torch.Generator().manual_seed(11)
    result = model.sample_trajectory(b, a)
    if trajectory == 'single':
        sampled = torch.rand((2, 1), generator=ref)
    else:
        k = .025 + torch.rand((2, 1), generator=ref)*(.10-.025)
        center = 1.5*k + torch.rand((2, 1), generator=ref)*(.9975-3*k)
        sampled = torch.cat((torch.ones_like(k), center + k*torch.tensor([1.5,.5,-.5,-1.5])), dim=1)
    expected = [torch.zeros_like(b['target_mask']) for _ in range(sampled.shape[1])]
    for row in range(2):
        positions = b['target_mask'][row].nonzero().flatten()
        n = len(positions)
        order = positions[torch.randperm(n, generator=ref)]
        previous = n+1
        for index in range(sampled.shape[1]):
            m = max(1, min(n, int(torch.round(sampled[row,index]*n))))
            if trajectory == 'five':
                m = n if index == 0 else max(5-index, min(previous-1, m))
            expected[index][row, order[:m]] = True
            previous = m
    for observed, wanted in zip(result['masks'], expected):
        assert torch.equal(observed, wanted)
    assert torch.equal(a.get_state(), ref.get_state())


@pytest.mark.parametrize('change', [dict(corruption_mode='typo'),
    dict(corruption_timesteps=0), dict(corruption_timesteps=-1),
    dict(corruption_timesteps=2.5), dict(corruption_timesteps=True),
    dict(trajectory='five'), dict(neighbors=True), dict(memory_mode='both'),
    dict(attention_mode='separate'), dict(corruption_mode='rounded_count')])
def test_invalid_or_nonbaseline_settings_fail(change):
    with pytest.raises(ValueError):
        ReasoningModel(plain_config(**change))


def train_args(data, run, steps=2):
    return runner.parser().parse_args([
        'train', '--suite', 'split', '--variant', 'mdm', '--task', 'countdown',
        '--data-dir', str(data), '--run-dir', str(run), '--size', 'debug',
        '--device', 'cpu', '--precision', 'fp32', '--cpu-threads', '1',
        '--global-batch', '2', '--micro-batch', '2', '--max-steps', str(steps),
        '--val-every', '1', '--validation-examples', '1', '--eval-batch-size', '1',
        '--save-every', '1', '--save-seconds', '0', '--log-every', '1',
        '--corruption-mode', 'bernoulli'])


def test_resume_rng_optimizer_and_sampler_contract(tmp_path):
    data = tmp_path / 'data'
    prepare_dataset(data, 'countdown', train_size=4, valid_size=1, test_size=1, seed=33)
    direct, resumed = tmp_path / 'direct', tmp_path / 'resumed'
    runner.train(train_args(data, direct))
    runner.train(train_args(data, resumed, steps=1))
    runner.train(train_args(data, resumed))
    a = runner.load_checkpoint(direct / 'checkpoints/last.pt')
    b = runner.load_checkpoint(resumed / 'checkpoints/last.pt')
    for key in ('model', 'optimizer', 'rng_by_rank', 'contract'):
        assert_tree_equal(a[key], b[key])
    assert a['model_config']['corruption_timesteps'] == 64
    evaluation = tmp_path / 'generation.json'
    runner.evaluate(runner.parser().parse_args([
        'evaluate', '--checkpoint', str(resumed / 'checkpoints/last.pt'),
        '--data-dir', str(data), '--output', str(evaluation), '--examples', '1',
        '--batch-size', '1', '--device', 'cpu', '--cpu-threads', '1']))
    assert json.loads(evaluation.read_text())['contract']['model_config']['corruption_mode'] == 'bernoulli'
    for mode, steps in [('rounded_count', None), ('bernoulli', 32)]:
        args = train_args(data, resumed)
        args.corruption_mode, args.corruption_timesteps = mode, steps
        with pytest.raises(ValueError, match='Resume contract differs'):
            runner.train(args)
    assert (resumed / 'validation/step-000000002.json').exists()
    dataset = ReasoningDataset(data)
    args = train_args(data, tmp_path / 'unlaunched')
    args.variant = 'tt'
    with pytest.raises(ValueError, match='plain single-forward MDM'):
        runner.build_model_config(args, dataset)
    args.variant, args.corruption_mode, args.corruption_timesteps = 'mdm', 'rounded_count', 64
    with pytest.raises(ValueError, match='requires'):
        runner.build_model_config(args, dataset)


def test_launcher_plan_is_isolated_and_only_changes_corruption():
    root = Path(__file__).resolve().parents[1]
    env = {k:v for k,v in os.environ.items() if not k.startswith('DCACHE_')}
    result = subprocess.run(['bash', str(root / 'scripts/reasoning/run_zebra_mdm_bernoulli_2x3090.sh'), 'plan'],
                            cwd=root, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    for expected in ('CUDA_VISIBLE_DEVICES=2,3', '--variant mdm', '--suite split',
                     '--epochs 3', '--micro-batch 32', '--global-batch 128', '--lr 0.0003',
                     '--corruption-mode bernoulli', '--corruption-timesteps 64',
                     '--validation-examples 1000', '--policy top_prob', '--examples 1000',
                     'outputs/reasoning/zebra-mdm-bernoulli-t64-3ep-2x3090-mb32'):
        assert expected in result.stdout
    assert '--stress-memory-routes' not in result.stdout
    assert '/tmp/' not in result.stdout
