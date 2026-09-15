"""Corrected merged memory controls: real CPU forwards, gradients and recovery."""

import copy
import json
import socket

import pytest
import torch
from torch.nn import functional as F

from reasoning import runner
from reasoning.data import ReasoningDataset, prepare_dataset
from reasoning.model import DEFAULTS, ReasoningModel


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    for name in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')

    def forbidden(*_args, **_kwargs):
        raise AssertionError('Corrected reasoning tests must not use CUDA or network')

    for name in ('_lazy_init', 'set_device', 'get_rng_state'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


def config(**overrides):
    return dict(vocab_size=17, hidden_size=32, n_heads=4, n_layers=2,
                max_length=16, special_ids=[0, 1, 2, 3],
                memory_mode='both', merged_policy='current_preserving',
                gate_enabled=False, cache_only_probability=0.,
                **overrides)


def batch():
    clean = torch.tensor([[2, 4, 5, 3, 6, 7, 8, 9, 10, 11, 3, 0, 0, 0],
                          [2, 5, 4, 3, 9, 6, 7, 8, 11, 10, 3, 0, 0, 0]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    valid[:, -1] = False
    target = valid.clone()
    target[:, :4] = False
    return dict(input_ids=clean, attention_mask=valid, target_mask=target)


def initialized_model(**overrides):
    model = ReasoningModel(config(**overrides))
    # Output projections/latent fusion start at zero; open them for causal
    # gradient tests rather than accepting vacuous zero gradients at init.
    torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=.05)
    model.backbone.dcachehooping_latent_norm.weight.data.fill_(.2)
    return model


@pytest.fixture(scope='module')
def pilot(tmp_path_factory):
    directory = tmp_path_factory.mktemp('corrected-reasoning')
    prepare_dataset(directory, 'countdown', train_size=4, valid_size=2,
                    test_size=1, seed=33)
    return directory, ReasoningDataset(directory)


def args(pilot, run, variant='both_aux', steps=2, *extra):
    return runner.parser().parse_args([
        'train', '--task', 'countdown', '--variant', variant,
        '--data-dir', str(pilot[0]), '--run-dir', str(run),
        '--size', 'debug', '--device', 'cpu', '--precision', 'fp32',
        '--global-batch', '2', '--micro-batch', '2',
        '--max-steps', str(steps), '--warmup-steps', '2',
        '--val-every', '1', '--validation-examples', '2',
        '--eval-batch-size', '2', '--save-every', '1', '--save-seconds', '0',
        '--log-every', '1', '--seed', '14', '--cpu-threads', '1',
        '--merged-policy', 'current_preserving', *extra])


@pytest.mark.parametrize('variant', ['both', 'both_aux'])
@pytest.mark.parametrize('robustness', [True, False])
def test_corrected_cli_policy_and_backbone(pilot, tmp_path, variant, robustness):
    options = args(pilot, tmp_path / 'not-launched', variant)
    options.no_robustness = not robustness
    model = ReasoningModel(runner.build_model_config(options, pilot[1]))
    assert model.has_dcache and model.has_final
    assert model.config['gradient_mode'] == 'adjacent'
    assert model.config['merged_policy'] == 'current_preserving'
    assert not model.config['gate_enabled']
    assert model.config['cache_only_probability'] == 0
    assert model.config['current_only_probability'] == (.05 if robustness else 0)
    assert model.config['final_dropout'] == (.10 if robustness else 0)
    assert model.config['identity_probability'] == (.25 if robustness else 0)
    assert model.config['neighbors'] == variant.endswith('_aux')
    assert model.backbone.dc_final_writer is not None
    assert model.backbone.dcachehooping_latent_norm is not None
    for block in model.backbone.blocks:
        assert block.attention_mode == 'merged'
        assert block.merged_policy == 'current_preserving'
        assert block.step_memory_gate is None
        assert block.dc_qkv is None  # The normal attention projection is shared.
    assert not any('step_memory_gate' in name for name, _ in model.named_parameters())


def test_legacy_default_contract_is_unchanged(pilot, tmp_path):
    options = args(pilot, tmp_path / 'not-launched', 'both')
    options.merged_policy = 'legacy'
    supplied = runner.build_model_config(options, pilot[1])
    assert 'merged_policy' not in supplied
    expected = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(supplied)}
    model = ReasoningModel(supplied)
    assert json.dumps(model.config) == json.dumps(expected)
    explicit = ReasoningModel({**supplied, 'merged_policy': 'legacy'})
    assert json.dumps(explicit.config) == json.dumps(expected)
    assert model.config['gate_enabled']
    assert model.config['cache_only_probability'] == .20
    assert all(block.merged_policy == 'legacy' for block in model.backbone.blocks)
    explicit.load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize('change', [dict(gate_enabled=True),
                                    dict(cache_only_probability=.20),
                                    dict(attention_mode='vanilla'),
                                    dict(merged_policy='typo')])
def test_inconsistent_corrected_config_is_rejected(change):
    settings = config()
    settings.update(change)
    with pytest.raises(ValueError, match='current_preserving|merged_policy'):
        ReasoningModel(settings)


@pytest.mark.parametrize('neighbors', [False, True])
@pytest.mark.parametrize('identity_final_probability', [0., 1.])
def test_five_states_identity_sources_and_real_backward(neighbors, identity_final_probability,
                                                       monkeypatch):
    model = initialized_model(neighbors=neighbors, identity_probability=1.,
                              final_dropout=0., current_only_probability=.05,
                              identity_final_probability=identity_final_probability)
    original = model.backbone.forward
    calls = []

    def record(*inputs, **kwargs):
        calls.append((inputs[0], kwargs))
        assert kwargs['detach_cache_backbone'] is False
        if kwargs['previous_final_hidden'] is not None:
            assert not kwargs['previous_final_hidden'].requires_grad
            assert kwargs['previous_final_hidden'].grad_fn is None
        source = kwargs['step_memory_source_mask']
        if source is not None:
            assert set(source.unique().tolist()) <= {0, 2}
            assert not source[inputs[0].ne(model.mask_id)].any()
        return original(*inputs, **kwargs)

    monkeypatch.setattr(model.backbone, 'forward', record)
    loss, metrics = model.compute_loss(batch(), step=2000,
                                      generator=torch.Generator().manual_seed(71))
    assert len(calls) == 6
    assert calls[0][1]['previous_step_kv'] is None
    assert calls[0][1]['previous_final_hidden'] is None
    for _, kwargs in calls[1:5]:
        assert kwargs['previous_step_kv'] is not None
        assert kwargs['previous_final_hidden'] is not None
    assert metrics['trajectory_forwards'] == 5
    assert metrics['adjacent_edges'] == 4
    assert metrics['identity_forwards'] == 1
    assert metrics['identity_final'] == identity_final_probability
    for name in ('full', 't0', 't1', 't2', 't3'):
        assert metrics['cache_only_fraction_' + name] == 0
    assert bool(metrics['neighbor_loss'] > 0) == neighbors
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.backbone.blocks[0].attn_qkv.weight.grad.abs().sum() > 0
    assert model.backbone.dcachehooping_latent_norm.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('neighbors', [False, True])
@pytest.mark.parametrize('target', [1, 2, 3, 4])
def test_corrected_loss_trains_exactly_one_previous_step(target, neighbors):
    model = initialized_model(weights=[float(i == target) for i in range(5)],
                              neighbors=neighbors, identity_probability=0.,
                              current_only_probability=0., final_dropout=0.)
    states = []

    def retain(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden.retain_grad()
        states.append(hidden)

    hook = model.backbone.blocks[-1].register_forward_hook(retain)
    loss, metrics = model.compute_loss(batch(), generator=torch.Generator().manual_seed(7))
    hook.remove()
    assert metrics['adjacent_edges'] == 4
    loss.backward()
    assert len(states) == 5
    for index, hidden in enumerate(states):
        nonzero = hidden.grad is not None and bool(hidden.grad.abs().sum() > 0)
        assert nonzero == (index in [target-1, target]), (target, index)


def test_corrected_final_feedback_stays_detached():
    model = initialized_model()
    previous = torch.randn(2, 14, 32, requires_grad=True)
    values = batch()
    model(values['input_ids'], values['attention_mask'],
          previous_final_hidden=previous)['logits'][..., 4:].square().mean().backward()
    assert previous.grad is None
    assert model.backbone.dcachehooping_latent_norm.weight.grad.abs().sum() > 0


def test_aux_does_not_change_initial_backbone_or_corruption_rng():
    torch.manual_seed(811)
    plain = ReasoningModel(config(neighbors=False))
    torch.manual_seed(811)
    auxiliary = ReasoningModel(config(neighbors=True))
    auxiliary_state = auxiliary.state_dict()
    for name, value in plain.state_dict().items():
        torch.testing.assert_close(value, auxiliary_state[name], rtol=0, atol=0)
    ga, gb = torch.Generator().manual_seed(11), torch.Generator().manual_seed(11)
    _, am = plain.compute_loss(batch(), generator=ga)
    _, bm = auxiliary.compute_loss(batch(), generator=gb)
    for name in ('full', 't0', 't1', 't2', 't3'):
        torch.testing.assert_close(am['mask_ratio_' + name], bm['mask_ratio_' + name])
    assert torch.equal(ga.get_state(), gb.get_state())


def test_aux_uses_masked_neighbor_targets_not_masked_sources(monkeypatch):
    model = initialized_model(neighbors=True, identity_probability=0., final_dropout=0.)
    outputs = []
    original = model.forward

    def record(state, *args, **kwargs):
        result = original(state, *args, **kwargs)
        outputs.append((state, result))
        return result

    monkeypatch.setattr(model, 'forward', record)
    values = batch()
    loss, metrics = model.compute_loss(values, generator=torch.Generator().manual_seed(7))
    valid = values['attention_mask'].clone()
    for special in model.config['special_ids']:
        valid &= values['input_ids'].ne(special)
    state_losses, clean_sources = [], 0
    with torch.no_grad():
        for state, output in outputs:
            directions = []
            for direction, source, target in [('prev', slice(1, None), slice(None, -1)),
                                               ('next', slice(None, -1), slice(1, None))]:
                pairs = valid[:, source] & valid[:, target] & state[:, target].eq(model.mask_id)
                clean_sources += int((pairs & state[:, source].ne(model.mask_id)).sum())
                hidden = output['final_hidden'][:, source][pairs]
                logits = model.backbone.neighbor_heads.heads[direction](hidden)
                logits[:, model.mask_id] = -torch.inf
                directions.append(F.cross_entropy(logits, values['input_ids'][:, target][pairs])
                                  if pairs.any() else logits.new_zeros(()))
            state_losses.append(sum(directions) / 2)
    assert clean_sources > 0
    expected = sum(w * value for w, value in zip(model.config['weights'], state_losses)) / 2.05
    torch.testing.assert_close(metrics['neighbor_loss'], expected)
    torch.testing.assert_close(loss.detach(), metrics['base_loss'] + .5 * expected)


def assert_tree_equal(actual, expected):
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_tree_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize('policy', ['legacy', 'current_preserving'])
def test_resume_matches_uninterrupted_and_rejects_other_policy(pilot, tmp_path, policy):
    direct, resumed = tmp_path / 'direct', tmp_path / 'resumed'
    options = args(pilot, direct, steps=2)
    options.merged_policy = policy
    runner.train(options)
    options.run_dir, options.max_steps = str(resumed), 1
    runner.train(options)
    options.max_steps = 2
    runner.train(options)
    expected = runner.load_checkpoint(direct / 'checkpoints/last.pt')
    actual = runner.load_checkpoint(resumed / 'checkpoints/last.pt')
    for key in ('model', 'optimizer', 'rng_by_rank', 'contract'):
        assert_tree_equal(actual[key], expected[key])
    assert ('merged_policy' in actual['model_config']) == (policy != 'legacy')
    options.merged_policy = 'current_preserving' if policy == 'legacy' else 'legacy'
    options.max_steps = 3
    with pytest.raises(ValueError, match='contract differs'):
        runner.train(options)
    assert runner.load_checkpoint(resumed / 'checkpoints/last.pt')['step'] == 2


@pytest.mark.parametrize('variant', ['both', 'both_aux'])
def test_stress_is_explicit_recorded_and_cannot_resume_production(pilot, tmp_path, variant, capsys):
    options = args(pilot, tmp_path / variant, variant, 1, '--stress-memory-routes')
    settings = runner.build_model_config(options, pilot[1])
    assert settings['identity_probability'] == 1
    assert settings['final_dropout'] == settings['cache_only_probability'] == 0
    assert settings['source_dropout_warmup_steps'] == 0
    assert settings['current_only_probability'] == .05
    runner.train(options)
    checkpoint = runner.load_checkpoint(tmp_path / variant / 'checkpoints/last.pt')
    assert checkpoint['contract']['stress_memory_routes'] is True
    assert 'SMOKE STRESS ONLY' in capsys.readouterr().out
    options.stress_memory_routes = False
    options.max_steps = 2
    with pytest.raises(ValueError, match='contract differs'):
        runner.train(options)


@pytest.mark.parametrize('field,value', [('variant', 'mdm_aux'), ('merged_policy', 'legacy'),
                                       ('micro_batch', 1), ('no_robustness', True)])
def test_invalid_stress_combinations_fail_before_training(pilot, tmp_path, field, value):
    options = args(pilot, tmp_path / 'not-launched', 'both', 1, '--stress-memory-routes')
    setattr(options, field, value)
    with pytest.raises(ValueError, match='stress-memory-routes'):
        runner.build_model_config(options, pilot[1])


def test_plot_rejects_labels_that_do_not_match_run_count(tmp_path):
    options = runner.parser().parse_args([
        'plot', '--runs', str(tmp_path / 'one'), str(tmp_path / 'two'),
        '--labels', 'Only one label', '--output', str(tmp_path / 'plot.png')])
    with pytest.raises(ValueError, match='exactly one plot label per run'):
        runner.plot(options)
    assert not (tmp_path / 'plot.png').exists()


def test_plot_display_overrides_preserve_metric_values(tmp_path, monkeypatch):
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = tmp_path / 'both-h100-very-long-experiment-directory'
    writer = runner.MetricWriter(run / 'logs/attempt-1', resume_step=0)
    writer.log(500, {'train/base_loss': 2.5, 'train/loss': 3.0, 'val/conditional_nll': 2.2})
    writer.log(1000, {'train/base_loss': 2.0, 'train/loss': 2.5, 'val/conditional_nll': 1.8})
    options = runner.parser().parse_args([
        'plot', '--runs', str(run), '--labels', 'both', '--smooth', '1',
        '--train-metric', 'train/base_loss',
        '--val-label', 'Cold validation NLL (10/30/50/70% masks)',
        '--output', str(tmp_path / 'plot.png')])
    original = plt.subplots
    captured = []

    def record(*args, **kwargs):
        figure, axes = original(*args, **kwargs)
        captured.extend(axes)
        return figure, axes

    monkeypatch.setattr(plt, 'subplots', record)
    runner.plot(options)
    assert len(captured) == 2
    for axis, expected in zip(captured, ([2.5, 2.0], [2.2, 1.8])):
        assert len(axis.lines) == 1
        assert axis.lines[0].get_label() == 'both'
        assert axis.lines[0].get_xdata().tolist() == [500, 1000]
        assert axis.lines[0].get_ydata().tolist() == expected
    assert captured[0].get_ylabel() == 'train/base_loss (masked-answer CE)'
    assert captured[1].get_ylabel() == 'Cold validation NLL (10/30/50/70% masks)'
    assert (tmp_path / 'plot.png').exists()
