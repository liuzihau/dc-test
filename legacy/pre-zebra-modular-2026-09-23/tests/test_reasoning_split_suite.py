"""Split capacity controls, padded recurrence and finite distributed tails."""
import math
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
from types import SimpleNamespace

import pytest
import torch

from reasoning.model import ReasoningModel
from reasoning.runner import build_model_config, distributed_microbatch, parser
from reasoning.variants import SPLIT_VARIANTS, split_variant_config
from reasoning.full_epoch_queue import split_training_command
from test_reasoning_model import config, batch, nonzero_head, small_threads


@pytest.mark.parametrize('variant', SPLIT_VARIANTS)
def test_split_controls_backward_roundtrip_and_no_history(variant):
    settings = split_variant_config(variant)
    model = nonzero_head(ReasoningModel(config(**settings, identity_probability=1.,
                                             final_dropout=0.)))
    loss, metrics = model.compute_loss(batch(), step=1500,
                                      generator=torch.Generator().manual_seed(7))
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics['trajectory_forwards'] == (1 if variant == 'mdm' else 5)
    assert metrics['adjacent_edges'] == (4 if '_rm' in variant else 0)
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    if '_ea' in variant:
        assert model.backbone.blocks[0].dc_qkv.weight.grad.abs().sum() > 0
    else:
        assert model.backbone.blocks[0].dc_qkv is None
    if '_rm' not in variant:
        assert model.backbone.dc_final_writer is None
        assert not model.has_final
        assert not model(batch()['input_ids'], batch()['attention_mask'])['step_kv']
    clone = ReasoningModel(model.config)
    clone.load_state_dict(model.state_dict(), strict=True)
    model.eval(); clone.eval()
    with torch.no_grad():
        a = model(batch()['input_ids'], batch()['attention_mask'])['logits']
        b = clone(batch()['input_ids'], batch()['attention_mask'])['logits']
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize('source', [0, 1, 2])
def test_split_padding_excludes_previous_and_current_keys(source):
    model = nonzero_head(ReasoningModel(config(attention_mode='separate', memory_mode='both'))).eval()
    b = batch(); valid = b['attention_mask']
    first = model(b['input_ids'], valid)
    cache = first['step_kv']
    source_mask = torch.full_like(b['input_ids'], source)
    with torch.no_grad():
        reference = model(b['input_ids'], valid, cache, first['final_hidden'], source_mask=source_mask)
        changed = b['input_ids'].clone(); changed[~valid] = 16
        for entry in cache:
            entry[~valid] = 10000
        candidate = model(changed, valid, cache, first['final_hidden'], source_mask=source_mask)
    torch.testing.assert_close(reference['logits'][valid], candidate['logits'][valid], rtol=0, atol=0)


def test_adjacent_reaches_previous_backbone_but_final_is_detached():
    model = nonzero_head(ReasoningModel(config(attention_mode='separate', memory_mode='both')))
    b = batch()
    first = model(b['input_ids'], b['attention_mask'])
    final = first['final_hidden'].detach().requires_grad_()
    second = model(b['input_ids'], b['attention_mask'], first['step_kv'], final)
    grad_cache, grad_final = torch.autograd.grad(second['logits'][..., 5].sum(),
                                [first['step_kv'][0], final], allow_unused=True)
    assert grad_cache.abs().sum() > 0
    assert grad_final is None


@pytest.mark.parametrize('n', [1, 3, 7, 17, 127, 128])
def test_distributed_tail_exact_coverage_and_mean_gradient(n):
    seen = []; total = 0.
    for micro in range(math.ceil(n / 16)):
        for rank in range(2):
            offset, count, weight = distributed_microbatch(n, 8, 2, rank, micro)
            seen.extend(range(offset, offset + count))
            total += weight / 2  # DDP averages the weighted local gradients.
    assert seen == list(range(n))
    assert total == pytest.approx(1.)


def test_new_and_legacy_mdm_have_unambiguous_contracts():
    dataset = SimpleNamespace(max_length=16, tokenizer=SimpleNamespace(
        vocab_size=17, pad_id=0, mask_id=1, special_ids=[0, 1]))
    args = parser().parse_args(['train', '--task', 'sudoku-benchmark', '--variant', 'mdm',
                               '--data-dir', 'unused', '--run-dir', 'unused'])
    assert build_model_config(args, dataset)['trajectory'] == 'five'
    args.suite = 'split'
    assert build_model_config(args, dataset)['trajectory'] == 'single'
    for variant in SPLIT_VARIANTS:
        args.variant = variant
        model = build_model_config(args, dataset)
        assert model['attention_mode'] == ('separate' if '_ea' in variant else 'vanilla')
        assert model['memory_mode'] == ('both' if '_rm' in variant else 'none')


def test_split_command_uses_two_ranks_and_same_global_batch():
    command = split_training_command('sudoku-benchmark', 'tt_ea_rm_np', 'data', 'out', 1000)
    for flag, value in [('--nproc_per_node', '2'), ('--micro-batch', '8'),
                        ('--global-batch', '128'), ('--suite', 'split'), ('--eval-batch-size', '8')]:
        assert command[command.index(flag) + 1] == value
    assert '--one-epoch' in command
    assert '--merged-policy' not in command


@pytest.mark.parametrize('variant', ['mdm', 'tt_ea_np', 'tt_ea_rm_np'])
def test_real_two_rank_epoch_tail_and_resume(tmp_path, variant):
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            if probe.getsockname()[1] == 0:
                pytest.skip('Sandbox does not expose a usable local rendezvous port')
    except PermissionError:
        pytest.skip('Local rendezvous sockets unavailable; run this integration test outside the sandbox')
    from reasoning.data import prepare_dataset
    from reasoning.runner import load_checkpoint
    data, run = tmp_path / 'data', tmp_path / 'run'
    prepare_dataset(data, 'countdown', train_size=9, valid_size=1, test_size=1, seed=48)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2',
        'scripts/reasoning/run_reasoning.py', 'train', '--suite', 'split', '--variant', variant,
        '--task', 'countdown', '--data-dir', str(data), '--run-dir', str(run), '--size', 'debug',
        '--device', 'cpu', '--precision', 'fp32', '--global-batch', '8', '--micro-batch', '2',
        '--one-epoch', '--val-every', '2', '--validation-examples', '1', '--eval-batch-size', '1',
        '--save-every', '1', '--log-every', '1', '--cpu-threads', '1', '--validation-protocol', 'both']
    if '_rm' in variant:
        command += ['--stress-memory-routes']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
        env.pop(key, None)
    for _ in range(2):  # Completed run must not add another epoch on restart.
        result = subprocess.run(command, env=env, cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        checkpoint = load_checkpoint(run / 'checkpoints/last.pt')
        assert checkpoint['step'] == 2 and checkpoint['examples_seen'] == 9
        assert len(checkpoint['rng_by_rank']) == 2
        assert json.loads((run / 'status.json').read_text())['status'] == 'finished'
