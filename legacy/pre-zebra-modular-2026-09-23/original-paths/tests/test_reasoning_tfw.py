import copy
import json

import pytest
import torch

from reasoning.tfw import PaperMDM, corrupt, decode, learning_rate, loss_terms, tail_mask, prediction_mask
from reasoning.tfw_runner import model_config, parser, train
from reasoning import runner
from reasoning.data import ReasoningDataset, write_prepared_dataset
from reasoning.benchmark import convert_zebra
from test_reasoning_benchmark import zebra_record
from test_reasoning_runner import assert_tree_equal, cpu_only


def config():
    return dict(family='tfw_gpt2_v1', vocab_size=12, mask_id=1, pad_id=0,
                bos_id=2, eos_id=4, n_positions=32, hidden_size=32,
                n_layers=2, n_heads=4, dropout=0., memory_mode='none')


def batch():
    return dict(input_ids=torch.tensor([[5, 3, 6, 4, 0, 0], [7, 3, 8, 4, 0, 0]]),
                target_mask=torch.tensor([[0, 0, 1, 1, 0, 0]] * 2, dtype=torch.bool),
                attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0]] * 2, dtype=torch.bool))


def test_bidirectional_future_context_shift_and_tied_weights():
    torch.manual_seed(1)
    model = PaperMDM(config()).eval()
    ids = torch.tensor([[5, 1, 1, 1, 6]])
    changed = ids.clone(); changed[0, -1] = 7
    with torch.no_grad():
        a, b = model(ids)['logits'], model(changed)['logits']
        hidden = model.gpt.transformer(ids, attention_mask=torch.ones_like(ids)).last_hidden_state
        raw = model.gpt.lm_head(hidden)
    assert torch.allclose(a[:, 1:], raw[:, :-1])
    assert torch.allclose(a[:, :1], raw[:, :1])
    assert not torch.allclose(a[:, 1], b[:, 1])  # Would fail with causal attention.
    assert model.gpt.lm_head.weight is model.gpt.transformer.wte.weight
    assert all(bool(layer.attn.bias.all()) for layer in model.gpt.transformer.h)


def test_no_shift_is_same_weights_and_same_position_only():
    torch.manual_seed(7)
    shifted = PaperMDM(config()).eval()
    torch.manual_seed(7)
    aligned = PaperMDM(dict(config(), logit_shift=0)).eval()
    assert_tree_equal(shifted.state_dict(), aligned.state_dict())
    ids = torch.tensor([[5, 1, 1, 1, 6]])
    with torch.no_grad():
        raw = aligned.gpt.lm_head(aligned.gpt.transformer(
            ids, attention_mask=torch.ones_like(ids)).last_hidden_state)
        a, b = shifted(ids)['logits'], aligned(ids)['logits']
    assert torch.equal(b, raw)
    assert torch.equal(a[:, 1:], b[:, :-1])
    assert torch.equal(a[:, :1], b[:, :1])
    assert 'logit_shift' not in shifted.config  # Old checkpoint convention.
    assert aligned.config['logit_shift'] == 0
    for value in (-1, 2, True):
        with pytest.raises(ValueError, match='logit_shift'):
            PaperMDM(dict(config(), logit_shift=value))


def test_corruption_masks_tail_padding_not_clues_or_gold_boundary():
    data = batch()
    state, masked, j, eligible = corrupt(data, 1, 1, torch.Generator().manual_seed(2))
    assert j.tolist() == [1, 1]
    assert torch.equal(masked, eligible)
    assert eligible.tolist() == [[False, False, True, True, True, True]] * 2
    assert torch.equal(state[:, :2], data['input_ids'][:, :2])
    assert state[:, 2:].eq(1).all()
    altered = copy.deepcopy(data); altered['input_ids'][:, 2:] = 9
    assert torch.equal(tail_mask(altered), eligible)


def test_weighted_loss_matches_upstream_formula_and_accumulation():
    torch.manual_seed(3)
    logits = torch.randn(2, 6, 12, requires_grad=True)
    data = batch()
    mask = tail_mask(data); mask[1, 4:] = False
    j = torch.tensor([2, 64])
    numerator, terms = loss_terms(logits, data['input_ids'], mask, j, data['target_mask'])
    reference = torch.nn.functional.cross_entropy(logits.reshape(-1, 12), data['input_ids'].reshape(-1),
                                                  reduction='none').reshape(2, 6)
    expected = (reference.masked_fill(~mask, 0) * (1 / j.float())[:, None]).sum() / mask.sum()
    assert torch.allclose(numerator / mask.sum(), expected)
    micro = sum(loss_terms(logits[i:i+1], data['input_ids'][i:i+1], mask[i:i+1], j[i:i+1],
                           data['target_mask'][i:i+1])[0] for i in range(2)) / mask.sum()
    assert torch.allclose(micro, expected)
    assert int(terms['answer_count']) == 4
    zero, _ = loss_terms(logits, data['input_ids'], torch.zeros_like(mask), j, data['target_mask'])
    zero.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.eq(0).all()


@pytest.mark.parametrize('policy', ['upstream_remask', 'paper_monotonic'])
def test_decoder_no_gold_leak_clues_fixed_and_budget(policy):
    class Stub:
        mask_id = 1
        def __init__(self): self.calls = []
        def __call__(self, ids):
            self.calls.append(ids.clone())
            logits = torch.zeros(*ids.shape, 12)
            logits[..., 9] = 5
            return {'logits': logits}
    data, altered = batch(), batch()
    altered['input_ids'][:, 2:] = 10
    model, other = Stub(), Stub()
    result = decode(model, data, torch.Generator().manual_seed(1), steps=5, policy=policy)
    compared = decode(other, altered, torch.Generator().manual_seed(1), steps=5, policy=policy)
    assert torch.equal(result, compared)
    assert len(model.calls) == 5 and model.calls[0][:, 2:].eq(1).all()
    assert all(torch.equal(x[:, :2], data['input_ids'][:, :2]) for x in model.calls)
    assert result[:, 2:].eq(9).all()


def test_lr_keeps_long_horizon_in_short_probe():
    assert learning_rate(0, .001, 2001300) == .001
    assert learning_rate(20013, .001, 2001300) == pytest.approx(.00099)
    assert learning_rate(2001300, .001, 2001300) == 0


@pytest.mark.parametrize('logit_shift', [0, 1])
@pytest.mark.parametrize('region,padding', [('padded_tail','visible'), ('answer','visible'), ('answer','masked')])
def test_full_state_resume_matches_uninterrupted_cpu(tmp_path, monkeypatch, cpu_only, logit_shift, region, padding):
    records = [convert_zebra(zebra_record(i)) for i in range(6)]
    data = tmp_path / 'data'
    write_prepared_dataset(data, 'zebra-benchmark',
                           dict(train=records[:4], validation=records[4:5], test=records[5:]), 17, {})
    monkeypatch.setattr('reasoning.tfw_runner.generation_report', lambda *a, **kw: None)
    def args(directory, stop):
        return parser().parse_args([
            '--data-dir', str(data), '--run-dir', str(directory), '--debug', '--device', 'cpu',
            '--logit-shift', str(logit_shift),
            '--target-region', region, '--padding-attention', padding,
            '--precision', 'fp32', '--global-batch', '2', '--micro-batch', '1',
            '--stop-after-steps', str(stop), '--cpu-threads', '1', '--validation-examples', '1',
            '--eval-batch-size', '1', '--save-every', '1', '--log-every', '1'])
    direct, resumed = tmp_path / 'direct', tmp_path / 'resumed'
    train(args(direct, 3)); train(args(resumed, 1)); train(args(resumed, 3))
    a, b = (runner.load_checkpoint(p / 'checkpoints/last.pt') for p in (direct, resumed))
    for key in ('model', 'optimizer', 'rng_by_rank', 'contract', 'step', 'examples_seen'):
        assert_tree_equal(a[key], b[key])
    assert b['step'] == 3 and b['examples_seen'] == 6
    changed = args(resumed, 4); changed.lr = .0003
    with pytest.raises(ValueError, match='contract mismatch'):
        train(changed)
    wrong_alignment = args(resumed, 4); wrong_alignment.logit_shift = 1 - logit_shift
    with pytest.raises(ValueError, match='contract mismatch'):
        train(wrong_alignment)
    wrong_region = args(resumed, 4)
    wrong_region.target_region = 'answer' if region == 'padded_tail' else 'padded_tail'
    wrong_region.padding_attention = 'visible'
    with pytest.raises(ValueError, match='contract mismatch'):
        train(wrong_region)


@pytest.mark.parametrize('logit_shift', [0, 1])
@pytest.mark.parametrize('region,padding', [('padded_tail','visible'), ('answer','masked')])
def test_real_generation_report_all_three_decoders(tmp_path, cpu_only, logit_shift, region, padding):
    from reasoning.tfw_runner import generation_report
    torch.set_num_threads(1)
    records = [convert_zebra(zebra_record(i)) for i in range(3)]
    data = tmp_path / 'data'
    write_prepared_dataset(data, 'zebra-benchmark',
                           dict(train=records[:1], validation=records[1:2], test=records[2:]), 17, {})
    dataset = ReasoningDataset(data, 'test')
    model = PaperMDM(model_config(dataset, debug=True, logit_shift=logit_shift,
                                  target_region=region, padding_attention=padding))
    optimizer = torch.optim.AdamW(model.parameters())
    run = tmp_path / 'run'
    runner.save_checkpoint(run, model, optimizer, 1, {'global_batch': 128}, 0, 1)
    args = parser().parse_args(['--data-dir', str(data), '--run-dir', str(run), '--precision', 'fp32',
                               '--eval-batch-size', '1', '--generation-examples', '1'])
    generation_report(model, dataset, args, torch.device('cpu'), 1, run, 'test')
    report = json.loads((run / 'generation/test-step-000000001.json').read_text())
    assert set(report['reports']) == {'upstream_remask', 'paper_monotonic', 'matched_candidate8'}
    assert report['num_examples'] == 1
    for part in report['reports'].values():
        assert len(part['examples']) == 1 and 0 <= part['metrics']['valid_solution'] <= 1
    assert model.training


def test_answer_only_corruption_matches_answer_masks_but_excludes_padding():
    data = batch()
    old = corrupt(data, 1, 64, torch.Generator().manual_seed(31))
    new = corrupt(data, 1, 64, torch.Generator().manual_seed(31), 'answer')
    assert torch.equal(old[2], new[2])
    assert torch.equal(old[1] & data['target_mask'], new[1])
    full, masked, _, eligible = corrupt(data, 1, 1, torch.Generator().manual_seed(3), 'answer')
    assert torch.equal(masked, data['target_mask'])
    assert torch.equal(eligible, data['target_mask'])
    assert torch.equal(full[~masked], data['input_ids'][~masked])
    altered = copy.deepcopy(data); altered['input_ids'][:, 2:4] = 9
    assert torch.equal(prediction_mask(altered, 'answer'), eligible)
    logits = torch.randn(2, 6, 12, requires_grad=True)
    numerator, terms = loss_terms(logits, data['input_ids'], masked, torch.ones(2), data['target_mask'])
    numerator.backward()
    assert logits.grad[~masked].eq(0).all()
    assert terms['padding_count'] == 0
    assert terms['answer_count'] == masked.sum()


def test_padding_options_do_not_change_initial_weights():
    torch.manual_seed(19)
    old = PaperMDM(dict(config(), logit_shift=0))
    torch.manual_seed(19)
    new = PaperMDM(dict(config(), logit_shift=0, target_region='answer', padding_attention='masked'))
    assert_tree_equal(old.state_dict(), new.state_dict())


@pytest.mark.parametrize('shift', [0, 1])
def test_masked_padding_cannot_affect_real_position_logits(shift):
    model = PaperMDM(dict(config(), logit_shift=shift, target_region='answer', padding_attention='masked')).eval()
    data = batch(); altered = data['input_ids'].clone(); altered[:, 4:] = 9
    with torch.no_grad():
        first = model(data['input_ids'], attention_mask=data['attention_mask'])['logits']
        second = model(altered, attention_mask=data['attention_mask'])['logits']
    torch.testing.assert_close(first[:, :4], second[:, :4], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match='public attention_mask'):
        model(data['input_ids'])


@pytest.mark.parametrize('policy', ['upstream_remask', 'paper_monotonic'])
def test_answer_only_decode_uses_public_canvas_not_gold_and_keeps_padding(policy):
    class Stub:
        mask_id = 1
        config = dict(target_region='answer', padding_attention='masked')
        def __init__(self): self.calls = []
        def __call__(self, ids, attention_mask):
            self.calls.append((ids.clone(), attention_mask.clone()))
            logits = torch.zeros(*ids.shape, 12); logits[..., 9] = 5
            return dict(logits=logits)
    data, altered = batch(), batch()
    altered['input_ids'][:, 2:4] = 10
    model, other = Stub(), Stub()
    a = decode(model, data, torch.Generator().manual_seed(1), steps=5, policy=policy)
    b = decode(other, altered, torch.Generator().manual_seed(1), steps=5, policy=policy)
    assert torch.equal(a, b)
    assert a[:, 2:4].eq(9).all() and a[:, 4:].eq(0).all()
    for ids, attention in model.calls:
        assert torch.equal(ids[:, :2], data['input_ids'][:, :2])
        assert ids[:, 4:].eq(0).all()
        assert torch.equal(attention, data['attention_mask'])


def test_overfit_diagnostic_resume_and_provenance(tmp_path, cpu_only):
    records = [convert_zebra(zebra_record(i)) for i in range(6)]
    data = tmp_path / 'data'
    write_prepared_dataset(data, 'zebra-benchmark',
                           dict(train=records[:4], validation=records[4:5], test=records[5:]), 17, {})
    run = tmp_path / 'overfit'
    args = parser().parse_args(['--data-dir', str(data), '--run-dir', str(run), '--device', 'cpu',
        '--precision', 'fp32', '--debug', '--overfit-examples', '2', '--stop-after-steps', '1',
        '--target-region', 'answer', '--padding-attention', 'masked', '--global-batch', '2',
        '--micro-batch', '1', '--validation-examples', '1', '--eval-batch-size', '2', '--cpu-threads', '1'])
    train(args)
    result = json.loads((run/'sanity/step-000000001.json').read_text())
    checkpoint = runner.load_checkpoint(run/'checkpoints/last.pt')
    assert result['split'] == 'training_memorization_diagnostic'
    assert result['puzzles'] == 2
    assert checkpoint['contract']['overfit_train_indices'] == [1, 2]
    assert checkpoint['contract']['overfit_cursor_version'] == 2
    # Full data has two updates/epoch; subset one. Its cursor must keep
    # advancing AFTER 300 tiny epochs while the LR horizon is 600 updates.
    assert checkpoint['contract']['epochs'] == 600
    assert runner.examples_at_step(400, checkpoint['contract']) == 800
    assert not (run/'generation').exists()  # Never mislabel training as test.
    args.stop_after_steps = 2
    train(args)
    assert runner.load_checkpoint(run/'checkpoints/last.pt')['examples_seen'] == 4
    val = json.loads((run/'validation/step-000000002.json').read_text())
    assert val['full_mask_diagnostic']['mask_ratio'] == 1.
    assert len(val['ratios']) == 4  # Historical macro-NLL unchanged.


def test_explicit_final_generation_at_non_epoch_stop(tmp_path, monkeypatch, cpu_only):
    records = [convert_zebra(zebra_record(i)) for i in range(6)]
    data = tmp_path/'data'
    write_prepared_dataset(data, 'zebra-benchmark',
                           dict(train=records[:4], validation=records[4:5], test=records[5:]), 17, {})
    calls = []
    monkeypatch.setattr('reasoning.tfw_runner.generation_report',
                        lambda model, data, args, device, step, run, split: calls.append((step, split)))
    args = parser().parse_args(['--data-dir', str(data), '--run-dir', str(tmp_path/'run'),
        '--debug', '--device', 'cpu', '--precision', 'fp32', '--global-batch', '2', '--micro-batch', '1',
        '--stop-after-steps', '3', '--final-generation', '--validation-examples', '1',
        '--eval-batch-size', '1', '--cpu-threads', '1'])
    train(args)
    assert calls == [(2, 'test'), (3, 'test')]
    train(args)  # Retry a generation interrupted after checkpoint commit.
    assert calls[-1] == (3, 'test')
