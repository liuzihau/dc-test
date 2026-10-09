import copy

import pytest
import torch
from torch.utils.data import default_collate

from reasoning.benchmark import convert_zebra
from reasoning.data import ReasoningDataset, TaskTokenizer, encode_record, write_prepared_dataset
from reasoning.tfw import PaperMDM, decode
from reasoning.tfw_runner import model_config, parser, train
from reasoning.zebra_encoding import audit_public_dimensions, public_features
from reasoning import runner
from test_reasoning_benchmark import zebra_record
from test_reasoning_runner import assert_tree_equal, cpu_only


def encoded(h=3, a=3):
    record = convert_zebra(zebra_record(h=h, a=a))
    tok = TaskTokenizer('zebra-benchmark')
    clean, prefix, used = encode_record(record, tok, 384)
    positions = torch.arange(384)
    batch = dict(input_ids=torch.tensor(clean)[None, :], attention_mask=(positions < used)[None, :],
                 target_mask=((positions >= prefix) & (positions < used))[None, :])
    c, n = tok.encode(['c', 'n'])
    return batch, dict(sep_id=tok.sep_id, c_id=c, n_id=n, digit_ids=tok.encode(list('012345')))


@pytest.mark.parametrize('h,a', [(h, a) for h in range(3, 7) for a in range(3, 7)])
def test_public_dimensions_coords_all_sizes_no_gold_features(h, a):
    batch, public = encoded(h, a)
    ids = batch['input_ids'].masked_fill(batch['target_mask'], 1)
    f = public_features(ids, batch['attention_mask'], **public)
    assert f['houses'].item() == h and f['attributes'].item() == a
    assert torch.equal(f['target_mask'], batch['target_mask'])
    assert f['value'][batch['target_mask']].eq(0).all()
    assert f['position_ids'][batch['target_mask']].tolist() == list(range(384, 385+h*a))
    assert f['attribute'][batch['target_mask']].tolist() == [i+1 for i in range(a) for _ in range(h)] + [0]
    assert f['house'][batch['target_mask']].tolist() == list(range(1, h+1))*a + [0]
    altered = batch['input_ids'].clone(); altered[batch['target_mask']] = public['sep_id']
    g = public_features(altered, batch['attention_mask'], **public)
    for key in ('position_ids', 'attribute', 'house', 'role', 'houses', 'attributes', 'target_mask'):
        assert torch.equal(f[key], g[key])  # Cannot inspect any gold answer values.


def test_relative_positions_stay_fixed_when_clue_prefix_length_changes():
    batch, public = encoded()
    ids, attention = batch['input_ids'], batch['attention_mask']
    # Duplicate one public clue at the start. No answer content is involved.
    tok = TaskTokenizer('zebra-benchmark')
    end = ids[0].tolist().index(tok.encode(['CLUE_END'])[0]) + 1
    changed = torch.cat((ids[:, :end], ids[:, :-end]), 1)
    changed_attention = torch.cat((attention[:, :end], attention[:, :-end]), 1)
    a = public_features(ids, attention, **public)
    b = public_features(changed, changed_attention, **public)
    for key in ('position_ids', 'attribute', 'house', 'value', 'role'):
        assert torch.equal(a[key][a['target_mask']], b[key][b['target_mask']])


@pytest.mark.parametrize('encoding', ['answer_relative', 'typed_coordinates'])
def test_encoding_real_forward_backward_decode_and_resume(tmp_path, monkeypatch, cpu_only, encoding):
    records = [convert_zebra(zebra_record(i)) for i in range(6)]
    path = tmp_path/'data'
    write_prepared_dataset(path, 'zebra-benchmark', dict(train=records[:4], validation=records[4:5], test=records[5:]), 17, {})
    dataset = ReasoningDataset(path)
    assert audit_public_dimensions(dataset)['rows'] == 4
    config = model_config(dataset, True, 0, 'answer', 'masked', encoding)
    model = PaperMDM(config).eval()
    batch = default_collate([dataset[0], dataset[1]])
    a = decode(model, batch, torch.Generator().manual_seed(1), steps=3, policy='paper_monotonic')
    other = copy.deepcopy(batch); other['input_ids'][other['target_mask']] = dataset.tokenizer.sep_id
    b = decode(model, other, torch.Generator().manual_seed(1), steps=3, policy='paper_monotonic')
    assert torch.equal(a, b)
    inputs = batch['input_ids'].masked_fill(batch['target_mask'], model.mask_id)
    model(inputs, attention_mask=batch['attention_mask'])['logits'].square().mean().backward()
    if encoding == 'typed_coordinates':
        assert model.coordinates.attribute.weight.grad.abs().sum() > 0
        assert model.coordinates.house.weight.grad.abs().sum() > 0
    monkeypatch.setattr('reasoning.tfw_runner.generation_report', lambda *a, **kw: None)
    def args(name, stop):
        return parser().parse_args(['--data-dir', str(path), '--run-dir', str(tmp_path/name),
            '--debug', '--device', 'cpu', '--precision', 'fp32', '--cpu-threads', '1', '--logit-shift', '0',
            '--target-region', 'answer', '--padding-attention', 'masked', '--zebra-encoding', encoding,
            '--global-batch', '2', '--micro-batch', '1', '--validation-examples', '1', '--eval-batch-size', '1',
            '--stop-after-steps', str(stop)])
    train(args('direct', 3)); train(args('resume', 1)); train(args('resume', 3))
    a, b = [runner.load_checkpoint(tmp_path / name / 'checkpoints/last.pt') for name in ('direct','resume')]
    for key in ('model', 'optimizer', 'rng_by_rank', 'contract', 'examples_seen', 'step'):
        assert_tree_equal(a[key], b[key])


def test_typed_coordinates_do_not_change_base_initialization(tmp_path, cpu_only):
    record = convert_zebra(zebra_record())
    write_prepared_dataset(tmp_path, 'zebra-benchmark', dict(train=[record], validation=[], test=[]), 17, {})
    dataset = ReasoningDataset(tmp_path)
    torch.manual_seed(1)
    plain = PaperMDM(model_config(dataset, True, 0, 'answer', 'masked', 'answer_relative'))
    torch.manual_seed(1)
    typed = PaperMDM(model_config(dataset, True, 0, 'answer', 'masked', 'typed_coordinates'))
    assert_tree_equal(plain.gpt.state_dict(), typed.gpt.state_dict())
    assert sum(p.numel() for p in typed.parameters()) - sum(p.numel() for p in plain.parameters()) == 25*32


def test_queue_commands_gates_and_predecessor_require_all_artifacts(tmp_path):
    import json
    from reasoning.zebra_encoding_queue import predecessor_complete, sanity_passed, train_command
    assert not predecessor_complete(tmp_path)
    (tmp_path/'generation').mkdir()
    (tmp_path/'status.json').write_text(json.dumps(dict(status='paused',step=20013)))
    (tmp_path/'generation/test-step-000020013.json').write_text(json.dumps(dict(step=20013)))
    assert not predecessor_complete(tmp_path)
    (tmp_path/'clue_audit.json').write_text(json.dumps(dict(step=20013)))
    assert predecessor_complete(tmp_path)
    cmd = train_command('typed_coordinates', tmp_path/'run', 40026)
    assert '--fork-from' not in cmd and '--overfit-examples' not in cmd
    assert cmd[cmd.index('--micro-batch')+1] == '32'
    assert cmd[cmd.index('--full-mask-probability')+1] == '0'
    assert cmd[cmd.index('--zebra-encoding')+1] == 'typed_coordinates'
    assert not sanity_passed(tmp_path)
    (tmp_path/'sanity').mkdir()
    path=tmp_path/'sanity/step-000001500.json'
    path.write_text(json.dumps(dict(split='training_memorization_diagnostic', puzzles=32,
                                   greedy_rollout_exact=1., greedy_rollout_pad_fraction=0.)))
    assert sanity_passed(tmp_path)
