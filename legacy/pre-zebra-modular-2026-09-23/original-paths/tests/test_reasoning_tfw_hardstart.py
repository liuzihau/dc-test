import copy
import json

import pytest
import torch

from reasoning import runner
from reasoning.benchmark import convert_zebra
from reasoning.data import TaskTokenizer, write_prepared_dataset
from reasoning.tfw import corrupt, full_mask_mixture, loss_terms
from reasoning.tfw_clue_audit import permutation_marginal, shuffle_clue_roles
from reasoning.tfw_runner import parser, train
from test_reasoning_benchmark import zebra_record
from test_reasoning_runner import assert_tree_equal, cpu_only
from test_reasoning_tfw import batch


def test_mixture_zero_preserves_draws_and_exact_control():
    generator = torch.Generator().manual_seed(19)
    a = corrupt(batch(), 1, 64, generator, 'answer')
    state = generator.get_state().clone()
    inputs, mask, j, forced = full_mask_mixture(*a, 1, generator, 0.)
    for left, right in zip((inputs, mask, j), a[:3]):
        assert torch.equal(left, right)
    assert torch.equal(state, generator.get_state()) and not forced.any()


def test_mixture_full_never_reads_gold_or_changes_clues_padding():
    data = batch()
    a = corrupt(data, 1, 64, torch.Generator().manual_seed(19), 'answer')
    inputs, mask, j, forced = full_mask_mixture(*a, 1, torch.Generator().manual_seed(8), 1.)
    assert torch.equal(mask, data['target_mask']) and j.eq(64).all() and forced.all()
    assert inputs[mask].eq(1).all()
    assert torch.equal(inputs[~mask], data['input_ids'][~mask])
    changed = copy.deepcopy(data); changed['input_ids'][changed['target_mask']] = 10
    b = corrupt(changed, 1, 64, torch.Generator().manual_seed(19), 'answer')
    assert torch.equal(inputs, full_mask_mixture(*b, 1, torch.Generator().manual_seed(8), 1.)[0])
    logits = torch.randn(2, 6, 12, requires_grad=True)
    numerator, _ = loss_terms(logits, data['input_ids'], mask, j, data['target_mask'])
    denominator = a[1].sum().clamp_min(1)
    expected = torch.nn.functional.cross_entropy(logits[mask], data['input_ids'][mask], reduction='sum') / 64 / denominator
    torch.testing.assert_close(numerator / denominator, expected)
    (numerator / denominator).backward()
    assert logits.grad[~mask].eq(0).all()


def test_mixture_probability_and_unforced_masks():
    data = {k: v.repeat(5000, 1) for k, v in batch().items()}
    a = corrupt(data, 1, 64, torch.Generator().manual_seed(19), 'answer')
    inputs, mask, j, forced = full_mask_mixture(*a, 1, torch.Generator().manual_seed(8), .5)
    assert .48 < forced.float().mean() < .52
    for left, right in zip((inputs, mask, j), a[:3]):
        assert torch.equal(left[~forced], right[~forced])
    assert mask[forced].eq(a[3][forced]).all()
    for probability in (-1, 2, float('nan')):
        with pytest.raises(ValueError):
            full_mask_mixture(*a, 1, torch.Generator(), probability)


def test_clue_blind_probabilities_and_clue_shuffle_leave_targets_untouched():
    tok = TaskTokenizer('zebra-benchmark')
    ids = torch.tensor(tok.encode(['=', 'LHS', 'c', '0', '1', 'RHS', 'n', '0', '2', 'CLUE_END',
                                   '[SEP]', '0', '[MASK]', '2', '[EOS]', '[PAD]']))
    target = torch.zeros_like(ids, dtype=torch.bool); target[11:15] = True
    masked = ids.eq(tok.mask_id)
    nll, acc, count = permutation_marginal(ids, masked, target, 3, 1, tok)
    assert nll == 0 and acc == 1 and count == 1
    altered = shuffle_clue_roles(ids, target, tok, 1)
    assert torch.equal(altered[target], ids[target])
    full = target & ids.ne(tok.eos_id)
    nll, acc, count = permutation_marginal(ids.masked_fill(full, tok.mask_id), full, target, 3, 1, tok)
    assert nll == pytest.approx(3 * __import__('math').log(3)) and acc == 1 and count == 3


@pytest.mark.parametrize('probability', [0., .5])
def test_fork_full_state_resume_and_control_equivalence(tmp_path, monkeypatch, cpu_only, probability):
    records = [convert_zebra(zebra_record(i)) for i in range(6)]
    data = tmp_path / 'data'
    write_prepared_dataset(data, 'zebra-benchmark',
        dict(train=records[:4], validation=records[4:5], test=records[5:]), 17, {})
    monkeypatch.setattr('reasoning.tfw_runner.generation_report', lambda *a, **kw: None)
    def args(name, stop, fork=None, p=0.):
        result = parser().parse_args(['--data-dir', str(data), '--run-dir', str(tmp_path / name),
            '--debug', '--device', 'cpu', '--precision', 'fp32', '--cpu-threads', '1',
            '--logit-shift', '0', '--target-region', 'answer', '--padding-attention', 'masked',
            '--global-batch', '2', '--micro-batch', '1', '--validation-examples', '1',
            '--eval-batch-size', '1', '--stop-after-steps', str(stop), '--log-every', '1', '--save-every', '1',
            '--full-mask-probability', str(p)])
        result.fork_from = str(fork) if fork else None
        return result
    train(args('parent', 1))
    parent = (tmp_path / 'parent/checkpoints/last.pt').resolve()
    parent_hash = runner.digest(parent)
    train(args('direct', 3, parent, probability))
    train(args('resumed', 2, parent, probability))
    train(args('resumed', 3, p=probability))  # No parent path needed for own resume.
    a, b = (runner.load_checkpoint(tmp_path / name / 'checkpoints/last.pt') for name in ('direct', 'resumed'))
    for key in ('model', 'optimizer', 'rng_by_rank', 'contract', 'examples_seen', 'step'):
        assert_tree_equal(a[key], b[key])
    assert b['examples_seen'] == 6 and b['contract']['fork']['step'] == 1
    assert runner.digest(parent) == parent_hash
    if probability == 0:
        train(args('parent', 3))
        old = runner.load_checkpoint(tmp_path / 'parent/checkpoints/last.pt')
        for key in ('model', 'optimizer', 'rng_by_rank', 'examples_seen', 'step'):
            assert_tree_equal(a[key], old[key])
    wrong = args('bad', 4, parent, probability); wrong.lr = .0003
    with pytest.raises(ValueError, match='Fork contract differs'):
        train(wrong)
    wrong = args('resumed', 4, p=.75)
    with pytest.raises(ValueError, match='contract mismatch'):
        train(wrong)
