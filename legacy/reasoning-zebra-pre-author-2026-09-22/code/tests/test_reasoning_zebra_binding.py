import copy
from argparse import Namespace
import json
import random

import pytest
import torch

from reasoning.data import ReasoningDataset, encode_record
from reasoning.tfw_runner import parser, train
from reasoning.zebra_binding import (EQUALITY_FAMILIES, EQUALITY_KIND, FAMILIES,
                                     TRAIN_FAMILIES, certified_solution, evaluate,
                                     prepare, prepare_equality_control, render)
from reasoning.zebra_binding_queue import train_command
from reasoning.zebra_equality_queue import train_command as equality_train_command
from reasoning.zebra_encoding import audit_public_dimensions
from reasoning.tasks import TaskTokenizer
from reasoning.zebra_official import parse_prompt
from test_reasoning_runner import cpu_only, assert_tree_equal
from reasoning.runner import load_checkpoint


@pytest.mark.parametrize('family', FAMILIES)
def test_clues_uniquely_determine_gold_without_reading_answer(family):
    rng = random.Random(94)
    tok = TaskTokenizer('zebra-benchmark')
    for _ in range(12):
        board = [rng.sample(range(5), 5) for _ in range(5)]
        record = render(board, family, rng)
        assert certified_solution(record) == record['answer']
        assert encode_record(record, tok, 384)[2] == 277
        poisoned = copy.deepcopy(record); poisoned['answer'] = ['wrong'] * 25
        assert certified_solution(poisoned) == record['answer']


def test_direct_equality_is_really_direct_and_graphs_are_distinct():
    board = [random.Random(100+a).sample(range(5), 5) for a in range(5)]
    by_family = {}
    for family in EQUALITY_FAMILIES:
        record = render(board, family, random.Random(22))
        _, _, clues = parse_prompt(['HOUSES', '5', 'ATTRS', '5'] + record['prompt'])
        forms = [(kind, tuple(ref[0] for ref in refs)) for kind, refs in clues]
        by_family[family] = forms
        assert len(clues) == 25 and set(kind for kind, _ in forms) == {'='}
    assert all(set(roles) == {'c', 'n'} for _, roles in by_family['direct_equality'])
    for family in ('equality_star', 'equality_chain'):
        assert sum(set(roles) == {'c', 'n'} for _, roles in by_family[family]) == 5
        assert sum(roles == ('c', 'c') for _, roles in by_family[family]) == 20


def test_preparation_board_disjoint_and_paired_eval(tmp_path):
    path = tmp_path/'data'
    manifest = prepare(path, 8, 3, 4, 19)
    assert manifest['source']['benchmark_equivalence'] is False
    sets = []
    for split, boards, families in (('train', 8, TRAIN_FAMILIES), ('validation', 3, FAMILIES), ('test', 4, FAMILIES)):
        data = ReasoningDataset(path, split)
        assert len(data) == boards*len(families)
        hashes = {r['metadata']['board_sha256'] for r in data.records}
        sets.append(hashes)
        assert len(hashes) == boards
        for key in hashes:
            matched = [r for r in data.records if r['metadata']['board_sha256']==key]
            assert {r['metadata']['family'] for r in matched} == set(families)
            assert len({tuple(r['answer']) for r in matched}) == 1
        assert audit_public_dimensions(data)['rows'] == len(data)
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert prepare(path, 8, 3, 4, 19) == manifest
    with pytest.raises(ValueError, match='recipe differs'):
        prepare(path, 8, 3, 4, 20)


def test_no_epoch_generation_is_opt_in_and_queue_is_fresh(tmp_path):
    args = parser().parse_args(['--data-dir', str(tmp_path), '--run-dir', str(tmp_path)])
    assert not args.no_epoch_end_generation
    cmd = train_command('typed_coordinates', tmp_path/'run', 3000)
    assert '--no-epoch-end-generation' in cmd and '--fork-from' not in cmd
    assert cmd[cmd.index('--stop-after-steps')+1] == '3000'
    assert cmd[cmd.index('--global-batch')+1] == '128'
    cmd = equality_train_command('typed_coordinates', tmp_path/'eq', 1500)
    assert '--fork-from' not in cmd and '--no-epoch-end-generation' in cmd
    assert cmd[cmd.index('--stop-after-steps')+1] == '1500'


def test_equality_control_preparation_is_separate_and_board_disjoint(tmp_path):
    path = tmp_path/'eq'
    manifest = prepare_equality_control(path, 7, 3, 4, 818)
    assert manifest['source']['kind'] == EQUALITY_KIND
    seen = []
    for split, boards, families in (('train', 7, ('direct_equality',)),
                                    ('validation', 3, EQUALITY_FAMILIES),
                                    ('test', 4, EQUALITY_FAMILIES)):
        data = ReasoningDataset(path, split)
        assert len(data) == boards*len(families)
        hashes = {r['metadata']['board_sha256'] for r in data.records}
        seen.append(hashes); assert len(hashes) == boards
        assert {r['metadata']['source'] for r in data.records} == {EQUALITY_KIND}
        for record in data.records:
            assert certified_solution(record) == record['answer']
        assert audit_public_dimensions(data)['rows'] == len(data)
    assert not (seen[0]&seen[1] or seen[0]&seen[2] or seen[1]&seen[2])
    assert prepare_equality_control(path, 7, 3, 4, 818) == manifest


def test_existing_trainer_smoke_resume_and_generation_suppression(tmp_path, monkeypatch, cpu_only):
    path = tmp_path/'data'; prepare(path, 1, 1, 1, 36)
    def unexpected(*args, **kwargs):
        raise AssertionError('Epoch-end generation should be suppressed')
    monkeypatch.setattr('reasoning.tfw_runner.generation_report', unexpected)
    def args(run, stop):
        return parser().parse_args(['--data-dir', str(path), '--run-dir', str(tmp_path/run),
            '--device', 'cpu', '--precision', 'fp32', '--debug', '--cpu-threads', '1',
            '--global-batch', '4', '--micro-batch', '2', '--logit-shift', '0',
            '--target-region', 'answer', '--padding-attention', 'masked', '--zebra-encoding', 'typed_coordinates',
            '--stop-after-steps', str(stop), '--validation-examples', '1', '--eval-batch-size', '1',
            '--generation-every', '0', '--no-epoch-end-generation'])
    train(args('direct', 2)); train(args('resumed', 1)); train(args('resumed', 2))
    a, b = [load_checkpoint(tmp_path/run/'checkpoints/last.pt') for run in ('direct', 'resumed')]
    for key in ('model', 'optimizer', 'rng_by_rank', 'contract', 'examples_seen', 'step'):
        assert_tree_equal(a[key], b[key])
    output = tmp_path/'report.json'
    evaluate(Namespace(data_dir=str(path), checkpoint=str(tmp_path/'direct/checkpoints/last.pt'),
                       split='test', device='cpu', output=str(output)))
    result = json.loads(output.read_text())
    assert result['benchmark_equivalence'] is False
    assert set(result['groups']) == set(FAMILIES)
    assert set(result['reports']) == {'sample', 'argmax'}
    for family, group in result['groups'].items():
        assert group['examples'] == 1 and group['content_tokens'] == 25
        assert group['trained_family'] == (family in TRAIN_FAMILIES)
