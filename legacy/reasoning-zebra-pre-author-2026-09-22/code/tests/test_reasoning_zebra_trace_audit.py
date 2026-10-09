import copy

import pytest
import torch
from torch.utils.data import DataLoader, default_collate

from reasoning.benchmark import convert_zebra
from reasoning.data import ReasoningDataset, write_prepared_dataset
from reasoning.evaluation import evaluate_generation
from reasoning.tfw import PaperMDM
from reasoning.tfw_runner import model_config
from reasoning.zebra_trace_audit import (aggregate, clue_scores, fullmask_probe,
                                        trace_batch, unary_constraints)
from test_reasoning_benchmark import zebra_record


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1)
    records = [convert_zebra(zebra_record(i, h=3+i % 2)) for i in range(4)]
    training = convert_zebra(zebra_record(11, h=5))
    write_prepared_dataset(tmp_path, 'zebra-benchmark', dict(train=[training], validation=records, test=[]), 17, {})
    dataset = ReasoningDataset(tmp_path, 'validation')
    torch.manual_seed(17)
    model = PaperMDM(model_config(dataset, True, 0, 'answer', 'masked', 'typed_coordinates')).eval()
    return model, dataset


@pytest.mark.parametrize('selection', ['sample', 'argmax'])
def test_trace_matches_existing_evaluator_and_batching(setup, selection):
    model, dataset = setup
    batch = default_collate([dataset[i] for i in range(4)])
    with torch.inference_mode():
        traced = trace_batch(model, batch, dataset.tokenizer, dataset.records, selection=selection)
        _, official = evaluate_generation(model, DataLoader(dataset, batch_size=4), tokenizer=dataset.tokenizer,
            records=dataset.records, seed=2026, token_selection=selection)
        singles = [trace_batch(model, default_collate([dataset[i]]), dataset.tokenizer, dataset.records,
                              selection=selection)[0] for i in range(4)]
    for a, b, c in zip(traced, official, singles):
        for key in ('decode_order', 'predicted_answer_slots', 'scores'):
            assert a[key] == b[key] == c[key]


def test_gold_not_used_for_normal_decode_but_oracle_is_explicit(setup):
    model, dataset = setup
    batch = default_collate([dataset[0], dataset[1]])
    poisoned = copy.deepcopy(batch)
    poisoned['input_ids'][poisoned['target_mask']] = dataset.tokenizer.sep_id
    with torch.inference_mode():
        actual = trace_batch(model, batch, dataset.tokenizer, dataset.records)
        other = trace_batch(model, poisoned, dataset.tokenizer, dataset.records)
        oracle = trace_batch(model, batch, dataset.tokenizer, dataset.records, replay=actual)
        for a, b, o in zip(actual, other, oracle):
            assert a['decode_order'] == b['decode_order'] == o['decode_order']
            assert a['predicted_answer_slots'] == b['predicted_answer_slots']
            assert 'scores' not in o
            assert all(e['committed_correct'] for e in o['events'])
            assert o['predicted_answer_slots'] == dataset.records[a['record_index']]['answer'] + ['[EOS]']
        actual[0]['decode_order'][0] = -1
        with pytest.raises(ValueError, match='every public target'):
            trace_batch(model, batch, dataset.tokenizer, dataset.records, replay=actual)


def test_clue_metrics_preserve_all_denominators(setup):
    _, dataset = setup
    record = dataset.records[0]
    good = clue_scores(record, record['answer'] + ['[EOS]'])
    bad = clue_scores(record, ['[MASK]'] * (len(record['answer']) + 1))
    assert good['all_rows_valid'] and not bad['all_rows_valid']
    for kind in good['by_type']:
        assert good['by_type'][kind]['total'] == good['by_type'][kind]['satisfied']
        assert bad['by_type'][kind]['total'] == good['by_type'][kind]['total']
        assert bad['by_type'][kind]['satisfied'] == 0
        assert bad['by_type'][kind]['invalid_reference'] == bad['by_type'][kind]['total']


@pytest.mark.parametrize('kind,refs,allowed', [
    ('=', [('c', 0, 2), ('n', 0, 1)], [1]),
    ('!=', [('n', 0, 1), ('c', 0, 2)], [0, 2]),
    ('left-of', [('c', 0, 2), ('n', 0, 1)], [0]),
    ('immediate-left', [('n', 0, 0), ('c', 0, 2)], [1]),
    ('nbr', [('c', 0, 2), ('n', 0, 1)], [0, 2]),
    ('ends', [('c', 0, 2)], [0, 2]),
    ('inbetween', [('n', 0, 0), ('c', 0, 2), ('n', 0, 2)], [1]),
])
def test_public_single_entity_relation_directions(kind, refs, allowed):
    prompt = [kind, 'LHS'] + list(map(str, refs[0])) + ['RHS']
    for ref in refs[1:]:
        prompt.extend(map(str, ref))
    prompt.append('CLUE_END')
    record = dict(metadata=dict(houses=3, attributes=3), prompt=prompt)
    assert unary_constraints(record) == [dict(kind=kind, attribute=0, value=2, allowed=allowed)]


def test_aggregate_excludes_eos_and_does_not_score_oracle_solves(setup):
    model, dataset = setup
    batch = default_collate([dataset[0], dataset[1]])
    with torch.inference_mode():
        probes = fullmask_probe(model, batch, dataset.tokenizer, dataset.records)
        sample = trace_batch(model, batch, dataset.tokenizer, dataset.records)
        greedy = trace_batch(model, batch, dataset.tokenizer, dataset.records, selection='argmax')
        oracle = trace_batch(model, batch, dataset.tokenizer, dataset.records, replay=sample)
    summary = aggregate(probes, dict(sample=sample, argmax=greedy, oracle_replay=oracle))
    expected = sum(len(r['answer']) for r in dataset.records[:2])
    assert summary['fullmask_groups']['all_content']['tokens'] == expected
    for condition in summary['conditions'].values():
        assert condition['content_tokens'] == expected
    assert 'scores' not in summary['conditions']['oracle_replay']
    assert sum(e['content'] for e in sample[0]['events']) == len(dataset.records[0]['answer'])
