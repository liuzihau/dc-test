import copy
import json

import pytest

from reasoning.tasks import TaskTokenizer, score_prediction, validate_record
from reasoning.zebra_official_report import sha256, summarize, wilson


@pytest.fixture
def prepared(tmp_path):
    directory = tmp_path / 'data'
    directory.mkdir()
    records = []
    for houses in range(3, 7):
        for attributes in range(3, 7):
            record = dict(task='zebra-official', prompt=['HOUSES', str(houses), 'ATTRS', str(attributes),
                '=', 'LHS', 'c', '0', '0', 'RHS', 'n', '0', '0', 'CLUE_END'],
                answer=list(map(str, range(houses))) * attributes,
                metadata=dict(houses=houses, attributes=attributes))
            records.append(validate_record(record))
    split = directory / 'test.jsonl'
    split.write_text(''.join(json.dumps(record) + '\n' for record in records))
    manifest = dict(schema_version=1, task='zebra-official', max_length=400, answer_slots=37,
                    vocab=TaskTokenizer('zebra-official').tokens,
                    splits=dict(test=dict(filename='test.jsonl', sha256=sha256(split), records=16)))
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    examples = []
    for index, record in enumerate(records):
        raw = record['answer'] + ['[EOS]'] + ['[PAD]'] * (36-len(record['answer'])) if index % 2 == 0 else ['[PAD]'] * 37
        examples.append(dict(id=record['id'], record_index=index, task='zebra-official',
            all_slots_completed=True, remaining_masked_slots=0, token_selection='sample', nfe=37,
            predicted_answer_slots=raw, scores=score_prediction(record, raw)))
    evaluation = dict(checkpoint='/not-present/source-checkpoint.pt', step=5000,
        contract=dict(task='zebra-official', variant='vanilla', data_sha256=sha256(directory / 'manifest.json')),
        arguments=dict(protocol='generate', split='test', policy='top_prob', memory_condition='correct', seed=2026, examples=16),
        metrics=dict(evaluation='closed_loop_generation', policy='top_prob', candidate_k=8, memory_condition='correct',
            token_selection='paper', tokens_per_step=1, max_steps=None, seed=2026, mean_nfe_per_example=37,
            num_examples=16, valid_solution=.5, exact_match=.5), examples=examples)
    path = tmp_path / 'evaluation.json'
    path.write_text(json.dumps(evaluation))
    return directory, path, evaluation


def test_official_report_sizes_wilson_content_and_input_preservation(prepared, tmp_path):
    directory, path, evaluation = prepared
    before = path.read_bytes()
    result = summarize(path, directory, tmp_path / 'report')
    assert len(result['by_size']) == 16
    assert result['overall']['solved'] == 8
    assert result['overall']['examples'] == 16
    assert result['overall']['solve_accuracy'] == .5
    assert result['overall']['content_tokens'] == sum(h*a for h in range(3, 7) for a in range(3, 7))
    # Malformed answers contribute their whole content count to the denominator.
    assert result['by_size'][1]['content_correct'] == 0
    assert result['by_size'][1]['content_tokens'] == 12
    assert result['by_size'][0]['content_accuracy'] == 1
    assert result['overall']['solve_ci95_low'] < .5 < result['overall']['solve_ci95_high']
    assert path.read_bytes() == before
    assert (tmp_path / 'report/official_zebra_accuracy_by_size.png').exists()
    assert 'not an exact reproduction' in (tmp_path / 'report/summary.md').read_text()


@pytest.mark.parametrize('mutation', [
    lambda data: data['contract'].update(data_sha256='0'*64),
    lambda data: data['arguments'].update(protocol='cold'),
    lambda data: data['examples'][0].update(id='wrong'),
    lambda data: data['examples'][0]['scores'].update(valid_solution=False),
    lambda data: data['metrics'].update(valid_solution=.6),
    lambda data: data['metrics'].update(mean_nfe_per_example=26),
])
def test_official_report_rejects_mismatched_data_protocol_or_scores(prepared, tmp_path, mutation):
    directory, path, data = prepared
    mutation(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        summarize(path, directory, tmp_path / 'report')


def test_cannot_overwrite_prepared_data_or_evaluation(prepared, tmp_path):
    directory, path, data = prepared
    with pytest.raises(ValueError, match='immutable'):
        summarize(path, directory, directory / 'report')
    collision = tmp_path / 'summary.json'
    collision.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='overwrite'):
        summarize(collision, directory, tmp_path)


def test_wilson_boundary_cases():
    assert wilson(0, 0) == (None, None)
    low, high = wilson(0, 100)
    assert low == pytest.approx(0)
    assert 0 < high < .05
    low, high = wilson(100, 100)
    assert .95 < low < 1
    assert high == pytest.approx(1)
