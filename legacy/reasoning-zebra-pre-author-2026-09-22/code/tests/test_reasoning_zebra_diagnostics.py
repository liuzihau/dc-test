import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reasoning.tasks import _serialize_zebra, score_prediction
from reasoning.zebra_diagnostics import diagnostic_rows, summarize_zebra


def fixture(tmp_path):
    directory = tmp_path / 'data'
    directory.mkdir()
    records, examples = [], []
    for index in range(3):
        row = list(range(1, 6))
        row = row[index:] + row[:index]
        answer = row * 5
        constraints = [('AT', i, position) for i, position in enumerate(answer)]
        a, b = row.index(1), row.index(2)
        constraints += [('SAME', 0, 5), ('LEFT', a, b), ('NEXT', a, b)]
        record = dict(id=f'puzzle-{index}', task='zebra', prompt=_serialize_zebra(constraints),
                      answer=[str(value) for value in answer])
        records.append(record)
        prediction = list(record['answer'])
        if index == 1:
            prediction[5], prediction[6] = prediction[6], prediction[5]
        if index == 2:
            prediction[0] = '[MASK]'
        prediction += ['[EOS]']
        examples.append(dict(id=record['id'], record_index=index, task='zebra',
            predicted_answer_slots=prediction, scores=score_prediction(record, prediction)))
    testfile = directory / 'test.jsonl'
    testfile.write_text(''.join(json.dumps(record) + '\n' for record in records))
    manifest = dict(schema_version=1, task='zebra', splits=dict(test=dict(
        filename='test.jsonl', records=3, sha256=hashlib.sha256(testfile.read_bytes()).hexdigest())))
    manifest_path = directory / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    payload = dict(arguments=dict(data_dir=str(directory)), step=5000, examples=examples,
                   contract=dict(data_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()))
    return SimpleNamespace(payload=payload, variant='mdm', task='zebra', path=tmp_path / 'evaluation.json'), directory


def test_oracle_wrong_relation_and_malformed_have_explicit_denominators(tmp_path):
    evaluation, _ = fixture(tmp_path)
    rows = {row['metric']: row for row in diagnostic_rows(evaluation)}
    assert rows['wellformed25digits']['numerator'] == 2
    assert rows['wellformed25digits']['denominator'] == 3
    assert rows['has_eos']['numerator'] == rows['has_eos']['denominator'] == 3
    assert rows['position_accuracy']['numerator'] == 48
    assert rows['position_accuracy']['denominator'] == 50
    assert rows['all_five_permutations']['percentage'] == 100
    assert rows['all_five_permutations']['denominator'] == 2
    assert rows['clue_AT']['numerator'] == 48
    assert rows['clue_AT']['denominator'] == 50
    assert rows['clue_SAME']['numerator'] == 1
    assert rows['clue_SAME']['denominator'] == 2
    assert rows['category_C0']['percentage'] == 100
    assert rows['category_C1']['percentage'] == 80
    assert 'malformed answers excluded' in rows['clue_SAME']['conditioning']
    assert 'NOT whole-puzzle' in rows['clue_SAME']['meaning']


@pytest.mark.parametrize('change,match', [
    (lambda e, d: e.payload['contract'].update(data_sha256='b'*64), 'manifest SHA256'),
    (lambda e, d: (d / 'test.jsonl').write_text('{}\n'), 'test split SHA256'),
    (lambda e, d: e.payload['examples'][0].update(id='wrong'), 'ID/task mismatch'),
    (lambda e, d: e.payload['examples'][0].update(record_index=-1), 'out of range'),
    (lambda e, d: e.payload['examples'][0]['scores'].update(valid_solution=False), 'rescored'),
])
def test_diagnostic_provenance_failure(tmp_path, change, match):
    evaluation, directory = fixture(tmp_path)
    change(evaluation, directory)
    with pytest.raises(ValueError, match=match):
        diagnostic_rows(evaluation)


def test_unavailable_dataset_warns_without_aborting_primary_report(tmp_path):
    evaluation, _ = fixture(tmp_path)
    evaluation.payload['arguments']['data_dir'] = str(tmp_path / 'unavailable')
    warnings = summarize_zebra([evaluation], tmp_path / 'report')
    assert len(warnings) == 1 and 'diagnostics skipped' in warnings[0]
    assert (tmp_path / 'report/zebra_diagnostics.csv').read_text().count('\n') == 1
    assert (tmp_path / 'report/zebra_diagnostics.png').stat().st_size > 1000


def test_diagnostics_generate_csv_and_png_without_mutation(tmp_path):
    evaluation, directory = fixture(tmp_path)
    before = copy.deepcopy(evaluation.payload)
    raw = (directory / 'test.jsonl').read_bytes()
    assert not summarize_zebra([evaluation], tmp_path / 'report')
    assert evaluation.payload == before
    assert (directory / 'test.jsonl').read_bytes() == raw
    assert 'clue_SAME,1,2,50.0' in (tmp_path / 'report/zebra_diagnostics.csv').read_text()
    assert (tmp_path / 'report/zebra_diagnostics.png').stat().st_size > 1000
