import copy
import json
import math

import pytest

from reasoning.zebra_audit_report import read_audit, summarize


def payload(variant='both', step=5000):
    data = dict(schema_version=1, checkpoint='/cloud/checkpoint.pt', checkpoint_sha256='a' * 64,
                data_sha256='b' * 64, step=step,
                contract=dict(task='zebra', variant=variant, data_sha256='b' * 64, seed=1, global_batch=128),
                arguments=dict(trust_checkpoint=True, seed=2026, batch_size=8, examples=2),
                protocol=dict(splits=['train', 'validation'], no_test_tuning=True,
                              cold_memory='none; independent first forward'), splits={})
    for split in ('train', 'validation'):
        categories = {name: dict(tokens=50 if name == 'all' else 10, conditional_nll=math.log(5),
                                top1_accuracy=.2, expected_uniform_tie_accuracy=.2)
                      for name in ('all', 'C0', 'C1', 'C2', 'C3', 'C4')}
        summary = [dict(mask_ratio=1., condition=condition, categories=copy.deepcopy(categories))
                   for condition in ('original', 'clue_content_permuted')]
        examples = [dict(id=f'{split}-{i}', mask_ratio=1., condition=condition)
                    for condition in ('original', 'clue_content_permuted') for i in range(2)]
        generation = {}
        for condition in ('correct', 'none'):
            generation[condition] = dict(metrics=dict(num_examples=2, memory_condition=condition,
                policy='top_prob', candidate_k=8, tokens_per_step=1, seed=2026, valid_solution=0.,
                token_selection='paper', max_steps=None, mean_nfe_per_example=26),
                examples=[dict(id=f'{split}-{i}') for i in range(2)],
                diagnostics=dict(values={'C0': dict(accuracy=.2), 'clue_AT': dict(accuracy=.2)}))
        data['splits'][split] = dict(examples=2, cold=dict(summary=summary, examples=examples,
            permutation_shortcut=dict(summary=[dict(mask_ratio=1., categories=categories)])), generation=generation)
    return data


def write(path, data):
    path.write_text(json.dumps(data))
    return path


def test_pending_report_accepts_no_audits(tmp_path):
    audits = tmp_path / 'audits'
    audits.mkdir()
    output = tmp_path / 'report'
    assert summarize(audits, output) == []
    assert 'Pending' in (output / 'summary.md').read_text()
    assert (output / 'zebra_audit.png').exists()


def test_report_preserves_input_and_computes_5k_10k_deltas(tmp_path):
    audits = tmp_path / 'audits'
    audits.mkdir()
    first = write(audits / '5k.json', payload())
    before = first.read_bytes()
    later = payload(step=10000)
    for section in later['splits'].values():
        section['cold']['summary'][0]['categories']['all']['conditional_nll'] = 1.2
        section['cold']['summary'][0]['categories']['all']['top1_accuracy'] = .3
        section['generation']['correct']['metrics']['valid_solution'] = .05
    write(audits / '10k.json', later)
    output = tmp_path / 'report'
    rows = summarize(audits, output)
    assert len(rows) == 4
    assert first.read_bytes() == before
    final = next(row for row in rows if row['step'] == 10000 and row['split'] == 'validation')
    assert final['cold_all_intact_minus_permuted_pp'] == pytest.approx(10)
    assert final['generation_correct_minus_none_pp'] == 5
    assert 'cold_original_all_nll_delta_10k_minus_5k' in (output / 'audit_5k_to_10k_deltas.csv').read_text()
    assert '| both | Complete | Complete |' in (output / 'summary.md').read_text()
    assert len(json.loads((output / 'provenance.json').read_text())) == 2


@pytest.mark.parametrize('mutate', [
    lambda data: data.update(checkpoint_sha256='not-a-sha'),
    lambda data: data['contract'].update(data_sha256='c' * 64),
    lambda data: data['arguments'].update(trust_checkpoint=False),
    lambda data: data['splits']['train']['cold']['examples'][0].update(id='wrong'),
    lambda data: data['splits']['validation']['generation']['correct']['metrics'].update(token_selection='argmax'),
    lambda data: data['splits']['train']['cold']['summary'][0]['categories']['all'].update(tokens=99),
])
def test_rejects_invalid_provenance_ids_or_scoring(tmp_path, mutate):
    data = payload()
    mutate(data)
    with pytest.raises(ValueError):
        read_audit(write(tmp_path / 'invalid.json', data))


def test_rejects_incompatible_comparison_and_conflicting_duplicates(tmp_path):
    audits = tmp_path / 'audits'
    audits.mkdir()
    write(audits / 'first.json', payload())
    second = payload(variant='mdm')
    second['arguments']['seed'] = 100
    for split in second['splits'].values():
        for generation in split['generation'].values():
            generation['metrics']['seed'] = 100
    second_path = write(audits / 'second.json', second)
    with pytest.raises(ValueError, match='Incompatible'):
        summarize(audits, tmp_path / 'report')
    second = payload()
    second['checkpoint_sha256'] = 'c' * 64
    write(second_path, second)
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        summarize(audits, tmp_path / 'report')


def test_rejects_report_written_inside_input_directory(tmp_path):
    with pytest.raises(ValueError, match='outside'):
        summarize(tmp_path, tmp_path / 'reports')
