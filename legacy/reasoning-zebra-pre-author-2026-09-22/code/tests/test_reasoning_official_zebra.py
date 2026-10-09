"""Source-aligned import safety, formats, scoring and split invariants."""
import copy
import io
import itertools
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from reasoning.data import ReasoningDataset
from reasoning.tasks import (TASK_LENGTHS, TaskTokenizer, generate_record, normalize_task,
                             score_prediction, task_identity, validate_record)
from reasoning.zebra_official import (RestrictedZebraUnpickler, convert_source_record,
                                     import_dataset, load_source, parse_prompt,
                                     relation_holds)


def raw_record(index=0, houses=3, attributes=3):
    permutation = list(itertools.permutations(range(houses)))[index]
    table = [list(range(houses))] + [list(permutation)] + [list(range(houses)) for _ in range(attributes - 1)]
    prompt = []
    for category, row in enumerate(table[1:]):
        # The final position is implied by the permutation; retain realistic
        # source lengths rather than encoding a redundant clue for every cell.
        for position, value in enumerate(row[:-1]):
            prompt += ['=', 'LHS', 'c', str(category), str(value), 'RHS', 'n', '0', str(position), 'CLUE_END']
    trace = np.arange(houses * attributes * 2, dtype=np.int16).reshape(houses * attributes, 2)
    return [prompt + ['ANSWER', 'DO_NOT_FEED_THIS_TRACE'], table, trace]


def safe_raw(index=0, houses=3, attributes=3):
    return RestrictedZebraUnpickler(io.BytesIO(pickle.dumps(raw_record(index, houses, attributes), protocol=4))).load()


def write_pickle(path, data):
    path.write_bytes(pickle.dumps(data, protocol=4))
    return path


def test_safe_legacy_numpy_trace_and_answer_stripping(tmp_path):
    path = write_pickle(tmp_path / 'test.pkl', [raw_record()])
    raw = load_source(path)[0]
    record = validate_record(convert_source_record(raw, 'test', 5))
    assert normalize_task('zebra_official') == 'zebra-official'
    assert record['prompt'][:4] == ['HOUSES', '3', 'ATTRS', '3']
    assert 'ANSWER' not in record['prompt']
    assert 'DO_NOT_FEED_THIS_TRACE' not in record['prompt']
    assert record['answer'] == list('012012012')
    assert record['metadata']['source_index'] == 5
    assert score_prediction(record, record['answer'] + ['[EOS]'] + ['[PAD]'] * 27)['valid_solution']
    assert not score_prediction(record, record['answer'] + ['[EOS]', '0'])['valid_solution']


@pytest.mark.parametrize('houses,attributes', itertools.product(range(3, 7), repeat=2))
def test_every_released_dimension(houses, attributes):
    record = validate_record(convert_source_record(safe_raw(0, houses, attributes), 'train', 0))
    assert len(record['answer']) == houses * attributes
    assert score_prediction(record, record['answer'])['valid_solution']
    wrong = record['answer'].copy()
    wrong[0], wrong[1] = wrong[1], wrong[0]
    assert not score_prediction(record, wrong)['valid_solution']


@pytest.mark.parametrize('kind,positions,positive', [
    ('=', [1, 1], True), ('=', [0, 1], False), ('!=', [0, 1], True),
    ('immediate-left', [0, 1], True), ('immediate-left', [0, 2], False),
    ('nbr', [2, 1], True), ('nbr', [0, 2], False),
    ('ends', [0], True), ('ends', [2], True), ('ends', [1], False),
    ('left-of', [0, 2], True), ('left-of', [2, 0], False),
    ('inbetween', [0, 1, 2], True), ('inbetween', [1, 0, 2], False),
    ('inbetween', [2, 1, 0], False),
])
def test_seven_relation_semantics(kind, positions, positive):
    assert relation_holds(kind, positions, 3) is positive


def test_clue_arity_and_reference_range_fail_closed():
    prefix = ['HOUSES', '3', 'ATTRS', '3']
    valid = ['ends', 'LHS', 'c', '0', '1', 'RHS', 'CLUE_END']
    assert len(parse_prompt(prefix + valid)[2]) == 1
    with pytest.raises(ValueError, match='arity'):
        parse_prompt(prefix + valid[:-1] + ['n', '0', '0', 'CLUE_END'])
    with pytest.raises(ValueError, match='index'):
        parse_prompt(prefix + ['=', 'LHS', 'c', '3', '0', 'RHS', 'n', '0', '0', 'CLUE_END'])
    with pytest.raises(ValueError):
        parse_prompt(prefix + valid[:-1])


def test_duplicate_clue_order_and_symmetric_identity():
    record = validate_record(convert_source_record(safe_raw(), 'train', 0))
    same = ['=', 'LHS', 'c', '0', '0', 'RHS', 'c', '1', '0', 'CLUE_END']
    reverse = ['=', 'LHS', 'c', '1', '0', 'RHS', 'c', '0', '0', 'CLUE_END']
    a, b = copy.deepcopy(record), copy.deepcopy(record)
    a['prompt'] += same
    b['prompt'] = b['prompt'][:4] + reverse + b['prompt'][4:] + reverse
    assert task_identity(a) == task_identity(b)


def test_reject_executable_pickles_and_trailing_payload(tmp_path):
    class Malicious:
        def __reduce__(self):
            return eval, ('1 + 2',)
    path = write_pickle(tmp_path / 'bad.pkl', [Malicious()])
    with pytest.raises(ValueError, match='Forbidden pickle global'):
        load_source(path)
    path = write_pickle(tmp_path / 'trailing.pkl', [raw_record()])
    path.write_bytes(path.read_bytes() + b'x')
    with pytest.raises(ValueError, match='trailing'):
        load_source(path)
    bad = raw_record()
    bad[2] = bad[2].astype(np.float64)
    path = write_pickle(tmp_path / 'dtype.pkl', [bad])
    with pytest.raises(ValueError, match='int16'):
        load_source(path)


def test_import_preserves_holdout_and_is_deterministic(tmp_path):
    # Include a held-out duplicate in train: it must never enter either split.
    train = write_pickle(tmp_path / 'train.pkl', [raw_record(i) for i in range(5)])
    test = write_pickle(tmp_path / 'test.pkl', [raw_record(4), raw_record(5)])
    manifest = import_dataset(train, test, tmp_path / 'one', train_size=2, valid_size=1, test_size=1)
    other = import_dataset(train, test, tmp_path / 'two', train_size=2, valid_size=1, test_size=1)
    assert manifest == other
    assert manifest['source']['benchmark_equivalence'] is False
    assert manifest['source']['source_files']['train']['duplicates_excluded'] == 1
    assert manifest['source']['source_files']['train']['validated_records'] == 5
    all_records = []
    for split in ('train', 'validation', 'test'):
        data = ReasoningDataset(tmp_path / 'one', split)
        all_records += data.records
        assert data[0]['input_ids'].shape == (400,)
        assert data[0]['target_mask'].sum() == 37
        assert all(r['metadata']['source_split'] == ('test' if split == 'test' else 'train') for r in data.records)
    assert len({r['id'] for r in all_records}) == len(all_records)
    with pytest.raises(FileExistsError):
        import_dataset(train, test, tmp_path / 'one', train_size=2, valid_size=1, test_size=1)


def test_invalid_source_and_no_synthetic_fallback(tmp_path):
    raw = safe_raw()
    raw[1][1][0] = 1
    with pytest.raises(ValueError, match='permutations'):
        convert_source_record(raw, 'train', 0)
    with pytest.raises(ValueError, match='no synthetic fallback'):
        generate_record('zebra-official', None)
    assert TaskTokenizer('zebra').tokens == (
        ['[PAD]', '[MASK]', '[BOS]', '[SEP]', '[EOS]'] + list('12345')
        + ['C%d' % i for i in range(5)] + ['V%d' % i for i in range(5)]
        + ['AT', 'SAME', 'LEFT', 'NEXT', ';'])


def test_real_release_smoke_when_available():
    path = Path('imports/official-reasoning/raw/zebra-test-data.retry')
    if not path.exists():
        pytest.skip('Optional downloaded public release not present')
    records = load_source(path)
    for index, raw in enumerate(records[:100]):
        record = validate_record(convert_source_record(raw, 'test', index))
        assert score_prediction(record, record['answer'])['valid_solution']


def test_failed_import_never_publishes_partial_directory(tmp_path, monkeypatch):
    import reasoning.data as data
    train = write_pickle(tmp_path / 'train.pkl', [raw_record(i) for i in range(5)])
    test = write_pickle(tmp_path / 'test.pkl', [raw_record(5)])
    original = data.write_prepared_dataset

    def fail_after_splits(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('simulated interruption before atomic publication')

    monkeypatch.setattr(data, 'write_prepared_dataset', fail_after_splits)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        import_dataset(train, test, tmp_path / 'prepared', train_size=2, valid_size=1, test_size=1)
    assert not (tmp_path / 'prepared').exists()
    staged = list(tmp_path.glob('.prepared.import-*'))
    assert len(staged) == 1 and (staged[0] / 'manifest.json').is_file()


@pytest.fixture
def official_data(tmp_path):
    train = write_pickle(tmp_path / 'train.pkl', [raw_record(i, 4, 4) for i in range(8)])
    test = write_pickle(tmp_path / 'test.pkl', [raw_record(i, 4, 4) for i in (8, 9)])
    out = tmp_path / 'official'
    import_dataset(train, test, out, train_size=4, valid_size=2, test_size=2)
    return out


@pytest.fixture
def cpu_pipeline(monkeypatch):
    # Register torch.compile/Dynamo functions before replacing CUDA functions;
    # assigning the same guard to multiple names during Dynamo import would
    # create an artificial duplicate-rule error unrelated to the model.
    import reasoning.model  # noqa: F401
    for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def no_cuda(*args, **kwargs):
        raise AssertionError('Official CPU integration test attempted CUDA')

    for key in ('_lazy_init', 'set_device', 'get_rng_state', 'reset_peak_memory_stats',
                'max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, key, no_cuda)
    yield
    torch.set_num_threads(old_threads)


@pytest.mark.parametrize('variant', ['vanilla', 'mdm_aux', 'both', 'both_aux'])
def test_official_full_debug_fit_validation_and_generation(
        official_data, tmp_path, cpu_pipeline, variant):
    from reasoning import runner

    run = tmp_path / ('run-' + variant)
    args = runner.parser().parse_args([
        'train', '--task', 'zebra-official', '--variant', variant,
        '--data-dir', str(official_data), '--run-dir', str(run),
        '--size', 'debug', '--device', 'cpu', '--precision', 'fp32',
        '--micro-batch', '2', '--global-batch', '2', '--max-steps', '1',
        '--warmup-steps', '2', '--val-every', '1', '--validation-examples', '2',
        '--eval-batch-size', '2', '--save-every', '1', '--save-seconds', '0',
        '--log-every', '1', '--seed', '17', '--cpu-threads', '1',
        '--merged-policy', 'legacy' if variant == 'vanilla' else 'current_preserving',
        '--gradient-mode', 'adjacent' if variant in ('both', 'both_aux') else 'detached',
    ])
    runner.train(args)
    checkpoint = runner.load_checkpoint(run / 'checkpoints/last.pt')
    assert checkpoint['step'] == 1 and checkpoint['examples_seen'] == 2
    assert checkpoint['contract']['task'] == 'zebra-official'
    assert checkpoint['model_config']['max_length'] == TASK_LENGTHS['zebra-official'] == 400
    validation = json.loads((run / 'validation/step-000000001.json').read_text())
    assert validation['protocol'] == 'cold_independent'
    assert validation['num_examples'] == 2 and validation['reset_each_ratio']
    assert set(validation['ratios']) == {'0.1', '0.3', '0.5', '0.7'}
    output = tmp_path / ('generated-' + variant + '.json')
    runner.evaluate(runner.parser().parse_args([
        'evaluate', '--checkpoint', str(run / 'checkpoints/last.pt'),
        '--data-dir', str(official_data), '--output', str(output),
        '--split', 'test', '--protocol', 'generate', '--examples', '2',
        '--batch-size', '2', '--device', 'cpu', '--cpu-threads', '1', '--seed', '2026',
    ]))
    result = json.loads(output.read_text())
    assert result['metrics']['nfe'] == 37
    assert len(result['examples']) == 2
    assert all(r['all_slots_completed'] for r in result['examples'])
    assert all(math.isfinite(float(value)) for value in result['metrics'].values()
               if isinstance(value, (int, float)))


def test_official_oracle_closed_loop_and_corruption(official_data, cpu_pipeline):
    """An explicit test oracle verifies evaluator plumbing, NOT model accuracy.

    The oracle stores reference outputs deliberately. The evaluator must still
    hide ALL answer/EOS/PAD tokens on its first generation call and decode all
    37 slots without reading reference answer lengths to terminate early.
    """
    from reasoning.evaluation import evaluate_corruption, evaluate_generation

    dataset = ReasoningDataset(official_data, 'test')
    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))

    class Oracle(torch.nn.Module):
        config = dict(memory_mode='none')

        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, input_ids, attention_mask, **kwargs):
            self.calls.append(input_ids.clone())
            assert torch.equal(input_ids[~batch['target_mask']], batch['input_ids'][~batch['target_mask']])
            assert kwargs['previous_step_kv'] is None and kwargs['previous_final_hidden'] is None
            logits = torch.full((*input_ids.shape, dataset.tokenizer.vocab_size), -100.)
            logits.scatter_(-1, batch['input_ids'].unsqueeze(-1), 100.)
            return dict(logits=logits)

    oracle = Oracle()
    metrics, details = evaluate_generation(oracle, loader, device='cpu', seed=2026)
    assert metrics['valid_solution'] == metrics['exact_match'] == 1.0
    assert metrics['nfe'] == len(oracle.calls) == 37
    assert (oracle.calls[0][batch['target_mask']] == dataset.tokenizer.mask_id).all()
    assert all(item['all_slots_completed'] and item['scores']['constraints_satisfied'] for item in details)
    cold, _ = evaluate_corruption(Oracle(), loader, device='cpu', seed=2026)
    assert cold['evaluation'] == 'fixed_corruption'
    for ratio in cold['ratios'].values():
        assert ratio['conditional_nll'] == pytest.approx(0, abs=1e-6)
        assert ratio['masked_token_accuracy'] == 1.0
    # Fixed-slot cold NLL includes supervised PAD. It is not content-only
    # reasoning accuracy or directly comparable with the fixed-size pilot.
    assert ((batch['input_ids'] == dataset.tokenizer.pad_id) & batch['target_mask']).any()
