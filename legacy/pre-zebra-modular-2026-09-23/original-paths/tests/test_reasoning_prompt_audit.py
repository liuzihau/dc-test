import argparse
import math
import random

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from reasoning.data import ReasoningDataset, prepare_dataset
from reasoning.prompt_audit import (audit, category_masks, corrupt_clue_content,
                                    evaluate_cold_prompt, generation_diagnostics,
                                    isolated_rng, target_mask_for_ratio)


@pytest.fixture
def zebra(tmp_path):
    directory = tmp_path / 'data'
    prepare_dataset(directory, 'zebra', train_size=4, valid_size=2, test_size=1, seed=17)
    return ReasoningDataset(directory, 'train')


class UniformModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.calls = []

    def forward(self, input_ids, attention_mask, **kwargs):
        assert kwargs['previous_step_kv'] is None
        assert kwargs['previous_final_hidden'] is None
        assert kwargs['return_memory'] is False
        self.calls.append(input_ids.clone())
        return {'logits': torch.zeros(*input_ids.shape, self.vocab)}


def test_prompt_permutation_preserves_layout_targets_and_histograms(zebra):
    batch = next(iter(DataLoader(zebra, batch_size=4)))
    ids, attention, target = [batch[key] for key in ('input_ids', 'attention_mask', 'target_mask')]
    indices = batch['record_index'].tolist()
    output = corrupt_clue_content(ids, attention, target, indices, zebra.tokenizer, 23, 'train')
    again = corrupt_clue_content(ids, attention, target, indices, zebra.tokenizer, 23, 'train')
    assert torch.equal(output, again)
    assert torch.equal(ids[target], output[target])
    fixed = ~attention | target
    for name in ('AT', 'SAME', 'LEFT', 'NEXT', ';', '[BOS]', '[SEP]', '[PAD]', '[EOS]'):
        fixed |= ids == zebra.tokenizer.token_to_id[name]
    assert torch.equal(ids[fixed], output[fixed])
    assert (ids != output).any()
    for before, after in zip(ids, output):
        assert torch.equal(before.sort().values, after.sort().values)
    # Same record corruption in a singleton batch.
    singleton = corrupt_clue_content(ids[2:3], attention[2:3], target[2:3], [2], zebra.tokenizer, 23, 'train')
    assert torch.equal(singleton, output[2:3])


def test_exact_nested_masks_and_category_eos_exclusion(zebra):
    batch = next(iter(DataLoader(zebra, batch_size=4)))
    target, gold = batch['target_mask'], batch['input_ids']
    indices = batch['record_index'].tolist()
    previous = None
    for ratio, expected in ((1, 26), (.7, 19), (.3, 8)):
        masked = target_mask_for_ratio(target, indices, ratio, 23, 'validation')
        assert masked.sum(-1).tolist() == [expected] * 4
        assert not (masked & ~target).any()
        if previous is not None:
            assert not (masked & ~previous).any()
        previous = masked
    groups = category_masks(gold, target, zebra.tokenizer)
    assert groups['all'].sum(-1).tolist() == [25] * 4
    for category in range(5):
        assert groups[f'C{category}'].sum(-1).tolist() == [5] * 4
    assert not (groups['all'] & gold.eq(zebra.tokenizer.eos_id)).any()


def test_cold_report_resets_memory_scores_content_and_restores_training(zebra):
    model = UniformModel(zebra.tokenizer.vocab_size)
    model.train()
    result = evaluate_cold_prompt(model, DataLoader(zebra, batch_size=2), 'cpu',
                                 tokenizer=zebra.tokenizer, records=zebra.records, seed=3, split='train')
    assert model.training
    assert len(model.calls) == 12
    assert len(result['examples']) == 4 * 3 * 2
    for row in result['summary']:
        all_counts = row['categories']['all']
        assert all_counts['conditional_nll'] == pytest.approx(math.log(zebra.tokenizer.vocab_size - 1), abs=1e-6)
        assert all_counts['top1_accuracy'] == 0
        assert sum(row['categories'][f'C{i}']['tokens'] for i in range(5)) == all_counts['tokens']
        if row['mask_ratio'] == 1:
            assert all_counts['tokens'] == 100
    # Conditions keep target positions identical and mask every slot at ratio 1.
    for first, second in zip(model.calls[::2], model.calls[1::2]):
        assert torch.equal(first == zebra.tokenizer.mask_id, second == zebra.tokenizer.mask_id)
    calibration = result['permutation_shortcut']
    assert len(calibration['examples']) == 4 * 3
    full = calibration['summary'][0]
    assert full['mask_ratio'] == 1
    assert full['categories']['all']['tokens'] == 100
    assert full['categories']['all']['conditional_nll'] == pytest.approx(math.log(5), abs=1e-6)
    assert full['categories']['all']['expected_uniform_tie_accuracy'] == pytest.approx(.2)
    for calibrated in calibration['summary']:
        matching = next(row for row in result['summary']
                        if row['mask_ratio'] == calibrated['mask_ratio'] and row['condition'] == 'original')
        assert calibrated['categories']['all']['tokens'] == matching['categories']['all']['tokens']
        assert calibrated['categories']['all']['conditional_nll'] <= math.log(5) + 1e-6


def test_rng_context_restores_python_and_torch():
    torch_state, python_state = torch.get_rng_state(), random.getstate()
    with isolated_rng(torch.device('cpu')):
        torch.rand(10)
        random.random()
    assert torch.equal(torch_state, torch.get_rng_state())
    assert python_state == random.getstate()


def test_generation_diagnostics_use_actual_scorer_and_fixed_denominators(zebra):
    from reasoning.tasks import score_prediction
    details = []
    for index, record in enumerate(zebra.records[:2]):
        raw = record['answer'] + ['[EOS]'] if index == 0 else ['[EOS]'] * 26
        details.append(dict(id=record['id'], record_index=index,
                            predicted_answer_slots=raw, scores=score_prediction(record, raw)))
    result = generation_diagnostics(details, zebra.records)['values']
    assert result['wellformed']['accuracy'] == .5
    assert result['position_accuracy'] == dict(numerator=25, denominator=50, accuracy=.5)
    assert result['C4'] == dict(numerator=5, denominator=10, accuracy=.5)
    details[0]['scores']['valid_solution'] = False
    with pytest.raises(ValueError, match='scorer'):
        generation_diagnostics(details, zebra.records)


def test_requires_explicit_checkpoint_trust_before_loading(tmp_path):
    args = argparse.Namespace(trust_checkpoint=False)
    with pytest.raises(ValueError, match='trust-checkpoint'):
        audit(args)


def test_refuses_existing_output_before_checkpoint_loading(tmp_path):
    output = tmp_path / 'audit.json'
    output.write_text('{}')
    args = argparse.Namespace(trust_checkpoint=True, examples=2, batch_size=2,
                              cpu_threads=1, output=str(output))
    with pytest.raises(FileExistsError):
        audit(args)


def test_checkpoint_audit_end_to_end_never_loads_test_and_keeps_checkpoint(tmp_path, zebra, monkeypatch):
    import json
    import reasoning.data as data_module
    from reasoning.model import ReasoningModel
    from reasoning.runner import digest, save_checkpoint
    data_dir = tmp_path / 'data'
    model = ReasoningModel(dict(vocab_size=zebra.tokenizer.vocab_size,
        special_ids=list(zebra.tokenizer.special_ids), hidden_size=16, n_heads=4, n_layers=1,
        max_length=384, memory_mode='both', merged_policy='current_preserving',
        gate_enabled=False, cache_only_probability=0))
    optimizer = torch.optim.AdamW(model.parameters())
    contract = dict(task='zebra', data_sha256=digest(data_dir / 'manifest.json'), precision='fp32', global_batch=2)
    save_checkpoint(tmp_path / 'run', model, optimizer, 1, contract, 0, 1)
    path = tmp_path / 'run/checkpoints/last.pt'
    before = digest(path)
    loaded = []
    original = data_module.ReasoningDataset
    def checked_dataset(directory, split):
        assert split in ('train', 'validation')
        loaded.append(split)
        return original(directory, split)
    monkeypatch.setattr(data_module, 'ReasoningDataset', checked_dataset)
    output = tmp_path / 'audit.json'
    args = argparse.Namespace(trust_checkpoint=True, checkpoint=str(path), data_dir=str(data_dir),
        output=str(output), device='cpu', examples=1, batch_size=1, seed=2026,
        cpu_threads=1, skip_generation=False)
    result = audit(args)
    assert loaded == ['train', 'validation']
    assert digest(path) == before
    assert json.loads(output.read_text())['checkpoint_sha256'] == before
    assert result['step'] == 1
    for split in ('train', 'validation'):
        for condition in ('correct', 'none'):
            generation = result['splits'][split]['generation'][condition]
            assert generation['metrics']['num_examples'] == 1
            assert generation['metrics']['mean_nfe_per_example'] == 26
            assert generation['diagnostics']['values']['position_accuracy']['denominator'] == 25
