"""Read-only Zebra checkpoint diagnostics on train and validation, never test.

Prompt destruction is deliberately NOT a matched-distribution intervention:
it measures sensitivity to supplied clue content, not reasoning causality.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .evaluation import _generator, _evaluating, evaluate_generation


RATIOS = (1.0, 0.7, 0.3)
CONDITIONS = ('original', 'clue_content_permuted')


def corrupt_clue_content(ids, attention, target, indices, tokenizer, seed, split):
    """Permute category/value/house symbols separately; preserve syntax slots.

    Target tokens, BOS/SEP/EOS/PAD, operators and semicolons are untouched.
    Independent permutations can destroy clue consistency/uniqueness. Gold
    answers are kept only as scoring targets for the original puzzles.
    """
    result = ids.clone()
    classes = ([f'C{i}' for i in range(5)], [f'V{i}' for i in range(5)], list('12345'))
    prompt = attention.bool() & ~target.bool()
    for row, index in enumerate(indices):
        for class_index, names in enumerate(classes):
            symbols = ids.new_tensor([tokenizer.token_to_id[name] for name in names])
            eligible = prompt[row] & (ids[row, :, None] == symbols[None, :]).any(-1)
            positions = eligible.nonzero(as_tuple=False).flatten()
            permutation = torch.randperm(len(positions), generator=_generator(
                seed, int(index), f'prompt-audit:{split}:symbols:{class_index}'))
            result[row, positions] = ids[row, positions[permutation.to(positions.device)]]
    return result


def target_mask_for_ratio(target, indices, ratio, seed, split):
    """Exact ceil(ratio * 26) answer slots; nested, model/batch independent."""
    if not 0 < ratio <= 1:
        raise ValueError('Mask ratio must be in (0,1]')
    masked = torch.zeros_like(target, dtype=torch.bool)
    for row, index in enumerate(indices):
        positions = target[row].nonzero(as_tuple=False).flatten()
        if not len(positions):
            raise ValueError('Each example needs fixed answer slots')
        order = torch.randperm(len(positions), generator=_generator(
            seed, int(index), f'prompt-audit:{split}:mask'))
        masked[row, positions[order[:math.ceil(ratio * len(positions))].to(positions.device)]] = True
    return masked


def category_masks(gold, target, tokenizer):
    """Category-major Zebra answer content; never EOS/PAD or prompt symbols."""
    offsets = target.long().cumsum(-1) - 1
    house_ids = gold.new_tensor([tokenizer.token_to_id[name] for name in '12345'])
    content = target & (gold[:, :, None] == house_ids[None, None, :]).any(-1)
    if not bool((content.sum(-1) == 25).all()):
        raise ValueError('Zebra requires exactly 25 house-position answer symbols')
    return {'all': content, **{f'C{i}': content & (offsets // 5 == i) for i in range(5)}}


def _empty_counts():
    return {'nll_sum': 0.0, 'correct': 0, 'tokens': 0}


def _finish_counts(counts):
    n = counts['tokens']
    output = {**counts, 'conditional_nll': counts['nll_sum'] / n if n else None,
              'top1_accuracy': counts['correct'] / n if n else None}
    if 'expected_uniform_tie_correct' in counts:
        output['expected_uniform_tie_accuracy'] = counts['expected_uniform_tie_correct'] / n if n else None
    return output


def generation_diagnostics(details, records):
    """Fixed denominators: malformed answers count as wrong, never disappear."""
    from .tasks import _parse_zebra_prompt, _strip_answer, _zebra_relation, score_prediction
    counts = {key: [0, 0] for key in ('wellformed', 'position_accuracy', 'C0', 'C1', 'C2', 'C3', 'C4',
                                     'clue_AT', 'clue_SAME', 'clue_LEFT', 'clue_NEXT')}
    for example in details:
        record = records[example['record_index']]
        if record['id'] != example['id']:
            raise ValueError('Generated record ID mismatch')
        raw = example['predicted_answer_slots']
        if score_prediction(record, raw) != example['scores']:
            raise ValueError('Saved generation scores differ from actual task scorer')
        answer = _strip_answer(raw)
        formed = answer is not None and len(answer) == 25 and all(token in '12345' and len(token) == 1 for token in answer)
        counts['wellformed'][0] += int(formed)
        counts['wellformed'][1] += 1
        correct = [a == b for a, b in zip(answer, record['answer'])] if formed else [False] * 25
        counts['position_accuracy'][0] += sum(correct)
        counts['position_accuracy'][1] += 25
        for category in range(5):
            counts[f'C{category}'][0] += sum(correct[5*category:5*category+5])
            counts[f'C{category}'][1] += 5
        positions = list(map(int, answer)) if formed else None
        for kind, a, b in _parse_zebra_prompt(record['prompt']):
            matched = formed and (positions[a] == b if kind == 'AT' else _zebra_relation(kind, positions[a], positions[b]))
            counts['clue_' + kind][0] += int(matched)
            counts['clue_' + kind][1] += 1
    return dict(conditioning='all examples; malformed answers count as wrong for every position/clue',
                values={key: dict(numerator=numerator, denominator=denominator,
                                  accuracy=numerator / denominator if denominator else None)
                        for key, (numerator, denominator) in counts.items()})


def evaluate_cold_prompt(model, loader, device, *, tokenizer, records, seed, split):
    """Original and destroyed prompts share exactly the same target corruption."""
    from .zebra_shortcut import permutation_shortcut_probabilities
    totals = {(ratio, condition): {key: _empty_counts() for key in ('all', 'C0', 'C1', 'C2', 'C3', 'C4')}
              for ratio in RATIOS for condition in CONDITIONS}
    shortcut_totals = {ratio: {key: {**_empty_counts(), 'expected_uniform_tie_correct': 0.0}
                              for key in ('all', 'C0', 'C1', 'C2', 'C3', 'C4')} for ratio in RATIOS}
    details, shortcut_details = [], []
    with _evaluating(model):
        for batch in loader:
            gold = batch['input_ids'].to(device)
            attention = batch['attention_mask'].to(device).bool()
            target = batch['target_mask'].to(device).bool()
            indices = batch['record_index'].tolist()
            groups = category_masks(gold, target, tokenizer)
            permuted = corrupt_clue_content(gold, attention, target, indices, tokenizer, seed, split)
            for ratio in RATIOS:
                masked = target_mask_for_ratio(target, indices, ratio, seed, split)
                shortcut = permutation_shortcut_probabilities(gold.masked_fill(masked, tokenizer.mask_id), target, tokenizer)
                if not torch.equal(shortcut['eligible'], groups['all'] & masked):
                    raise ValueError('Permutation calibration and model scoring target sets differ')
                distributions = shortcut['probabilities']
                assigned = distributions.gather(-1, gold[..., None]).squeeze(-1)
                selected = shortcut['eligible']
                shortcut_losses = torch.zeros_like(assigned)
                shortcut_losses[selected] = -assigned[selected].log()
                if not bool(torch.isfinite(shortcut_losses).all()):
                    raise ValueError('Target incompatible with clue-blind permutation calibration')
                shortcut_correct = distributions.argmax(-1).eq(gold)
                for row, index in enumerate(indices):
                    row_metrics = {}
                    for name, group in groups.items():
                        eligible = group[row] & selected[row]
                        counts = dict(tokens=int(eligible.sum()), correct=int((shortcut_correct[row] & eligible).sum()),
                                      nll_sum=float(shortcut_losses[row, eligible].sum()),
                                      expected_uniform_tie_correct=float(assigned[row, eligible].sum()))
                        for key, value in counts.items():
                            shortcut_totals[ratio][name][key] += value
                        row_metrics[name] = _finish_counts(counts)
                    shortcut_details.append(dict(id=records[index]['id'], record_index=index,
                                                 mask_ratio=ratio, categories=row_metrics))
                for condition in CONDITIONS:
                    current = (gold if condition == 'original' else permuted).masked_fill(masked, tokenizer.mask_id)
                    output = model(current, attention_mask=attention, previous_step_kv=None,
                                   previous_final_hidden=None, return_memory=False,
                                   detach_cache_backbone=False, source_mask=None)
                    logits = output['logits'].float().clone()
                    logits[..., tokenizer.mask_id] = -torch.inf
                    selected = groups['all'] & masked
                    loss_grid = torch.zeros_like(gold, dtype=torch.float)
                    loss_grid[selected] = F.cross_entropy(logits[selected], gold[selected], reduction='none')
                    if not bool(torch.isfinite(loss_grid).all()):
                        raise ValueError('Nonfinite cold content loss')
                    correct = logits.argmax(-1).eq(gold)
                    for row, index in enumerate(indices):
                        row_metrics = {}
                        for name, group in groups.items():
                            eligible = group[row] & masked[row]
                            counts = dict(tokens=int(eligible.sum()), correct=int((correct[row] & eligible).sum()),
                                          nll_sum=float(loss_grid[row, eligible].sum()))
                            for key, value in counts.items():
                                totals[(ratio, condition)][name][key] += value
                            row_metrics[name] = _finish_counts(counts)
                        details.append(dict(id=records[index]['id'], record_index=index, mask_ratio=ratio,
                                            condition=condition, masked_answer_slots=int(masked[row].sum()),
                                            changed_prompt_tokens=int((gold[row] != permuted[row]).sum()),
                                            categories=row_metrics))
    summary = [dict(mask_ratio=ratio, condition=condition,
                    categories={key: _finish_counts(value) for key, value in counts.items()})
               for (ratio, condition), counts in totals.items()]
    return dict(summary=summary, examples=details, permutation_shortcut=dict(
        description='Clue-blind uniform unused house digits per category; exact same teacher-forced target mask',
        limitation='Uses revealed answer digits and public answer schema, not prompts; NOT a learned solver or whole-puzzle accuracy',
        top1_ties='lowest vocabulary ID; expected_uniform_tie_accuracy averages uniform tie-breaking',
        summary=[dict(mask_ratio=ratio, categories={key: _finish_counts(value) for key, value in counts.items()})
                 for ratio, counts in shortcut_totals.items()], examples=shortcut_details))


@contextmanager
def isolated_rng(device):
    state = random.getstate()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(state)


def audit(args):
    from .data import ReasoningDataset
    from .model import ReasoningModel
    from .runner import atomic_json, digest, load_checkpoint
    if not args.trust_checkpoint:
        raise ValueError('Supply --trust-checkpoint only for your own trusted checkpoint (pickle is loaded)')
    if min(args.examples, args.batch_size, args.cpu_threads) < 1:
        raise ValueError('examples, batch-size and cpu-threads must be positive')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite audit: {output}')
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    checkpoint = load_checkpoint(checkpoint_path)
    data_dir = Path(args.data_dir).resolve(strict=True)
    data_hash = digest(data_dir / 'manifest.json')
    if checkpoint['contract']['task'] != 'zebra':
        raise ValueError('This diagnostic is Zebra-only')
    if data_hash != checkpoint['contract']['data_sha256']:
        raise ValueError('Dataset manifest does not match checkpoint training data')
    device = torch.device(args.device)
    old_threads = torch.get_num_threads()
    result = dict(schema_version=1, checkpoint=str(checkpoint_path),
                  checkpoint_sha256=digest(checkpoint_path), step=checkpoint['step'],
                  contract=checkpoint['contract'], model_config=checkpoint['model_config'],
                  data_dir=str(data_dir), data_sha256=data_hash, arguments=vars(args),
                  protocol=dict(splits=['train', 'validation'], subset='first N in immutable split order',
                                cold_memory='none; independent first forward', ratios=list(RATIOS),
                                masks='ceil(ratio * fixed answer slots), includes EOS; scores exclude EOS',
                                categories='category-major five values each, C0..C4',
                                intervention='separate within-prompt permutations of C symbols, V symbols, house digits; layout unchanged',
                                intervention_limit='Destructive/OOD diagnostic, not distribution-matched causal evidence; targets remain original puzzle answers',
                                generation='original prompts only; top_prob candidate_k=8; sampled tokens; one reveal per iteration',
                                no_test_tuning=True), splits={})
    try:
        torch.set_num_threads(args.cpu_threads)
        with isolated_rng(device):
            model = ReasoningModel(checkpoint['model_config']).to(device)
            model.load_state_dict(checkpoint['model'], strict=True)
            model.eval()
            del checkpoint['model']
            for split in ('train', 'validation'):
                dataset = ReasoningDataset(data_dir, split)
                if not len(dataset):
                    raise ValueError(f'Empty {split} split')
                loader = DataLoader(Subset(dataset, range(min(args.examples, len(dataset)))),
                                    batch_size=args.batch_size, shuffle=False)
                with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=checkpoint['contract']['precision'] == 'bf16'):
                    section = dict(examples=min(args.examples, len(dataset)), cold=evaluate_cold_prompt(
                        model, loader, device, tokenizer=dataset.tokenizer, records=dataset.records, seed=args.seed, split=split))
                    if not args.skip_generation:
                        section['generation'] = {}
                        for condition in ('correct', 'none'):
                            metrics, details = evaluate_generation(model, loader, device, tokenizer=dataset.tokenizer,
                                records=dataset.records, seed=args.seed, memory_condition=condition,
                                policy='top_prob', candidate_k=8)
                            section['generation'][condition] = dict(metrics=metrics, examples=details,
                                diagnostics=generation_diagnostics(details, dataset.records))
                result['splits'][split] = section
                print(json.dumps(dict(split=split, cold=section['cold']['summary'],
                    generation={key: value['metrics'] for key, value in section.get('generation', {}).items()})), flush=True)
        atomic_json(output, result)
    finally:
        torch.set_num_threads(old_threads)
    print(f'Wrote read-only checkpoint audit: {output}', flush=True)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint', 'data-dir', 'output'):
        result.add_argument('--' + flag, required=True)
    result.add_argument('--trust-checkpoint', action='store_true')
    result.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    result.add_argument('--examples', type=int, default=128)
    result.add_argument('--batch-size', type=int, default=8)
    result.add_argument('--seed', type=int, default=2026)
    result.add_argument('--cpu-threads', type=int, default=4)
    result.add_argument('--skip-generation', action='store_true')
    return result


def main(argv=None):
    audit(parser().parse_args(argv))


if __name__ == '__main__':
    main()
