"""Frozen-weight Zebra clue and decoding diagnostics on validation only.

Normal decoding exactly matches evaluation.evaluate_generation's candidate-8
rule (sampled or argmax tokens). Oracle replay is NOT a generation score: it
reuses the sampled run's chosen positions and commits gold tokens instead.
Clue-derived labels and gold answers are used only for scoring, never to select
positions, restrict probabilities, or alter normal model inputs.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from .data import ReasoningDataset
from .evaluation import _generator, _probabilities
from .runner import atomic_json, digest, load_checkpoint, move_batch
from .tasks import score_prediction
from .tfw import PaperMDM
from .zebra_official import parse_prompt, relation_holds


def public_clues(record):
    meta = record['metadata']
    prefix = ['HOUSES', str(meta['houses']), 'ATTRS', str(meta['attributes'])]
    return parse_prompt(prefix + record['prompt'])


def unary_constraints(record):
    """Single-clue restrictions for one distinct entity, using public clues only."""
    houses, _, clues = public_clues(record)
    result = []
    for kind, refs in clues:
        entities = {ref for ref in refs if ref[0] == 'c'}
        if len(entities) != 1:
            continue
        entity = next(iter(entities))
        allowed = [h for h in range(houses) if relation_holds(
            kind, [h if ref[0] == 'c' else ref[2] for ref in refs], houses)]
        if not allowed:
            raise ValueError('Inconsistent single-entity public clue')
        result.append(dict(kind=kind, attribute=entity[1], value=entity[2], allowed=allowed))
    return result


def clue_scores(record, slot_tokens):
    """All clues in denominator; invalid referenced category rows count as failures.

    This is a diagnostic of the fixed content slots, not the answer/EOS parser.
    Whole-puzzle accuracy always uses the unchanged score_prediction instead.
    """
    houses, attributes, clues = public_clues(record)
    positions = {}
    valid_rows = 0
    for a in range(attributes):
        row = slot_tokens[a * houses:(a + 1) * houses]
        if len(row) == houses and set(row) == set(map(str, range(houses))):
            positions[a] = {int(value): h for h, value in enumerate(row)}
            valid_rows += 1
    counts = defaultdict(lambda: dict(total=0, satisfied=0, invalid_reference=0))
    for kind, refs in clues:
        count = counts[kind]
        count['total'] += 1
        if any(ref[0] == 'c' and ref[1] not in positions for ref in refs):
            count['invalid_reference'] += 1
            continue
        hs = [ref[2] if ref[0] == 'n' else positions[ref[1]][ref[2]] for ref in refs]
        count['satisfied'] += int(relation_holds(kind, hs, houses))
    return dict(by_type=dict(counts), valid_rows=valid_rows, rows=attributes,
                all_rows_valid=valid_rows == attributes)


def fullmask_probe(model, batch, tokenizer, records):
    current = batch['input_ids'].masked_fill(batch['target_mask'], tokenizer.mask_id)
    probs = _probabilities(model(current, attention_mask=batch['attention_mask'])['logits'], tokenizer.mask_id)
    results = []
    for row, index in enumerate(batch['record_index'].tolist()):
        record = records[index]
        houses, attributes, _ = public_clues(record)
        slots = batch['target_mask'][row].nonzero().flatten().cpu().tolist()
        if len(slots) != houses * attributes + 1:
            raise ValueError('Public target layout mismatch')
        gold = batch['input_ids'][row, slots[:-1]].cpu()
        predictions = probs[row, slots[:-1]].argmax(-1)
        restrictions = unary_constraints(record)
        direct = {item['attribute'] * houses + item['allowed'][0]: item['value']
                  for item in restrictions if item['kind'] == '=' and len(item['allowed']) == 1}
        for offset, value in direct.items():
            if gold[offset].item() != tokenizer.encode([str(value)])[0]:
                raise ValueError('Public equality/gold disagreement')
        groups = {}
        for name, offsets in (
            ('direct_equality', list(direct)),
            ('other_content', [i for i in range(len(gold)) if i not in direct]),
            ('all_content', list(range(len(gold)))),
        ):
            groups[name] = dict(tokens=len(offsets),
                correct=int((predictions[offsets] == gold[offsets]).sum()),
                nll_sum=sum(-math.log(max(float(probs[row, slots[i], gold[i]]), 1e-30)) for i in offsets),
                uniform_correct=len(offsets) / houses)
        unary = []
        for item in restrictions:
            # Diagnostic entity-house readout, not a coherent joint distribution.
            positions = slots[item['attribute'] * houses:(item['attribute'] + 1) * houses]
            value_id = tokenizer.encode([str(item['value'])])[0]
            best_house = int(probs[row, positions, value_id].argmax())
            unary.append(dict(**item, predicted_house=best_house,
                allowed_hit=best_house in item['allowed'], uniform_hit=len(item['allowed']) / houses))
        results.append(dict(record_index=index, id=record['id'], groups=groups, unary=unary))
    return results


def trace_batch(model, batch, tokenizer, records, *, seed=2026, selection='sample', replay=None):
    """Trace candidate8 decode, or oracle replay of a previously sampled order.

    Only the explicitly labelled replay branch reads gold to construct inputs.
    Public EOS slot is the final target slot, independent of hidden answer IDs.
    """
    if selection not in ('sample', 'argmax'):
        raise ValueError('Unknown token selection')
    indices = batch['record_index'].tolist()
    gold = batch['input_ids']
    current = gold.masked_fill(batch['target_mask'], tokenizer.mask_id)
    slots = [row.nonzero().flatten().cpu().tolist() for row in batch['target_mask']]
    schedules, generators = [], []
    for index, positions in zip(indices, slots):
        order = torch.randperm(len(positions), generator=_generator(seed, index, 'order'))
        schedules.append([positions[i] for i in order.tolist()])
        generators.append(_generator(seed, index, 'tokens'))
    if replay is not None:
        if len(replay) != len(indices):
            raise ValueError('Replay batch size differs')
        for row, reference in enumerate(replay):
            if reference['record_index'] != indices[row] or sorted(reference['decode_order']) != slots[row]:
                raise ValueError('Replay must contain every public target exactly once')
            schedules[row] = list(reference['decode_order'])
    events = [[] for _ in indices]
    for step in range(max(map(len, slots))):
        probs = _probabilities(model(current, attention_mask=batch['attention_mask'])['logits'], tokenizer.mask_id)
        for row, schedule in enumerate(schedules):
            if not schedule:
                continue
            candidates = schedule[:1] if replay is not None else schedule[:8]
            position = candidates[int(probs[row, candidates].amax(-1).argmax())]
            distribution = probs[row, position]
            greedy = int(distribution.argmax())
            proposal = greedy if selection == 'argmax' or replay is not None else int(
                torch.multinomial(distribution, 1, generator=generators[row]))
            target = int(gold[row, position])  # Scoring only, except explicit oracle branch.
            committed = proposal if replay is None else target
            current[row, position] = committed
            schedule.remove(position)
            events[row].append(dict(step=step + 1, position=position, offset=slots[row].index(position),
                content=position != slots[row][-1], proposal=proposal, committed=committed,
                target=target, proposal_correct=proposal == target, argmax_correct=greedy == target,
                committed_correct=committed == target, gold_probability=float(distribution[target]),
                confidence=float(distribution.max())))
    results = []
    for row, index in enumerate(indices):
        raw = tokenizer.decode(current[row, slots[row]].cpu().tolist())
        result = dict(record_index=index, id=records[index]['id'], events=events[row],
            decode_order=[e['position'] for e in events[row]], predicted_answer_slots=raw)
        if replay is None:
            result.update(scores=score_prediction(records[index], raw), clues=clue_scores(records[index], raw))
        # Oracle final output is trivially gold; deliberately no solve-rate field.
        results.append(result)
    return results


def _rate(num, den):
    return num / den if den else None


def aggregate(probes, conditions):
    groups, unary = {}, {}
    for name in ('direct_equality', 'other_content', 'all_content'):
        counts = {k: sum(p['groups'][name][k] for p in probes)
                  for k in ('tokens', 'correct', 'nll_sum', 'uniform_correct')}
        groups[name] = dict(**counts, accuracy=_rate(counts['correct'], counts['tokens']),
                           nll=_rate(counts['nll_sum'], counts['tokens']))
    for p in probes:
        for item in p['unary']:
            row = unary.setdefault(item['kind'], dict(clues=0, allowed_hits=0, uniform_hits=0))
            row['clues'] += 1; row['allowed_hits'] += int(item['allowed_hit']); row['uniform_hits'] += item['uniform_hit']
    summaries = {}
    for name, examples in conditions.items():
        rank_counts, clue_counts = {}, {}
        first_errors, total, correct, argmax, nll = [], 0, 0, 0, 0.
        for example in examples:
            content = [e for e in example['events'] if e['content']]
            errors = [i + 1 for i, e in enumerate(content) if not e['proposal_correct']]
            first_errors.append(errors[0] if errors else None)
            for rank, event in enumerate(content, 1):
                c = rank_counts.setdefault(str(rank), dict(tokens=0, correct=0, argmax_correct=0, nll_sum=0.))
                c['tokens'] += 1; c['correct'] += event['proposal_correct']; c['argmax_correct'] += event['argmax_correct']
                loss = -math.log(max(event['gold_probability'], 1e-30)); c['nll_sum'] += loss
                total += 1; correct += event['proposal_correct']; argmax += event['argmax_correct']; nll += loss
            for kind, counts in example.get('clues', {}).get('by_type', {}).items():
                c = clue_counts.setdefault(kind, dict(total=0, satisfied=0, invalid_reference=0))
                for key in c:
                    c[key] += counts[key]
        summary = dict(content_tokens=total, proposal_accuracy=_rate(correct, total),
            argmax_accuracy=_rate(argmax, total), selected_content_nll=_rate(nll, total),
            by_content_rank=rank_counts, clue_satisfaction=clue_counts)
        if name != 'oracle_replay':
            score_keys = sorted({k for e in examples for k, v in e['scores'].items() if isinstance(v, (bool, int, float))})
            summary.update(scores={k: sum(float(e['scores'].get(k, 0)) for e in examples) / len(examples) for k in score_keys},
                all_rows_valid=sum(e['clues']['all_rows_valid'] for e in examples) / len(examples),
                first_content_error_counts={str(rank): first_errors.count(rank) for rank in range(1, 37)},
                no_content_errors=first_errors.count(None),
                prefix_survival={str(k): sum(err is None or err > k for err in first_errors) / len(examples)
                                 for k in (1, 2, 4, 8)})
        summaries[name] = summary
    # Same selected positions, same mask pattern, different revealed token VALUES.
    deltas, token_n, token_delta, nll_delta = [], 0, 0, 0.
    for actual, oracle in zip(conditions['sample'], conditions['oracle_replay']):
        if actual['id'] != oracle['id'] or actual['decode_order'] != oracle['decode_order']:
            raise ValueError('Unpaired oracle replay')
        earlier_error, values = False, []
        for a, o in zip(actual['events'], oracle['events']):
            if earlier_error and a['content']:
                value = int(o['argmax_correct']) - int(a['argmax_correct'])
                values.append(value); token_n += 1; token_delta += value
                nll_delta += math.log(max(a['gold_probability'], 1e-30)) - math.log(max(o['gold_probability'], 1e-30))
            earlier_error |= not a['committed_correct']  # Includes previously wrong EOS slot.
        if values:
            deltas.append(sum(values) / len(values))
    mean = sum(deltas) / len(deltas) if deltas else None
    se = math.sqrt(sum((x-mean)**2 for x in deltas) / (len(deltas)-1) / len(deltas)) if len(deltas) > 1 else None
    return dict(fullmask_groups=groups, fullmask_single_clue_readouts=unary, conditions=summaries,
        oracle_after_first_error=dict(examples=len(deltas), tokens=token_n,
            paired_example_mean_argmax_gain=mean, paired_example_se=se,
            token_weighted_argmax_gain=_rate(token_delta, token_n),
            token_weighted_nll_delta=_rate(nll_delta, token_n)))


def run(args):
    torch.set_num_threads(4)
    started = time.monotonic()
    output = Path(args.output_dir)
    if (output / 'audit.json').exists():
        raise FileExistsError('Completed audit exists; select a new output directory')
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    checkpoint = load_checkpoint(checkpoint_path)
    dataset = ReasoningDataset(args.data_dir, 'validation')
    if checkpoint['contract']['data_sha256'] != digest(Path(args.data_dir) / 'manifest.json'):
        raise ValueError('Checkpoint/data manifest mismatch')
    device = torch.device(args.device)
    model = PaperMDM(checkpoint['model_config']).to(device).eval()
    if model.logit_shift != 0 or model.target_region != 'answer' or model.padding_attention != 'masked':
        raise ValueError('Only repaired same-position, answer-only models supported')
    model.load_state_dict(checkpoint['model'], strict=True)
    count = min(args.examples, len(dataset))
    if count < 1:
        raise ValueError('At least one validation example required')
    loader = DataLoader(Subset(dataset, range(count)), batch_size=args.batch_size, shuffle=False)
    provenance = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=digest(checkpoint_path),
        step=checkpoint['step'], model_config=checkpoint['model_config'],
        data_sha256=checkpoint['contract']['data_sha256'], split='validation', examples=count,
        seed=args.seed, batch_size=args.batch_size, device=str(device),
        protocol='zebra-trace-audit-v1', precision='bf16' if device.type == 'cuda' else 'fp32')
    del checkpoint
    probes, conditions = [], dict(sample=[], argmax=[], oracle_replay=[])
    atomic_json(output / 'status.json', dict(status='running', completed_examples=0, **provenance))
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for batch in loader:
            batch = move_batch(batch, device)
            probes.extend(fullmask_probe(model, batch, dataset.tokenizer, dataset.records))
            sampled = trace_batch(model, batch, dataset.tokenizer, dataset.records, seed=args.seed)
            conditions['sample'].extend(sampled)
            conditions['argmax'].extend(trace_batch(model, batch, dataset.tokenizer, dataset.records, seed=args.seed, selection='argmax'))
            conditions['oracle_replay'].extend(trace_batch(model, batch, dataset.tokenizer, dataset.records, seed=args.seed, replay=sampled))
            status = dict(status='running', completed_examples=len(probes), elapsed_seconds=time.monotonic()-started, **provenance)
            atomic_json(output / 'status.json', status)
            print(json.dumps({k: status[k] for k in ('status', 'completed_examples', 'elapsed_seconds')}), flush=True)
    result = dict(**provenance, summary=aggregate(probes, conditions), probes=probes, conditions=conditions,
        notes=['Oracle replay commits gold and is NOT a solve rate.',
               'Clue diagnostics do not constrain decoding or provide solver inputs.',
               'Single-clue entity-house readout is not a joint distribution.',
               'Invalid referenced category rows count as unsatisfied clues.',
               'Validation-only diagnosis; no training, test selection, or benchmark reproduction claim.'])
    atomic_json(output / 'audit.json', result)
    atomic_json(output / 'summary.json', dict(**provenance, **result['summary'], notes=result['notes']))
    atomic_json(output / 'status.json', dict(status='finished', completed_examples=count,
        elapsed_seconds=time.monotonic()-started, **provenance))
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--examples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=2026)
    run(parser.parse_args())
