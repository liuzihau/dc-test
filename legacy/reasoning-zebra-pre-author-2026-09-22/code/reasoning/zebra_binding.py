"""Synthetic clue-binding diagnostic, NOT the released Zebra benchmark.

Four training families teach direct and one-hop rules. Two held-out graph
families require longer composition. All use existing tokenizer/model inputs;
there are no solver features or extra supervision heads. Never modifies the
real Zebra cache. Board-disjoint splits forbid memorizing a held-out table via
another clue rendering.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from .data import ReasoningDataset, encode_record, write_prepared_dataset
from .evaluation import evaluate_generation, _probabilities
from .runner import atomic_json, digest, load_checkpoint, move_batch
from .tasks import TaskTokenizer, validate_record
from .tfw import PaperMDM
from .zebra_official import parse_prompt, relation_holds


TRAIN_FAMILIES = ('direct_equality', 'direct_adjacency', 'equality_star', 'adjacency_pairs')
OOD_FAMILIES = ('equality_chain', 'adjacency_chain')
FAMILIES = TRAIN_FAMILIES + OOD_FAMILIES
KIND = 'synthetic_clue_binding_v1'
EQUALITY_KIND = 'synthetic_equality_control_v2'
EQUALITY_FAMILIES = ('direct_equality', 'equality_star', 'equality_chain')


def board_key(board):
    return hashlib.sha256(json.dumps(board, separators=(',', ':')).encode()).hexdigest()


def render(board, family, rng):
    """Exactly one binary clue per cell; clue order never carries answer order."""
    if family not in FAMILIES:
        raise ValueError('Unknown clue family')
    attributes, houses = len(board), len(board[0])
    if (houses, attributes) != (5, 5) or any(sorted(row) != list(range(houses)) for row in board):
        raise ValueError('Binding v1 deliberately uses fixed 5x5 permutation tables')
    clues = []
    def entity(a, h):
        return ('c', a, board[a][h])
    def house(h):
        return ('n', 0, h)
    def add(kind, a, b):
        if kind == '=' and rng.random() < .5:
            a, b = b, a
        clues.append((kind, (a, b)))
    if family == 'direct_equality':
        for a in range(attributes):
            for h in range(houses):
                add('=', entity(a, h), house(h))
    elif family in ('equality_star', 'equality_chain'):
        categories = list(range(attributes)); rng.shuffle(categories)
        root = categories[0]
        for h in range(houses):
            add('=', entity(root, h), house(h))
        for j, a in enumerate(categories[1:], 1):
            parent = root if family == 'equality_star' else categories[j-1]
            for h in range(houses):
                add('=', entity(a, h), entity(parent, h))
    else:
        for a in range(attributes):
            anchor = rng.randrange(houses)
            for h in range(houses):
                if family == 'direct_adjacency':
                    # Randomly use either valid one-step numeric-house reference.
                    choices = []
                    if h > 0: choices.append((house(h-1), entity(a, h)))
                    if h + 1 < houses: choices.append((entity(a, h), house(h+1)))
                    add('immediate-left', *rng.choice(choices))
                elif family == 'adjacency_pairs':
                    if h % 2 == 0:
                        add('=', entity(a, h), house(h))
                    else:
                        add('immediate-left', entity(a, h-1), entity(a, h))
                elif family == 'adjacency_chain':
                    if h == 0:
                        add('=', entity(a, anchor), house(anchor))
                    else:
                        add('immediate-left', entity(a, h-1), entity(a, h))
    rng.shuffle(clues)
    prompt = []
    for kind, (lhs, rhs) in clues:
        prompt.extend([kind, 'LHS'] + list(map(str, lhs)) + ['RHS'] + list(map(str, rhs)) + ['CLUE_END'])
    record = dict(task='zebra-benchmark', prompt=prompt,
        answer=[str(value) for row in board for value in row],
        metadata=dict(houses=houses, attributes=attributes, source=KIND, family=family,
                      board_sha256=board_key(board), solver_order_used=False,
                      source_answer_trace_used=False, benchmark_equivalence=False))
    record['id'] = hashlib.sha256(json.dumps(prompt, separators=(',', ':')).encode()).hexdigest()
    return validate_record(record)


def certified_solution(record):
    """Preparation/test-only singleton propagation; NEVER used as model input.

    If all domains become singletons through sound constraint propagation, the
    consistent solution is unique. Fail closed otherwise. Supports our '=' and
    directed adjacency subset, not a general Zebra solver.
    """
    meta = record['metadata']
    h, a, clues = parse_prompt(['HOUSES', str(meta['houses']), 'ATTRS', str(meta['attributes'])] + record['prompt'])
    domains = {('c', i, v): set(range(h)) for i in range(a) for v in range(h)}
    domains.update({('n', 0, j): {j} for j in range(h)})
    for _ in range(2 * h * a):
        before = {k: frozenset(v) for k, v in domains.items()}
        for kind, refs in clues:
            if kind not in ('=', 'immediate-left') or len(refs) != 2:
                raise ValueError('Outside certified diagnostic relation subset')
            left, right = refs
            pairs = [(x, y) for x in domains[left] for y in domains[right] if relation_holds(kind, [x, y], h)]
            domains[left].intersection_update(x for x, _ in pairs)
            domains[right].intersection_update(y for _, y in pairs)
        for i in range(a):
            fixed = [next(iter(domains[('c', i, v)])) for v in range(h) if len(domains[('c', i, v)]) == 1]
            if len(fixed) != len(set(fixed)):
                raise ValueError('Conflicting singleton houses')
            for v in range(h):
                domain = domains[('c', i, v)]
                if len(domain) > 1:
                    domain.difference_update(fixed)
        if any(not d for d in domains.values()):
            raise ValueError('Inconsistent diagnostic puzzle')
        if before == {k: frozenset(v) for k, v in domains.items()}:
            break
    answer = [['?'] * h for _ in range(a)]
    for i in range(a):
        for v in range(h):
            if len(domains[('c', i, v)]) != 1:
                raise ValueError('Uniqueness not certified')
            answer[i][next(iter(domains[('c', i, v)]))] = str(v)
    return sum(answer, [])


def prepare(path, train_boards=5120, validation_boards=128, test_boards=256, seed=71023):
    path = Path(path)
    recipe = dict(kind=KIND, benchmark_equivalence=False, seed=seed,
        train_boards=train_boards, validation_boards=validation_boards, test_boards=test_boards,
        training_families=list(TRAIN_FAMILIES), held_out_graph_families=list(OOD_FAMILIES),
        split_policy='whole-board-disjoint; paired clue families within each split',
        shape=[5, 5], prompt_order='randomized; no solver trace',
        answer_uniqueness='certified by public-clue-only propagation for every row')
    if path.exists():
        existing = json.loads((path/'manifest.json').read_text())
        if existing['source'] != json.loads(json.dumps(recipe)):
            raise ValueError('Existing diagnostic data recipe differs')
        for split in ('train', 'validation', 'test'):
            ReasoningDataset(path, split)
        return existing
    rng, seen, splits = random.Random(seed), set(), {}
    tok = TaskTokenizer('zebra-benchmark')
    for split, n in (('train', train_boards), ('validation', validation_boards), ('test', test_boards)):
        rows = []
        for _ in range(n):
            while True:
                board = [rng.sample(range(5), 5) for _ in range(5)]
                key = board_key(board)
                if key not in seen:
                    seen.add(key); break
            for family in TRAIN_FAMILIES if split == 'train' else FAMILIES:
                record = render(board, family, rng)
                if certified_solution(record) != record['answer']:
                    raise ValueError('Certified solution differs from generated label')
                _, _, used = encode_record(record, tok, 384)
                if used != 277:
                    raise ValueError('Matched clue count/length changed')
                rows.append(record)
        rng.shuffle(rows)
        splits[split] = rows
    return write_prepared_dataset(path, 'zebra-benchmark', splits, seed, recipe)


def prepare_equality_control(path, train_boards=4096, validation_boards=128,
                             test_boards=256, seed=81727):
    """Corrected equality-only control; keep flawed v1 data immutable."""
    path = Path(path)
    recipe = dict(kind=EQUALITY_KIND, benchmark_equivalence=False, seed=seed,
        train_boards=train_boards, validation_boards=validation_boards,
        test_boards=test_boards, training_families=['direct_equality'],
        evaluation_families=list(EQUALITY_FAMILIES),
        held_out_graph_families=['equality_star', 'equality_chain'],
        split_policy='whole-board-disjoint; paired evaluation renderings', shape=[5, 5],
        prompt_order='randomized; no solver trace',
        answer_uniqueness='certified by public-clue-only propagation for every row',
        correction='direct_equality contains 25 c-to-n equality clues; v1 preserved')
    if path.exists():
        existing = json.loads((path/'manifest.json').read_text())
        if existing['source'] != recipe:
            raise ValueError('Existing equality-control recipe differs')
        for split in ('train', 'validation', 'test'):
            ReasoningDataset(path, split)
        return existing
    rng, seen, splits = random.Random(seed), set(), {}
    tok = TaskTokenizer('zebra-benchmark')
    for split, n in (('train', train_boards), ('validation', validation_boards),
                     ('test', test_boards)):
        families = ('direct_equality',) if split == 'train' else EQUALITY_FAMILIES
        rows = []
        for _ in range(n):
            while True:
                board = [rng.sample(range(5), 5) for _ in range(5)]
                key = board_key(board)
                if key not in seen:
                    seen.add(key); break
            for family in families:
                record = render(board, family, rng)
                record['metadata']['source'] = EQUALITY_KIND
                if certified_solution(record) != record['answer']:
                    raise ValueError('Certified solution differs from generated equality label')
                if encode_record(record, tok, 384)[2] != 277:
                    raise ValueError('Equality control clue count/length changed')
                rows.append(record)
        rng.shuffle(rows); splits[split] = rows
    return write_prepared_dataset(path, 'zebra-benchmark', splits, seed, recipe)


def evaluate(args):
    torch.set_num_threads(4)
    data = ReasoningDataset(args.data_dir, args.split)
    source = data.manifest['source']
    if source.get('kind') not in (KIND, EQUALITY_KIND):
        raise ValueError('Evaluation requires isolated synthetic diagnostic data')
    path = Path(args.checkpoint).resolve(strict=True)
    ckpt = load_checkpoint(path)
    if ckpt['contract']['data_sha256'] != digest(Path(args.data_dir)/'manifest.json'):
        raise ValueError('Data/checkpoint contract mismatch')
    output = Path(args.output)
    provenance = dict(checkpoint=str(path), checkpoint_sha256=digest(path), step=ckpt['step'],
        data_sha256=ckpt['contract']['data_sha256'], split=args.split, seed=2026,
        protocol=source['kind'], benchmark_equivalence=False, model_config=ckpt['model_config'])
    if output.exists():
        saved = json.loads(output.read_text())
        if any(saved[k] != v for k, v in provenance.items()):
            raise ValueError('Existing diagnostic report provenance differs')
        return
    device = torch.device(args.device)
    model = PaperMDM(ckpt['model_config']).to(device).eval()
    model.load_state_dict(ckpt['model'], strict=True)
    del ckpt
    loader = DataLoader(data, batch_size=32, shuffle=False)
    families = tuple(source.get('evaluation_families', FAMILIES))
    trained = tuple(source['training_families'])
    groups = {k: dict(examples=0, content_tokens=0, fullmask_correct=0,
        fullmask_exact=0, fullmask_nll_sum=0.) for k in families}
    reports = {}
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
        for batch in loader:
            batch = move_batch(batch, device)
            ids, target = batch['input_ids'], batch['target_mask']
            logits = model(ids.masked_fill(target, model.mask_id), attention_mask=batch['attention_mask'])['logits']
            probs = _probabilities(logits, model.mask_id)
            for row, index in enumerate(batch['record_index'].tolist()):
                positions = target[row].nonzero().flatten().cpu()[:-1]  # Fixed public EOS slot last.
                gold = ids[row, positions].cpu()
                p = probs[row, positions]
                correct = p.argmax(-1).eq(gold)
                group = groups[data.records[index]['metadata']['family']]
                group['examples'] += 1; group['content_tokens'] += len(gold)
                group['fullmask_correct'] += int(correct.sum()); group['fullmask_exact'] += int(correct.all())
                group['fullmask_nll_sum'] += float(-p[torch.arange(len(gold)), gold].clamp_min(1e-30).log().sum())
        selections = ('argmax', 'sample') if args.split == 'test' else ('argmax',)
        for selection in selections:
            _, details = evaluate_generation(model, loader, device, tokenizer=data.tokenizer,
                records=data.records, seed=2026, candidate_k=8, token_selection=selection)
            for ex in details:
                family = data.records[ex['record_index']]['metadata']['family']
                ex['family'] = family
                group = groups[family]
                for score in ('valid_solution', 'format_success'):
                    key = selection + '_' + score
                    group[key] = group.get(key, 0) + int(ex['scores'][score])
            reports[selection] = details
    for family, group in groups.items():
        group['trained_family'] = family in trained
        group['fullmask_accuracy'] = group['fullmask_correct']/group['content_tokens']
        group['fullmask_nll'] = group['fullmask_nll_sum']/group['content_tokens']
        for selection in selections:
            group[selection+'_solve_rate'] = group[selection+'_valid_solution']/group['examples']
    atomic_json(output, dict(**provenance, groups=groups, reports=reports,
        note='Synthetic 5x5 board-disjoint diagnostic. Chains are unseen graph structures; NOT original Zebra accuracy.'))
    print(json.dumps(dict(step=provenance['step'], split=args.split, groups=groups), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--data-dir', required=True)
    p.add_argument('--equality-control', action='store_true')
    p = sub.add_parser('evaluate')
    p.add_argument('--data-dir', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True); p.add_argument('--split', choices=('validation', 'test'), default='validation')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = prepare_equality_control(args.data_dir) if args.equality_control else prepare(args.data_dir)
        print(json.dumps(manifest['source'], indent=2))
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
