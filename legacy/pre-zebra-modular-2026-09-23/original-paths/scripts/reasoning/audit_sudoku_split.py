#!/usr/bin/env python3
"""Read-only audit of saved Sudoku generations; writes only audit reports.

No model execution, GPU use, training mutation, or score-based selection.
The independent scorer below does not call the project's Sudoku solver.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

VARIANTS = ('mdm', 'tt', 'tt_ea', 'tt_ea_np', 'tt_ea_rm', 'tt_ea_rm_np')
LABELS = ('MDM', 'TT', 'TT+EA', 'TT+EA+NP', 'TT+EA+RM', 'TT+EA+RM+NP')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def csv_file(path, rows):
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)


def valid_grid(tokens):
    if len(tokens) != 81 or any(t not in set('123456789') for t in tokens):
        return False
    board = np.asarray(tokens).reshape(9, 9)
    units = [board[i, :] for i in range(9)] + [board[:, j] for j in range(9)]
    units += [board[r:r+3, c:c+3].flatten() for r in (0, 3, 6) for c in (0, 3, 6)]
    return all(set(u) == set('123456789') for u in units)


def paired(a, b, seed=2026):
    delta = b.astype(int)-a.astype(int)
    gained, lost = int((delta == 1).sum()), int((delta == -1).sum())
    n = gained+lost
    p = min(1., 2*sum(math.comb(n, k) for k in range(min(gained, lost)+1))/2**n) if n else 1.
    # Paired-example bootstrap: multinomial counts of {-1,0,+1} differences.
    counts = np.random.default_rng(seed).multinomial(len(a),
        [(delta == -1).mean(), (delta == 0).mean(), (delta == 1).mean()], size=20000)
    boot = (counts[:, 2]-counts[:, 0])*100/len(a)
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(gained=gained, lost=lost, delta_pp=float(delta.mean()*100),
                ci95_low=float(lo), ci95_high=float(hi), paired_exact_p=p)


def training_rows(run):
    rows = {}
    for attempt in sorted((run/'logs').glob('attempt-*')):
        meta = read_json(attempt/'resume_attempt.json')
        rows = {s: r for s, r in rows.items() if s <= meta['resume_step']}
        with (attempt/'metrics.csv').open() as f:
            for row in csv.DictReader(f):
                if row.get('train/loss'):
                    rows[int(float(row['step']))] = row
    return rows


def audit(root, data, output, scan_train=True):
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(data/'manifest.json')
    for split in manifest['splits'].values():
        files = split['files'].values() if 'files' in split else [split]
        for item in files:
            assert sha256(data/item['filename']) == item['sha256']
    records = {s: [json.loads(l) for l in (data/manifest['splits'][s]['filename']).open()]
               for s in ('test', 'validation')}
    gold = {r['id']: r for r in records['test']}
    expected_ids = list(gold)
    assert len(expected_ids) == len(records['test']) == 1000
    summaries, details, training, validation, survivals, integrity = [], [], [], [], [], []
    correct = {}
    vocab = manifest['vocab']
    for variant, label in zip(VARIANTS, LABELS):
        run = root/variant
        g = read_json(run/'generation.json')
        contract, launch = read_json(run/'contract.json'), read_json(run/'launch.json')
        assert g['contract'] == contract and g['step'] == 14090
        assert contract['data_sha256'] == sha256(data/'manifest.json')
        assert contract['epochs'] == 1 and contract['epoch_examples'] == 1803463
        assert all(valid_grid(r['answer']) and all(t == '[MASK]' or t == r['answer'][i]
            for i, t in enumerate(r['prompt'])) for r in records['test'])
        assert [e['id'] for e in g['examples']] == expected_ids
        assert g['arguments']['batch_size'] == 32
        for k, v in dict(policy='top_prob', candidate_k=8, token_selection='paper',
                         tokens_per_step=1, memory_condition='correct', seed=2026,
                         num_examples=1000, mean_nfe_per_example=82.).items():
            assert g['metrics'][k] == v, (variant, k)
        checkpoint = Path(g['checkpoint'])
        receipt = read_json(checkpoint.with_suffix('.pt.json'))
        assert receipt['size'] == checkpoint.stat().st_size and receipt['sha256'] == sha256(checkpoint)
        integrity.append(dict(variant=variant, checkpoint=str(checkpoint), sha256=receipt['sha256'],
                              generation_sha256=sha256(run/'generation.json'), all_checks_pass=True))
        selected = []
        for e in g['examples']:
            r = gold[e['id']]
            pred = e['predicted_answer_slots']
            assert [vocab[i] for i in e['predicted_answer_ids']] == pred
            assert e['token_selection'] == 'sample' and e['nfe'] == 82
            assert sorted(e['decode_order']) == list(range(83, 165))
            assert e['remaining_masked_slots'] == 0
            grid = pred[:81]
            exact = grid == r['answer']
            valid = valid_grid(grid)
            given = [i for i, t in enumerate(r['prompt']) if t != '[MASK]']
            kept = all(grid[i] == r['prompt'][i] for i in given)
            solved = valid and kept
            scores = dict(exact_match=exact, valid_solution=solved, clues_preserved=kept,
                          constraints_satisfied=valid, format_success=len(pred) == 82 and pred[-1] == '[EOS]')
            # The saved scorer omits the clue diagnostic for malformed grids;
            # its aggregate defaults these to zero. Check the solve metric
            # independently, and retain actual clue agreement separately.
            assert all(e['scores'][k] == v for k, v in scores.items()
                       if k != 'clues_preserved'), (variant, e['id'])
            if 'clues_preserved' in e['scores']:
                assert e['scores']['clues_preserved'] == kept
            wrong = {i for i in range(81) if grid[i] != r['answer'][i]}
            first = next((s for s, p in enumerate(e['decode_order'], 1) if p-83 in wrong), None)
            first_pos = e['decode_order'][first-1]-83 if first else None
            row = dict(variant=variant, id=e['id'], givens=len(given), solved=int(solved),
                       exact=int(exact), valid_grid=int(valid), clues_preserved=int(kept),
                       invalid_grid_symbols=int(any(t not in set('123456789') for t in grid)),
                       saved_clue_diagnostic_missing=int('clues_preserved' not in e['scores']),
                       wrong_cells=len(wrong), wrong_given_cells=len(wrong.intersection(given)),
                       wrong_unknown_cells=len(wrong.difference(given)),
                       first_reference_error=first or 83,
                       first_error_is_given=int(first_pos in given) if first else None)
            selected.append(row); details.append(row)
        correct[variant] = np.asarray([bool(r['solved']) for r in selected])
        failures = [r for r in selected if not r['solved']]
        token_acc = 1-sum(r['wrong_cells'] for r in selected)/81000
        unknown_acc = 1-sum(r['wrong_unknown_cells'] for r in selected)/sum(81-r['givens'] for r in selected)
        last_val = read_json(run/'validation/step-000014090.json')
        summary = dict(variant=variant, label=label, parameters=launch['parameters'],
            solved=int(correct[variant].sum()), accuracy=float(correct[variant].mean()),
            generated_cell_accuracy=token_acc, generated_unknown_accuracy=unknown_acc,
            val_r70_accuracy=last_val['ratios']['0.7']['content_masked_token_accuracy'],
            val_r70_nll=last_val['ratios']['0.7']['content_conditional_nll'],
            valid_grid_wrong_clues=sum(r['valid_grid'] and not r['clues_preserved'] for r in selected),
            independently_clues_preserved=sum(r['clues_preserved'] for r in selected)/1000,
            saved_clues_preserved=g['metrics']['clues_preserved'],
            invalid_grid_symbols=sum(r['invalid_grid_symbols'] for r in selected),
            invalid_grid_correct_clues=sum(not r['valid_grid'] and r['clues_preserved'] for r in selected),
            invalid_grid_wrong_clues=sum(not r['valid_grid'] and not r['clues_preserved'] for r in selected),
            first_error_median_failed=statistics.median(r['first_reference_error'] for r in failures),
            errors_by_reveal24=sum(r['first_reference_error'] <= 24 for r in failures),
            first_errors_on_given=sum(r['first_error_is_given'] or 0 for r in failures),
            inference_seconds=g['metrics']['latency_seconds'])
        summaries.append(summary)
        for t in range(83):
            survivals.append(dict(variant=variant, reveals=t,
                no_reference_error_fraction=sum(r['first_reference_error'] > t for r in selected)/1000))
        rows = training_rows(run); last = max(rows)
        recent = [r for s, r in rows.items() if s > last-500]
        summary['logged_updates_last500'] = len(recent)
        for k in ('train/accuracy_full', 'train/loss_full', 'train/loss_t0', 'train/loss_t1',
                  'train/loss_t2', 'train/loss_t3', 'train/base_loss', 'train/neighbor_loss',
                  'train/identity_loss', 'train/num_forwards', 'train/adjacent_edges',
                  'train/cache_only_fraction_t2', 'train/current_only_fraction_t2',
                  'train/final_dropout', 'grad_norm', 'seconds_per_update'):
            values = [float(r[k]) for r in recent if r.get(k)]
            if values:
                summary['last500_'+k] = statistics.mean(values)
        summary['last500_clip_fraction'] = statistics.mean(float(r['grad_norm']) > 1 for r in recent)
        summary['approx_train_seconds_logged'] = sum(float(r['seconds_per_update'])*10 for r in rows.values())
        for s, r in rows.items():
            training.append(dict(variant=variant, step=s, **{k: r[k] for k in r if k.startswith('train/') or k=='grad_norm'}))
        for p in sorted((run/'validation').glob('step-*.json')):
            v = read_json(p); step = int(p.stem.split('-')[-1])
            for protocol, doc in (('cold', v), ('nested', v['nested_teacher_forced'])):
                for ratio, m in doc['ratios'].items():
                    validation.append(dict(variant=variant, step=step, protocol=protocol, ratio=ratio,
                                           nll=m['content_conditional_nll'], accuracy=m['content_masked_token_accuracy']))
    pairs = [dict(reference=a, condition=b, **paired(correct[a], correct[b])) for a, b in (
        ('mdm','tt'), ('tt','tt_ea'), ('tt_ea','tt_ea_np'), ('tt_ea','tt_ea_rm'),
        ('tt_ea_np','tt_ea_rm_np'), ('tt_ea_rm','tt_ea_rm_np'))]
    overlap = {}
    if scan_train:
        n = manifest['splits']['train']['records']
        paths = manifest['splits']['train']['files']
        tokens = np.memmap(data/paths['tokens']['filename'], mode='r', dtype='u1', shape=(n,192))
        ids = np.memmap(data/paths['identities']['filename'], mode='r', dtype='V32', shape=(n,))
        layout = np.memmap(data/paths['layout']['filename'], mode='r', dtype='<u2', shape=(n,2))
        assert np.all(layout[:,0] == 83) and np.all(layout[:,1] == 165)
        held_prompts = {hashlib.sha256(bytes(0 if t=='[MASK]' else int(t) for t in r['prompt'])).digest(): (s,r['id'])
                        for s, rs in records.items() for r in rs}
        held_solutions = {}
        for s, rs in records.items():
            for r in rs:
                key = bytes(vocab.index(t) for t in r['answer'])
                held_solutions.setdefault(key, []).append((s,r['id']))
        prompt_hits, solution_hits = [], set()
        invalid_solutions = inconsistent_givens = 0
        expected_digits = np.asarray([vocab.index(str(i)) for i in range(1,10)])
        mask_id = vocab.index('[MASK]')
        for start in range(0,n,32768):
            block = np.ascontiguousarray(tokens[start:start+32768,83:164])
            boards = block.reshape(-1,9,9)
            boxes = boards.reshape(-1,3,3,3,3).transpose(0,1,3,2,4).reshape(-1,9,9)
            good = ((np.sort(boards,axis=2) == expected_digits).all(axis=(1,2)) &
                    (np.sort(boards.transpose(0,2,1),axis=2) == expected_digits).all(axis=(1,2)) &
                    (np.sort(boxes,axis=2) == expected_digits).all(axis=(1,2)))
            invalid_solutions += int((~good).sum())
            clues = tokens[start:start+32768,1:82]
            inconsistent_givens += int(((clues != mask_id) & (clues != block)).any(axis=1).sum())
            for j, row in enumerate(block):
                key = row.tobytes()
                if key in held_solutions: solution_hits.update(held_solutions[key])
                pkey = ids[start+j].tobytes()
                if pkey in held_prompts: prompt_hits.append((start+j, held_prompts[pkey]))
        overlap = dict(scanned_train_rows=n, exact_prompt_overlaps=prompt_hits,
                       exact_solution_overlap_holdouts=sorted(solution_hits),
                       invalid_train_solution_grids=invalid_solutions,
                       train_prompts_inconsistent_with_solution=inconsistent_givens,
                       note='Exact byte matches only; not Sudoku symmetries or near-duplicates.')
    payload = dict(summary=summaries, paired_comparisons=pairs, integrity=integrity,
                   data_overlap=overlap, source=str(root.resolve()), data=str(data.resolve()),
                   caveat='One training seed and one generation seed; bootstrap is over the frozen test examples, not retraining seeds.')
    (output/'audit.json').write_text(json.dumps(payload, indent=2)+'\n')
    for filename, rows in (('summary.csv',summaries), ('paired.csv',pairs), ('per_example.csv',details),
                           ('validation.csv',validation), ('training.csv',training), ('survival.csv',survivals)):
        csv_file(output/filename, rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(17,4.8))
    bars = axes[0].bar(range(6), [r['accuracy']*100 for r in summaries], color=plt.cm.tab10.colors[:6])
    axes[0].bar_label(bars, labels=[str(r['solved'])+'/1000' for r in summaries], fontsize=8)
    axes[0].set(xticks=range(6), xticklabels=[s.replace('+','\n+') for s in LABELS], ylim=(0,105),
                ylabel='Solved puzzles (%)', title='Identical test / decoding protocol')
    for variant, label, color in zip(VARIANTS,LABELS,plt.cm.tab10.colors):
        rows=[r for r in survivals if r['variant']==variant]
        axes[1].plot([r['reveals'] for r in rows],[r['no_reference_error_fraction']*100 for r in rows],label=label,color=color)
        rows=[r for r in validation if r['variant']==variant and r['ratio']=='0.7' and r['protocol']=='cold']
        axes[2].plot([r['step'] for r in rows],[r['nll'] for r in rows],label=label,color=color)
    axes[1].axvline(24,ls=':',color='gray'); axes[1].set(xlabel='Generated answer tokens (including EOS)',ylabel='No reference error yet (%)',title='Error-free trajectory survival')
    axes[1].legend(fontsize=8)
    axes[2].set(xlabel='Optimizer updates',ylabel='Content conditional NLL',yscale='log',title='Cold 70%-mask validation (not solving)')
    fig.tight_layout(); fig.savefig(output/'sudoku_audit.png',dpi=180); plt.close(fig)
    print(json.dumps(dict(summary=[{k:r[k] for k in ('variant','solved','generated_cell_accuracy','valid_grid_wrong_clues','first_error_median_failed','first_errors_on_given','last500_clip_fraction')} for r in summaries],
                         pairs=pairs,overlap=overlap,output=str(output)),indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/reasoning/full-epoch-split-six-2x3090-mb32/sudoku-benchmark'))
    p.add_argument('--data', type=Path, default=Path('.cache/reasoning/sudoku-benchmark-full-v1'))
    p.add_argument('--output', type=Path, default=Path('results/generated/audits/sudoku-split-epoch1'))
    p.add_argument('--skip-train-scan',action='store_true')
    a=p.parse_args(); audit(a.root,a.data,a.output,not a.skip_train_scan)
