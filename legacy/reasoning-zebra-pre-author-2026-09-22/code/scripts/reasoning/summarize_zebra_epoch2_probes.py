#!/usr/bin/env python3
"""Summarize fixed-checkpoint inference probes; never edit official results."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.reasoning.audit_sudoku_split import csv_file, paired


def main():
    root = Path('results/generated/audits/reasoning-epoch2')
    probes = root / 'probes'
    if json.loads((probes / 'status.json').read_text())['status'] != 'complete':
        raise ValueError('Probe suite has not completed')
    records = {}
    for split in ('test', 'validation'):
        path = Path('.cache/reasoning/zebra-benchmark-full-v1') / (split+'.jsonl')
        records[split] = {r['id']: r for r in map(json.loads, path.open())}
    rows, docs, paired_rows, cold = [], {}, [], []
    for path in sorted(probes.glob('*.json')):
        if path.name == 'status.json':
            continue
        d = json.loads(path.read_text())
        if d['protocol'] == 'cold':
            cold.extend(dict(variant=d['variant'], ratio=r, **m)
                        for r, m in d['metrics']['ratios'].items())
            continue
        docs[path.stem] = d
        correct = count = 0
        for e in d['examples']:
            gold = records[d['split']][e['id']]['answer']
            correct += sum(a == b for a,b in zip(e['predicted_answer_slots'], gold))
            count += len(gold)
        rows.append(dict(name=path.stem, variant=d['variant'], epoch=d['epoch'],
            split=d['split'], condition=d['condition'], seed=d['seed'],
            tokens=d['token_selection'], candidate_k=d['candidate_k'],
            n=len(d['examples']), solved=sum(e['scores']['valid_solution'] for e in d['examples']),
            content_accuracy=correct/count, format_success=d['metrics']['format_success']))
    vectors = {}
    for seed in (2026, 2027, 2028):
        for epoch in (1, 2):
            d = docs[f'tt_ea_rm_np-e{epoch}-test-correct-s{seed}-paper-k8-generate']
            vectors[epoch, seed] = np.array([e['scores']['valid_solution'] for e in d['examples']])
        assert [e['id'] for e in docs[f'tt_ea_rm_np-e1-test-correct-s{seed}-paper-k8-generate']['examples']] == [e['id'] for e in d['examples']]
        paired_rows.append(dict(seed=seed, **paired(vectors[1,seed], vectors[2,seed])))
    # Average seeds WITHIN each puzzle, then bootstrap puzzles. Do not count
    # repeated seeds on the same puzzle as independent test examples.
    delta = np.stack([vectors[2,s].astype(float)-vectors[1,s] for s in (2026,2027,2028)]).mean(0)
    values, counts = np.unique(delta, return_counts=True)
    samples = np.random.default_rng(2026).multinomial(len(delta), counts/counts.sum(), size=20000)
    interval = np.quantile((samples @ values)*100/len(delta), [.025,.975])
    means = {str(e): float(np.stack([vectors[e,s] for s in (2026,2027,2028)]).mean()*100) for e in (1,2)}
    stability = dict(epoch_solved_percent=means, mean_delta_pp=float(delta.mean()*100),
        puzzle_clustered_ci95_pp=interval.tolist(), num_puzzles=len(delta), num_decoding_seeds=3,
        note='Conditional on these two checkpoints and three fixed decoding seeds; not training-seed uncertainty.',
        per_seed=paired_rows)
    for name,data in [('probe_summary',rows),('probe_cold',cold),('probe_paired_seeds',paired_rows)]:
        csv_file(root/(name+'.csv'), data)
    (root/'seed_stability.json').write_text(json.dumps(stability,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(15,4.5))
    for epoch,offset in [(1,-.2),(2,.2)]:
        values = [100*vectors[epoch,s].mean() for s in (2026,2027,2028)]
        bars = axes[0].bar(np.arange(3)+offset,values,width=.38,label=f'Epoch {epoch}')
        axes[0].bar_label(bars,fmt='%.1f')
    axes[0].set(xticks=range(3),xticklabels=['2026','2027','2028'],xlabel='Decoding seed',
                ylabel='Solved puzzles (%)',title='RM+NP: same 1,000 test puzzles',ylim=(0,5))
    axes[0].legend()
    conditions = ['correct','none','no_final','no_dcache','shuffle_dcache','shuffle_final','shuffle_both']
    labels = ['Correct','None','No final','No DCache','Shuffle D','Shuffle F','Shuffle both']
    values = [next(r['solved']/r['n']*100 for r in rows if r['split']=='validation'
              and r['variant']=='tt_ea_rm_np' and r['condition']==c and r['tokens']=='paper'
              and r['candidate_k']==8) for c in conditions]
    bars = axes[1].bar(labels,values)
    axes[1].bar_label(bars,fmt='%.2f',fontsize=8)
    axes[1].tick_params(axis='x',rotation=45)
    axes[1].set(ylabel='Solved puzzles (%)',title='Epoch 2 RM+NP memory interventions\nFirst 256 validation puzzles',ylim=(0,7))
    for variant,label in [('tt','TT'),('tt_ea_np','TT+EA+NP'),('tt_ea_rm_np','TT+EA+RM+NP')]:
        rs = sorted((r for r in cold if r['variant']==variant),key=lambda r:float(r['ratio']))
        axes[2].plot([100*float(r['ratio']) for r in rs],
                     [100*r['content_masked_token_accuracy'] for r in rs],marker='o',label=label)
    axes[2].set(xlabel='Answer mask ratio (%)',ylabel='Content-token accuracy (%)',
                title='Cold teacher-forced prediction\nSame 256 validation puzzles')
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root/'zebra_diagnostic_probes.png',dpi=180)
    print(json.dumps(stability,indent=2))
    for r in rows:
        if r['split']=='validation':
            print(r)


if __name__ == '__main__':
    main()
