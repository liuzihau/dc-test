#!/usr/bin/env python3
"""Read-only Zebra split-suite audit, saved predictions and fixed corruptions.

Writes diagnostic tables/figures only. No GPU execution or model updates.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.reasoning.audit_sudoku_split import (
    VARIANTS, LABELS, csv_file, paired, read_json, sha256, training_rows)


def parse_clues(prompt):
    """Separate parser/checker, not calls to the production scoring function."""
    result, chunk = [], []
    for token in prompt:
        if token != 'CLUE_END':
            chunk.append(token)
            continue
        assert chunk[1] == 'LHS' and chunk[5] == 'RHS'
        refs = [chunk[2:5]] + [chunk[i:i+3] for i in range(6, len(chunk), 3)]
        result.append((chunk[0], [(k, int(c), int(v)) for k, c, v in refs]))
        chunk = []
    assert not chunk and result
    return result


def diagnose(record, predicted):
    h, a = record['metadata']['houses'], record['metadata']['attributes']
    size = h*a
    cells = predicted[:size]
    formed = len(cells) == size and all(x in {str(i) for i in range(h)} for x in cells)
    rows = [cells[i*h:(i+1)*h] for i in range(a)]
    permutations = [len(row) == h and set(row) == {str(i) for i in range(h)} for row in rows]
    outcomes = []
    for relation, refs in parse_clues(record['prompt']):
        positions = []
        for kind, category, value in refs:
            if kind == 'n':
                assert category == 0 and 0 <= value < h
                positions.append(value)
            elif kind == 'c':
                assert 0 <= category < a and 0 <= value < h
                row = rows[category]
                positions.append(row.index(str(value)) if row.count(str(value)) == 1 else None)
            else:
                raise AssertionError(kind)
        if any(v is None for v in positions):
            satisfied = False
        else:
            x = positions[0]
            if relation == 'ends': satisfied = x == 0 or x == h-1
            elif relation == '=': satisfied = x == positions[1]
            elif relation == '!=': satisfied = x != positions[1]
            elif relation == 'immediate-left': satisfied = positions[1] - x == 1
            elif relation == 'nbr': satisfied = abs(x - positions[1]) == 1
            elif relation == 'left-of': satisfied = x < positions[1]
            elif relation == 'inbetween': satisfied = x < positions[1] < positions[2]
            else: raise AssertionError(relation)
        outcomes.append((relation, satisfied))
    constraints = formed and all(permutations) and all(v for _, v in outcomes)
    exact = cells == record['answer']
    return dict(houses=h, attributes=a, size=size, wellformed=int(formed),
                all_permutations=int(all(permutations)), valid_rows=sum(permutations),
                content_correct=sum(x == y for x, y in zip(cells, record['answer'])),
                exact=int(exact), constraints=int(constraints), solved=int(exact and constraints),
                format_success=int(len(predicted) == size+1 and predicted[-1] == '[EOS]'),
                clue_outcomes=outcomes)


def permutation_reference(data):
    """Analytical no-clue baseline for the EXACT saved validation mask sampler.

    Assume only that each attribute row is a permutation. Given its revealed
    gold values, distribute probability uniformly over remaining values.
    Accuracy is expected sampled accuracy (also expected tie-broken top1),
    not correctness of a particular deterministic argmax tie-break.
    """
    import torch
    from reasoning.data import ReasoningDataset
    from reasoning.evaluation import _generator
    dataset = ReasoningDataset(data, 'validation')
    result = []
    for ratio in (.7, .5, .3, .1):
        nll = accuracy = count = 0.
        for i, record in enumerate(dataset.records):
            row = dataset[i]
            positions = torch.where(row['target_mask'])[0]
            order = torch.randperm(len(positions), generator=_generator(2026, i, 'corruption'))
            flags = np.zeros(len(positions), dtype=bool)
            flags[order[:math.ceil(len(positions)*ratio)].numpy()] = True
            h, a = record['metadata']['houses'], record['metadata']['attributes']
            missing = flags[:h*a].reshape(a,h).sum(axis=1)
            for m in missing[missing > 0]:
                nll += m*math.log(m)
                accuracy += 1.  # m positions, each with success probability 1/m.
                count += m
        result.append(dict(ratio=ratio, content_tokens=int(count),
            content_nll=nll/count, expected_content_accuracy=accuracy/count,
            definition='Row permutation only; ignores all clue tokens; uniform remaining values'))
    return result


def paired_content_accuracy(detail, reference, condition, seed=2026):
    """Puzzle-level paired bootstrap of micro-averaged content accuracy."""
    a = [r for r in detail if r['variant'] == reference]
    b = [r for r in detail if r['variant'] == condition]
    assert [r['id'] for r in a] == [r['id'] for r in b]
    assert [r['size'] for r in a] == [r['size'] for r in b]
    delta = np.array([y['content_correct']-x['content_correct'] for x,y in zip(a,b)])
    sizes = np.array([r['size'] for r in a])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(a), size=(5000,len(a)))
    boot = 100*delta[draws].sum(axis=1)/sizes[draws].sum(axis=1)
    lo,hi = np.quantile(boot,[.025,.975])
    return dict(reference=reference, condition=condition,
        delta_pp=float(100*delta.sum()/sizes.sum()), ci95_low=float(lo),
        ci95_high=float(hi), resampling_unit='puzzle', bootstrap_samples=5000)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/reasoning/full-epoch-split-six-2x3090-mb32'))
    p.add_argument('--data',type=Path,default=Path('.cache/reasoning/zebra-benchmark-full-v1'))
    p.add_argument('--output',type=Path,default=Path('results/generated/audits/zebra-split-epoch1'))
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    manifest=read_json(args.data/'manifest.json')
    for split in manifest['splits'].values():
        for item in split.get('files', {'split':split}).values():
            assert sha256(args.data/item['filename']) == item['sha256']
    records={s:[json.loads(l) for l in (args.data/manifest['splits'][s]['filename']).open()]
             for s in ('test','validation')}
    for rs in records.values():
        for r in rs: assert diagnose(r, r['answer']+['[EOS]'])['solved']
    summaries=[]; detail=[]; groups=[]; curves=[]; traces=[]; clues=[]; successes={}; integrity=[]
    for variant,label in zip(VARIANTS,LABELS):
        run=args.root/'zebra-benchmark'/variant
        g=read_json(run/'generation.json'); contract=read_json(run/'contract.json')
        assert g['contract']==contract and g['step']==6671
        assert contract['data_sha256']==sha256(args.data/'manifest.json')
        assert contract['epochs']==1 and contract['epoch_examples']==853792
        assert [r['id'] for r in records['test']]==[e['id'] for e in g['examples']]
        assert len(g['examples'])==1000 and g['arguments']['batch_size']==32
        for key,value in dict(policy='top_prob',candidate_k=8,token_selection='paper',
            seed=2026,tokens_per_step=1,memory_condition='correct').items():
            assert g['metrics'][key]==value
        checkpoint=Path(g['checkpoint']);receipt=read_json(checkpoint.with_suffix('.pt.json'))
        assert receipt['size']==checkpoint.stat().st_size and receipt['sha256']==sha256(checkpoint)
        integrity.append(dict(variant=variant,checkpoint=str(checkpoint),sha256=receipt['sha256']))
        selected=[]; rels=defaultdict(lambda:[0,0])
        for e,r in zip(g['examples'],records['test']):
            d=diagnose(r,e['predicted_answer_slots'])
            prefix=len(r['prompt'])+1; target_length=d['size']+1
            assert sorted(e['decode_order'])==list(range(prefix,prefix+target_length))
            assert e['token_selection']=='sample' and e['remaining_masked_slots']==0
            assert [manifest['vocab'][i] for i in e['predicted_answer_ids']]==e['predicted_answer_slots']
            for key,ours in [('exact_match','exact'),('valid_solution','solved'),
                             ('constraints_satisfied','constraints'),('format_success','format_success')]:
                assert e['scores'][key]==bool(d[ours]),(variant,r['id'],key)
            assert e['scores']['strict_sequence_success']==bool(d['solved'] and d['format_success'])
            for kind,ok in d.pop('clue_outcomes'):
                rels[kind][0]+=int(ok); rels[kind][1]+=1
            wrong={i for i,(x,y) in enumerate(zip(e['predicted_answer_slots'],r['answer'])) if x!=y}
            first=next((t for t,pos in enumerate(e['decode_order'],1) if pos-prefix in wrong),target_length+1)
            d.update(variant=variant,id=r['id'],first_content_error=first,
                     decode_slots=target_length, nfe=e['nfe'])
            selected.append(d);detail.append(d)
        successes[variant]=np.array([r['solved'] for r in selected],dtype=bool)
        final=read_json(run/'validation/step-000006671.json')
        lastlogs=training_rows(run);recent=[r for s,r in lastlogs.items() if s>6171]
        summary=dict(variant=variant,label=label,parameters=read_json(run/'launch.json')['parameters'],
            solved=sum(r['solved'] for r in selected),examples=1000,
            content_accuracy=sum(r['content_correct'] for r in selected)/sum(r['size'] for r in selected),
            all_permutations=sum(r['all_permutations'] for r in selected),
            wellformed=sum(r['wellformed'] for r in selected),
            format_success=sum(r['format_success'] for r in selected),
            valid_alternative_nonreference=sum(r['constraints'] and not r['exact'] for r in selected),
            mean_first_error_failed=statistics.mean(r['first_content_error'] for r in selected if not r['solved']),
            last500_clip_fraction=statistics.mean(float(r['grad_norm'])>1 for r in recent),
            approx_train_seconds_logged=sum(float(r['seconds_per_update'])*10 for r in lastlogs.values()))
        for ratio,m in final['ratios'].items():
            summary['val_nll_'+ratio]=m['content_conditional_nll']
            summary['val_acc_'+ratio]=m['content_masked_token_accuracy']
        for k in ('train/loss_full','train/loss_t0','train/loss_t1','train/loss_t2','train/loss_t3',
                  'train/base_loss','train/neighbor_loss','train/identity_loss','train/accuracy_full',
                  'train/accuracy_t3','grad_norm','seconds_per_update'):
            vals=[float(r[k]) for r in recent if r.get(k)]
            if vals:summary['last500_'+k]=statistics.mean(vals)
        summaries.append(summary)
        for h in range(3,7):
            for a in range(3,7):
                es=[r for r in selected if (r['houses'],r['attributes'])==(h,a)]
                groups.append(dict(variant=variant,houses=h,attributes=a,examples=len(es),
                    solved=sum(r['solved'] for r in es),
                    content_accuracy=sum(r['content_correct'] for r in es)/(len(es)*h*a) if es else None,
                    all_permutations=sum(r['all_permutations'] for r in es)))
        for kind,(yes,n) in rels.items():
            clues.append(dict(variant=variant,relation=kind,correct=yes,total=n,accuracy=yes/n,
                definition='All clues; missing or nonunique referenced value counts wrong'))
        for step in range(1,38):
            es=[r for r in selected if r['decode_slots']>=step]
            traces.append(dict(variant=variant,reveals=step,examples=len(es),
                error_free=sum(r['first_content_error']>step for r in es)/len(es) if es else None))
        for f in sorted((run/'validation').glob('step-*.json')):
            d=read_json(f);step=int(f.stem.split('-')[-1])
            for protocol,doc in [('cold',d),('nested',d['nested_teacher_forced'])]:
                for ratio,m in doc['ratios'].items():
                    curves.append(dict(variant=variant,step=step,protocol=protocol,ratio=ratio,
                        nll=m['content_conditional_nll'],accuracy=m['content_masked_token_accuracy']))
    comparisons=[dict(reference=a,condition=b,**paired(successes[a],successes[b])) for a,b in
        [('mdm','tt'),('tt','tt_ea'),('tt_ea','tt_ea_np'),('tt_ea','tt_ea_rm'),
         ('tt_ea_np','tt_ea_rm_np'),('tt_ea_rm','tt_ea_rm_np')]]

    # Prepared logical identities canonicalize clause order and symmetric
    # relations; unlike solution equality, they identify the input puzzle.
    from reasoning.full_data import record_key
    n=manifest['splits']['train']['records'];files=manifest['splits']['train']['files']
    ids=np.memmap(args.data/files['identities']['filename'],mode='r',dtype='V32',shape=(n,))
    held={record_key(r) for rs in records.values() for r in rs}
    overlaps=sum(bytes(x) in held for x in ids)
    layout=np.memmap(args.data/files['layout']['filename'],mode='r',dtype='<u2',shape=(n,2))
    tokens=np.memmap(args.data/files['tokens']['filename'],mode='r',dtype='u1',shape=(n,384))
    histogram=Counter();bad=0
    for start in range(0,n,16384):
        x=tokens[start:start+16384];pos=layout[start:start+16384]
        active=(np.arange(384)[None]>=pos[:,0,None]) & (np.arange(384)[None]<pos[:,1,None]-1)
        houses=np.where(active,x,0).max(axis=1)-4
        cells=pos[:,1]-pos[:,0]-1;attributes=cells//houses
        assert np.all(cells%houses==0)
        for h in range(3,7):
            for a in range(3,7):
                idx=np.where((houses==h)&(attributes==a))[0]
                histogram[h,a]+=len(idx)
                answer=x[idx[:,None],pos[idx,0,None]+np.arange(h*a)[None]]
                rows=answer.reshape(-1,a,h)
                bad+=int((~(np.sort(rows,axis=2)==np.arange(5,5+h)).all(axis=(1,2))).sum())
    assert sum(histogram.values())==n
    population=[dict(split='train',houses=h,attributes=a,examples=v) for (h,a),v in histogram.items()]
    for s,rs in records.items():
        hist=Counter((r['metadata']['houses'],r['metadata']['attributes']) for r in rs)
        population.extend(dict(split=s,houses=h,attributes=a,examples=v) for (h,a),v in hist.items())
    baseline=permutation_reference(args.data)
    token_comparison=paired_content_accuracy(detail,'tt_ea_np','tt_ea_rm_np')
    test_houses=np.array([r['metadata']['houses'] for r in records['test']])
    test_attributes=np.array([r['metadata']['attributes'] for r in records['test']])
    generation_reference=dict(
        expected_content_accuracy=float(test_attributes.sum()/(test_houses*test_attributes).sum()),
        expected_exact_matches=sum(1/math.factorial(int(h))**int(a)
            for h,a in zip(test_houses,test_attributes)),
        examples=len(test_houses),
        definition='Independent uniform attribute-row permutations, no clues; exact reference equality')
    report=dict(summary=summaries,paired=comparisons,integrity=integrity,
        permutation_only_validation=baseline,
        paired_content_accuracy=token_comparison,
        permutation_only_generation=generation_reference,
        data_audit=dict(train_rows=n,heldout_logical_identity_overlaps=overlaps,
            invalid_training_attribute_permutations=bad,source_counters=manifest['source']['counters']),
        caveats=['Single training/decoding seed; paired intervals conditional on checkpoints.',
                 'No model inference or training changed. Exact primary scorer independently checked.',
                 'Clue diagnostics are not solve rates; size-stratified tables are essential.',
                 'Zero paired bootstrap width when both models solve nothing is a floor effect, not equivalence.'])
    for name,rows in [('summary',summaries),('paired',comparisons),('per_example',detail),
        ('by_size',groups),('validation',curves),('survival',traces),('clues',clues),
        ('population',population),('permutation_baseline',baseline),
        ('paired_content_accuracy',[token_comparison]),('permutation_generation',[generation_reference])]:
        csv_file(args.output/(name+'.csv'),rows)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=plt.cm.tab10.colors[:6]
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    bars=axes[0].bar(range(6),[r['solved']/10 for r in summaries],color=colors)
    axes[0].bar_label(bars,labels=[str(r['solved'])+'/1000' for r in summaries],fontsize=8)
    axes[0].set(xticks=range(6),xticklabels=[s.replace('+','\n+') for s in LABELS],
        ylim=(0,4),ylabel='Solved puzzles (%)',title='Zebra: one deduplicated epoch')
    for variant,label,color in zip(VARIANTS,LABELS,colors):
        vs=[r for r in curves if r['variant']==variant and r['protocol']=='cold' and r['ratio']=='0.7']
        axes[1].plot([r['step'] for r in vs],[r['nll'] for r in vs],label=label,color=color)
        vals=[]
        for h in range(3,7):
            rows=[r for r in groups if r['variant']==variant and r['houses']==h]
            vals.append(100*sum(r['solved'] for r in rows)/sum(r['examples'] for r in rows))
        axes[2].plot(range(3,7),vals,marker='o',label=label,color=color)
    axes[1].axhline(baseline[0]['content_nll'],color='black',ls=':',label='Permutation-only reference')
    axes[1].set(xlabel='Optimizer updates',ylabel='Content conditional NLL',title='Cold 70%-mask validation')
    axes[1].legend(fontsize=7)
    axes[2].set(xlabel='Number of houses',ylabel='Solved puzzles (%)',xticks=range(3,7),title='Where the successes occur')
    fig.tight_layout();fig.savefig(args.output/'zebra_audit.png',dpi=180);plt.close(fig)
    # Separate task panels: paper numbers are context, never a matched baseline.
    sudoku=read_json(Path('results/generated/audits/sudoku-split-epoch1/audit.json'))['summary']
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,name,values,paper in [(axes[0],'Sudoku-Puzzle',[r['solved']/10 for r in sudoku],80.2),
                                (axes[1],'Zebra',[r['solved']/10 for r in summaries],96.9)]:
        bars=ax.bar(range(6),values,color=colors);ax.bar_label(bars,fmt='%.1f%%',fontsize=8)
        ax.axhline(paper,color='black',ls='--',label=f'Paper MDM: {paper}% (unmatched reference)')
        ax.set(xticks=range(6),xticklabels=[s.replace('+','\n+') for s in LABELS],
               ylim=(0,110),ylabel='Solved puzzles (%)',title=name+' — our epoch 1')
        ax.legend(fontsize=8)
    fig.suptitle('Cross-paper context only: training budgets, retained data and exact architecture differ')
    fig.tight_layout();fig.savefig(args.output/'two_tasks_paper_context.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(output=str(args.output),solved={r['variant']:r['solved'] for r in summaries},
        paired_content_accuracy=token_comparison,permutation_only_generation=generation_reference,
        data_audit=report['data_audit']),indent=2))


if __name__=='__main__':
    main()
