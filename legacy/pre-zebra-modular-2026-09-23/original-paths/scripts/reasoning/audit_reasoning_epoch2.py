#!/usr/bin/env python3
"""Saved-artifact audit of both completed epochs, with no model execution."""
import json
import math
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.reasoning.audit_sudoku_split import VARIANTS,LABELS,read_json,csv_file,paired,sha256,training_rows,valid_grid
from scripts.reasoning.audit_zebra_split import diagnose,paired_content_accuracy


def main():
    output=Path('results/generated/audits/reasoning-epoch2')
    output.mkdir(parents=True,exist_ok=True)
    summary=[];details=[];comparisons=[];curves=[];clues=[];sizes=[];grads=[];checks=[]
    for task in ('sudoku-benchmark','zebra-benchmark'):
        data=Path('.cache/reasoning')/(task+'-full-v1')
        manifest=read_json(data/'manifest.json')
        assert sha256(data/'test.jsonl')==manifest['splits']['test']['sha256']
        records=[json.loads(x) for x in (data/'test.jsonl').open()]
        solved={}
        for variant,label in zip(VARIANTS,LABELS):
            old=None
            for epoch,prefix in [(1,'full'),(2,'second')]:
                run=Path(f'outputs/reasoning/{prefix}-epoch-split-six-2x3090-mb32')/task/variant
                g=read_json(run/'generation.json');c=read_json(run/'contract.json')
                assert g['contract']==c and c['epochs']==epoch
                assert c['data_sha256']==sha256(data/'manifest.json')
                assert g['step']==epoch*math.ceil(c['epoch_examples']/128)
                assert [e['id'] for e in g['examples']]==[r['id'] for r in records]
                assert g['metrics']['seed']==2026 and g['metrics']['candidate_k']==8
                assert g['metrics']['token_selection']=='paper' and g['arguments']['batch_size']==32
                path=Path(g['checkpoint']);receipt=read_json(path.with_suffix('.pt.json'))
                assert receipt['sha256']==sha256(path) and receipt['size']==path.stat().st_size
                if old is not None:
                    assert c==dict(old['contract'],epochs=2)
                    provenance=read_json(run/'continuation_source.json')
                    assert Path(provenance['source_checkpoint']).resolve()==Path(old['checkpoint']).resolve()
                    assert provenance['source_sha256']==sha256(old['checkpoint'])
                old=g
                checks.append(dict(task=task,variant=variant,epoch=epoch,checkpoint=str(path),sha256=receipt['sha256']))
                rels=defaultdict(lambda:[0,0]);selected=[]
                for r,e in zip(records,g['examples']):
                    pred=e['predicted_answer_slots'];gold=r['answer']
                    if task=='zebra-benchmark':
                        d=diagnose(r,pred)
                        for rel,ok in d.pop('clue_outcomes'):
                            rels[rel][0]+=int(ok);rels[rel][1]+=1
                    else:
                        valid=valid_grid(pred[:81]) and all(x=='[MASK]' or x==y for x,y in zip(r['prompt'],pred[:81]))
                        d=dict(size=81,content_correct=sum(x==y for x,y in zip(pred,gold)),solved=int(valid))
                    assert bool(d['solved'])==e['scores']['valid_solution']
                    prefix_len=len(r['prompt'])+(2 if task=='sudoku-benchmark' else 1)
                    assert sorted(e['decode_order'])==list(range(prefix_len,prefix_len+len(gold)+1))
                    ordered_correct=[pred[pos-prefix_len]==gold[pos-prefix_len]
                        for pos in e['decode_order'] if pos-prefix_len<len(gold)]
                    first_error=next((i+1 for i,ok in enumerate(ordered_correct) if not ok),len(gold)+1)
                    d.update(task=task,variant=variant,epoch=epoch,id=r['id'],first_content_error=first_error,
                             first_content_correct=int(ordered_correct[0]))
                    selected.append(d);details.append(d)
                solved[variant,epoch]=np.array([e['solved'] for e in selected],dtype=bool)
                val=read_json(run/f"validation/step-{g['step']:09d}.json")
                logs=training_rows(run);tail=list(logs.values())[-50:]
                norms=np.array([float(r['grad_norm']) for r in tail])
                row=dict(task=task,variant=variant,label=label,epoch=epoch,step=g['step'],solved=int(solved[variant,epoch].sum()),
                    content_accuracy=sum(d['content_correct'] for d in selected)/sum(d['size'] for d in selected),
                    first_content_accuracy=np.mean([d['first_content_correct'] for d in selected]),
                    mean_first_error=np.mean([d['first_content_error'] for d in selected]),
                    grad_norm_median=float(np.median(norms)),grad_norm_p90=float(np.quantile(norms,.9)),
                    grad_norm_max=float(norms.max()),clip_fraction=float((norms>1).mean()),
                    last500_main=float(np.mean([float(r['train/base_loss']) for r in tail])),
                    last500_neighbor=float(np.mean([float(r['train/neighbor_loss']) for r in tail])),
                    last500_identity=float(np.mean([float(r['train/identity_loss']) for r in tail])))
                for protocol,doc in [('cold',val),('nested',val['nested_teacher_forced'])]:
                    for ratio,m in doc['ratios'].items():
                        row[f'{protocol}_nll_{ratio}']=m['content_conditional_nll']
                        row[f'{protocol}_acc_{ratio}']=m['content_masked_token_accuracy']
                summary.append(row)
                for rel,(yes,n) in rels.items():
                    clues.append(dict(variant=variant,epoch=epoch,relation=rel,correct=yes,total=n,accuracy=yes/n))
                for h in range(3,7):
                    group=[d for d in selected if d.get('houses')==h]
                    if group:
                        sizes.append(dict(variant=variant,epoch=epoch,houses=h,examples=len(group),
                            solved=sum(d['solved'] for d in group),content_accuracy=sum(d['content_correct'] for d in group)/sum(d['size'] for d in group)))
                for f in sorted((run/'validation').glob('step-*.json')):
                    v=read_json(f)
                    for protocol,doc in [('cold',v),('nested',v['nested_teacher_forced'])]:
                        for ratio,m in doc['ratios'].items():
                            curves.append(dict(task=task,variant=variant,epoch=epoch,step=int(f.stem.split('-')[-1]),
                                protocol=protocol,ratio=ratio,nll=m['content_conditional_nll'],accuracy=m['content_masked_token_accuracy']))
                for step,r in logs.items():
                    grads.append(dict(task=task,variant=variant,epoch=epoch,step=step,
                                      grad_norm=float(r['grad_norm']),loss=float(r['train/loss'])))
            comparisons.append(dict(task=task,reference=variant+'-e1',condition=variant+'-e2',
                                    **paired(solved[variant,1],solved[variant,2])))
        for a,b in [('tt','tt_ea'),('tt_ea','tt_ea_np'),('tt_ea','tt_ea_rm'),('tt_ea_np','tt_ea_rm_np')]:
            comparisons.append(dict(task=task,reference=a+'-e2',condition=b+'-e2',**paired(solved[a,2],solved[b,2])))
    for name,rows in [('summary',summary),('paired',comparisons),('per_example',details),('by_houses',sizes),
                      ('validation',curves),('clues',clues),('gradients',grads)]:
        csv_file(output/(name+'.csv'),rows)
    (output/'audit.json').write_text(json.dumps(dict(summary=summary,paired=comparisons,integrity=checks),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    for row,task in enumerate(('sudoku-benchmark','zebra-benchmark')):
        for epoch,offset in [(1,-.2),(2,.2)]:
            rs=[r for r in summary if r['task']==task and r['epoch']==epoch]
            bars=axes[row,0].bar(np.arange(6)+offset,[r['solved']/10 for r in rs],width=.38,label=f'Epoch {epoch}')
            axes[row,0].bar_label(bars,fmt='%.1f',fontsize=8)
        axes[row,0].set(xticks=range(6),xticklabels=[v.replace('+','\n+') for v in LABELS],
                        ylabel='Solved puzzles (%)',title=task+' — same 1,000 tests')
        axes[row,0].set_ylim(0,110 if row==0 else 4.5);axes[row,0].legend()
        for variant,label in zip(VARIANTS,LABELS):
            rs=[r for r in curves if r['task']==task and r['variant']==variant and r['protocol']=='cold' and r['ratio']=='0.7']
            axes[row,1].plot([r['step'] for r in rs],[r['nll'] for r in rs],label=label)
        axes[row,1].set(xlabel='Optimizer updates',ylabel='Content NLL',title='70%-mask, memory reset for each example')
        axes[row,1].axvline(14090 if row==0 else 6671,color='gray',linestyle=':')
        axes[row,1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'epoch1_vs_epoch2.png',dpi=180)
    print(json.dumps(dict(summary=summary,paired=comparisons),indent=2))


if __name__=='__main__':
    main()
