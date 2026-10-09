"""Observed ELBO versus updates and elapsed time; no FLOP-equivalence claim."""
import argparse
import json
import math
import os
from pathlib import Path

from owt.research import ROOT, collect, read_csv, atomic_write


def budget_profile(snapshot):
    base=snapshot['variants']['mdm']
    if not base['completion'] or base['latest_validation']['optimizer_step']!=5000:
        raise ValueError('Requires the completed 5,000-update MDM reference')
    reference=base['latest_validation']
    budget=reference['elapsed_seconds']
    comparisons={}
    for name,item in snapshot['variants'].items():
        rows=item['validation_rows']
        if any(b['elapsed_seconds']<=a['elapsed_seconds'] for a,b in zip(rows,rows[1:])):
            raise ValueError(f'{name}: validation timer reset; cannot infer cumulative elapsed time')
        candidates=[r for r in rows if r['elapsed_seconds']<=budget]
        if not candidates:
            continue
        matched=max(candidates,key=lambda r:r['elapsed_seconds'])
        delta=matched['val_nll']-reference['val_nll']
        comparisons[name]=dict(selected_validation=matched,delta_nats=delta,
            ppl_bound_difference_percent=100*math.expm1(delta),
            unused_elapsed_budget_seconds=budget-matched['elapsed_seconds'])
    return dict(observed_at=snapshot['observed_at'],reference=reference,
                comparisons=comparisons,
                validation_curves={name:v['validation_rows'] for name,v in snapshot['variants'].items()},
                limitations=['Latest observed checkpoint within the elapsed budget; no interpolation.',
                    'Elapsed time includes periodic monitoring, not equal measured training FLOPs.',
                    'One paired training seed; runs were sequential and hardware load may differ.',
                    'A partial arm must not be described as a completed 5,000-update result.'])


def plot(data, output):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/runtime/analysis/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from owt.research import LABELS
    colors=['#555555','#2166ac','#238b45','#d95f0e']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4))
    for (name,rows),color in zip(data['validation_curves'].items(),colors):
        partial=rows[-1]['optimizer_step']<5000
        label=LABELS[name]+(' (partial)' if partial else '')
        for ax,key,scale in [(axes[0],'optimizer_step',1.),(axes[1],'elapsed_seconds',3600.)]:
            ax.plot([r[key]/scale for r in rows],[r['val_nll'] for r in rows],
                    'o--' if partial else 'o-',label=label,color=color,lw=1.7,ms=4)
    axes[0].set(title='Matched optimizer-update view',xlabel='Optimizer update')
    axes[1].set(title='Observed elapsed-time view',xlabel='Elapsed training and monitoring time (hours)')
    axes[1].axvline(data['reference']['elapsed_seconds']/3600.,color='#999999',ls=':',lw=1.2)
    for ax in axes:
        ax.set_ylabel('Validation main ELBO (nats/token)')
        ax.grid(alpha=.2)
    axes[1].legend(frameon=False,fontsize=8.5)
    fig.suptitle('OWT pilot: main prediction and observed computational cost')
    fig.text(.5,.015,'One paired seed, fixed 1,024-row validation. Elapsed time is not a FLOP-matched control.',
             ha='center',fontsize=8.5,color='#555555')
    fig.tight_layout(rect=(0,.05,1,.94))
    for extension in ['pdf','png']:
        fig.savefig(output/f'budget_profile.{extension}',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if (args.output/'summary.json').exists():
        parser.error('Use a fresh output directory to preserve earlier evidence')
    snapshot=collect()
    # Validation monotonicity alone might miss a reset between sparse checkpoints.
    for name in snapshot['variants']:
        train=read_csv(Path(snapshot['root'])/name/'local_metrics/train.csv')
        if any(b['elapsed_seconds']<a['elapsed_seconds'] for a,b in zip(train,train[1:])):
            raise ValueError(f'{name}: training timer reset; cumulative cost is unknown')
    data=budget_profile(snapshot)
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_write(args.output/'summary.json',json.dumps(data,indent=2)+'\n')
    plot(data,args.output)
    print(json.dumps(data['comparisons'],indent=2))


if __name__=='__main__':
    main()
