"""Paired document-bootstrap results from independently fitted linear probes."""
import json
import os
from pathlib import Path
import numpy as np
from analysis.local_denoising_metrics import bootstrap_ratio
from analysis.clean_neighbor_probe import OFFSETS
from owt.research import atomic_write


def macro_record(numerator,denominator,draws):
    ratios=numerator.sum(0)/denominator.sum(0)
    bootstrap=numerator[draws].sum(1)/denominator[draws].sum(1)
    return dict(value=float(ratios.mean()),ci95=np.quantile(bootstrap.mean(1),[.025,.975]).tolist())


def report(root,step,p):
    stats={};seed_metrics={}
    for model in ('MDM','A'):
        arrays=[];seed_metrics[model]=[]
        for seed in p['probe_seeds']:
            path=root/f'step{step}'/model/'fits'/f'seed-{seed}'
            a=np.load(path/'evaluation_stats.npy',allow_pickle=False);arrays.append(a)
            totals=a.sum((0,1))
            fit=json.loads((path/'complete.json').read_text())
            seed_metrics[model].append(dict(seed=seed,macro_ce=float(np.mean(totals[:,1]/totals[:,0])),
                macro_accuracy=float(np.mean(totals[:,2]/totals[:,0])),
                fitting_epochs=fit['epochs'],hit_epoch_budget=fit['max_epochs_reached']))
        stats[model]=np.mean(arrays,axis=0)
    if not np.array_equal(stats['MDM'][...,0],stats['A'][...,0]):raise ValueError('Evaluation pair denominators differ')
    draws=np.random.default_rng(20261007).integers(0,p['documents']['evaluation'],
        size=(p['bootstrap_draws'],p['documents']['evaluation']))
    result=dict(checkpoint_step=step,models={},paired={},seed_outcomes=seed_metrics,
        bootstrap_unit='document; retain all corruption seeds and average probe seeds',
        offset_order=list(OFFSETS),evaluation_documents=p['documents']['evaluation'],
        full_run=not p['smoke'],primary='equal-weight offset mean CE at50% reveal,7500 checkpoint')
    for model,a in stats.items():
        levels=[]
        for level,rate in enumerate(p['reveal_rates']):
            rows=[]
            for k,offset in enumerate(OFFSETS):
                d=a[:,level,k]
                rows.append(dict(offset=offset,count=int(d[:,0].sum()),
                    ce=bootstrap_ratio(d[:,1],d[:,0],draws),accuracy=bootstrap_ratio(d[:,2],d[:,0],draws),
                    repeated_id_count=int(d[:,3].sum()),
                    repeated_id_ce=bootstrap_ratio(d[:,4],d[:,3],draws),
                    different_id_ce=bootstrap_ratio(d[:,1]-d[:,4],d[:,0]-d[:,3],draws),
                    repeated_id_accuracy=bootstrap_ratio(d[:,5],d[:,3],draws),
                    different_id_accuracy=bootstrap_ratio(d[:,2]-d[:,5],d[:,0]-d[:,3],draws)))
            levels.append(dict(reveal_rate=rate,offsets=rows,
                macro_ce=macro_record(a[:,level,:,1],a[:,level,:,0],draws),
                macro_accuracy=macro_record(a[:,level,:,2],a[:,level,:,0],draws)))
        result['models'][model]=levels
    b=stats['MDM'];a=stats['A'];paired=[]
    for level,rate in enumerate(p['reveal_rates']):
        rows=[]
        for k,offset in enumerate(OFFSETS):
            rows.append(dict(offset=offset,
                ce_delta=bootstrap_ratio(a[:,level,k,1]-b[:,level,k,1],b[:,level,k,0],draws),
                accuracy_delta=bootstrap_ratio(a[:,level,k,2]-b[:,level,k,2],b[:,level,k,0],draws)))
        paired.append(dict(reveal_rate=rate,offsets=rows,
            macro_ce_delta=macro_record(a[:,level,:,1]-b[:,level,:,1],b[:,level,:,0],draws),
            macro_accuracy_delta=macro_record(a[:,level,:,2]-b[:,level,:,2],b[:,level,:,0],draws)))
    result['paired']=paired
    path=root/f'step{step}'/'summary.json'
    atomic_write(path,json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot(root/f'step{step}',result)
    return result


def plot(folder,s):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/runtime/clean-probe-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n=len(s['paired']);fig,axes=plt.subplots(2,n,figsize=(5*n,8),squeeze=False)
    for j,pair in enumerate(s['paired']):
        for i,metric in enumerate(('ce','accuracy')):
            ax=axes[i,j]
            for model,color in (('MDM','#247bb8'),('A','#d66034')):
                records=s['models'][model][j]['offsets']
                scale=100 if metric=='accuracy' else 1
                value=np.array([r[metric]['value'] for r in records])*scale
                ci=np.array([r[metric]['ci95'] for r in records])*scale
                ax.errorbar(range(4),value,yerr=np.maximum(np.stack([value-ci[:,0],ci[:,1]-value]),0),
                    label=model,color=color,marker='o',capsize=3)
            ax.set_xticks(range(4),['−2','−1','+1','+2']);ax.set_xlabel('Target offset from clean source')
            ax.set_ylabel('Cross-entropy' if metric=='ce' else 'Accuracy (%)')
            ax.set_title(f"Reveal {100*pair['reveal_rate']:.0f}%");ax.grid(alpha=.15);ax.legend(frameon=False)
    fig.suptitle(f"Clean-state probes · matched {s['checkpoint_step']:,}-step backbones",fontweight='bold')
    fig.tight_layout(rect=(0,0,1,.94))
    for extension in ('png','pdf'):fig.savefig(folder/('clean_neighbor_probes.'+extension),dpi=180,facecolor='white')
    plt.close(fig)
