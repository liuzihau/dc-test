"""Plot recorded probe fitting/development curves and paired fitting-seed variation."""
import csv
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/runtime/clean-probe-learning-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from owt.continuation import digest

RUN=ROOT/'outputs/analysis/owt-clean-neighbor-probes-20261007'


def refresh(run=RUN):
    p=json.loads((run/'protocol.json').read_text())
    curves={};sources={};outcomes={}
    for step in (5000,7500):
        curves[step]={};outcomes[step]={}
        for model in ('MDM','A'):
            series=[];eval_by_seed=[]
            for seed in p['probe_seeds']:
                folder=run/f'step{step}'/model/'fits'/f'seed-{seed}'
                file=folder/'learning_curve.csv'
                with file.open() as stream:rows=list(csv.DictReader(stream))
                a=np.array([[int(r['epoch']),float(r['training_ce']),float(r['development_ce'])] for r in rows])
                if not np.isfinite(a).all():raise ValueError('Nonfinite recorded loss')
                series.append(a);sources[str(file.relative_to(ROOT))]=digest(file)
                stats=np.load(folder/'evaluation_stats.npy',allow_pickle=False)
                counts=stats.sum(0)
                eval_by_seed.append(dict(seed=seed,
                    overall_ce=float(np.mean(counts[:,:,1].sum(0)/counts[:,:,0].sum(0))),
                    overall_accuracy=float(np.mean(counts[:,:,2].sum(0)/counts[:,:,0].sum(0))),
                    primary_ce=float(np.mean(counts[1,:,1]/counts[1,:,0])),
                    primary_accuracy=float(np.mean(counts[1,:,2]/counts[1,:,0]))))
            if not all(np.array_equal(series[0][:,0],s[:,0]) for s in series):raise ValueError('Epoch coverage differs')
            curves[step][model]=np.stack(series)
            outcomes[step][model]=eval_by_seed
        paired={}
        for metric in ('overall_ce','overall_accuracy','primary_ce','primary_accuracy'):
            values=np.array([a[metric]-b[metric] for a,b in zip(outcomes[step]['A'],outcomes[step]['MDM'])])
            paired[metric]=dict(mean=float(values.mean()),sample_std=float(values.std(ddof=1)),by_probe_seed=values.tolist())
        outcomes[step]['paired_seed_variation']=paired
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for column,step in enumerate((5000,7500)):
        for row,metric,name in ((0,1,'Fitting loss'),(1,2,'Validation loss (development)')):
            ax=axes[row,column]
            for model,color in (('MDM','#247bb8'),('A','#d66034')):
                data=curves[step][model];x=data[0,:,0];y=data[:,:,metric]
                mean=y.mean(0);std=y.std(0,ddof=1)
                ax.plot(x,mean,'o-',label=model,color=color,lw=2,ms=4)
                ax.fill_between(x,mean-std,mean+std,color=color,alpha=.2)
            ax.axvline(4,color='#777777',ls='--',lw=1,label='Selected epoch 4')
            ax.set_title(f'{step:,}-step backbone · {name}',loc='left',fontweight='bold')
            ax.set_xlabel('Probe fitting epoch');ax.set_ylabel('Mean four-head cross-entropy')
            ax.set_xticks(range(1,8));ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
            ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Frozen clean-source neighbor probes · MDM versus A',fontweight='bold',fontsize=16)
    fig.text(.5,.018,'Lines: mean of 3 probe seeds · Band: ±1 seed standard deviation · Fitting: online epoch mean; validation: epoch-end evaluation',
        ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    for extension in ('png','pdf'):fig.savefig(run/('probe_train_validation_grid.'+extension),dpi=200,facecolor='white')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ax,step in zip(axes,(5000,7500)):
        diff=curves[step]['A'][:,:,2]-curves[step]['MDM'][:,:,2]
        x=curves[step]['A'][0,:,0]
        for seed,line in zip(p['probe_seeds'],diff):ax.plot(x,line,lw=1,alpha=.55,label=f'Probe seed {seed}')
        ax.plot(x,diff.mean(0),color='#173f66',lw=2.5,label='Mean paired difference')
        ax.axhline(0,color='#777777',lw=1);ax.axvline(4,color='#777777',ls='--',lw=1)
        ax.set(title=f'{step:,}-step backbone',xlabel='Probe fitting epoch',ylabel='Development CE: A − MDM')
        ax.grid(alpha=.16);ax.legend(frameon=False,fontsize=9);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Paired development difference across probe seeds',fontweight='bold')
    fig.tight_layout(rect=(0,0,1,.92))
    for extension in ('png','pdf'):fig.savefig(run/('probe_development_difference.'+extension),dpi=200,facecolor='white')
    plt.close(fig)
    (run/'probe_learning_curve_review.json').write_text(json.dumps(dict(source_sha256=sources,
        source_run=str(run.relative_to(ROOT)),probe_seed_variation=outcomes,selected_epoch=4,
        bands='sample standard deviation across3 fitting seeds; not a document confidence interval',
        fitting_loss='online minibatch CE averaged during each epoch',development_loss='epoch-end CE on50 held-out development documents',
        new_model_fits=False,new_model_forwards=False),indent=2)+'\n')
    print(json.dumps({step:outcomes[step]['paired_seed_variation'] for step in outcomes},indent=2))


if __name__=='__main__':refresh()
