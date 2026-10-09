"""Absolute accuracy bars and paired effects with the original bootstrap intervals."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/runtime/clean-probe-comparison-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from owt.continuation import digest

RUN=ROOT/'outputs/analysis/owt-clean-neighbor-probes-20261007'


def values(records,key,scale=1):
    point=np.array([r[key]['value'] for r in records])*scale
    interval=np.array([r[key]['ci95'] for r in records])*scale
    return point,interval,np.maximum(np.stack([point-interval[:,0],interval[:,1]-point]),0)


def save(fig,folder,name,note):
    fig.text(.5,.018,note,ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=(0,.07,1,.92))
    for extension in ('png','pdf'):fig.savefig(folder/(name+'.'+extension),dpi=200,facecolor='white')
    plt.close(fig)


def refresh(folder):
    source=folder/'summary.json';s=json.loads(source.read_text());step=s['checkpoint_step']
    x=np.arange(4);labels=['−2','−1','+1','+2']
    all_accuracy=[100*r['accuracy']['value'] for model in ('MDM','A')
        for level in s['models'][model] for r in level['offsets']]
    accuracy_limits=[.95*min(all_accuracy),1.05*max(all_accuracy)]
    palette=plt.get_cmap('YlGn')
    base_colors=[palette(v) for v in (.88,.62,.36)]
    gain_colors=[palette(v) for v in (.68,.42,.14)]
    fig,ax=plt.subplots(figsize=(12,5.8))
    width=.24
    for level in range(3):
        bar_x=x+(level-1)*width
        reveal=int(100*s['paired'][level]['reveal_rate'])
        baseline,_,_=values(s['models']['MDM'][level]['offsets'],'accuracy',100)
        candidate,_,_=values(s['models']['A'][level]['offsets'],'accuracy',100)
        gain,interval,error=values(s['paired'][level]['offsets'],'accuracy_delta',100)
        if not np.allclose(baseline+gain,candidate,rtol=0,atol=1e-10):
            raise ValueError('Stacked gain does not reproduce A accuracy')
        ax.bar(bar_x,baseline,width*.91,label=f'{reveal}% reveal: MDM',color=base_colors[level])
        ax.bar(bar_x,gain,width*.91,bottom=baseline,label=f'{reveal}% reveal: gain',
            color=gain_colors[level],edgecolor=base_colors[level],linewidth=.65)
        # Translate the paired difference interval to the MDM mean. This is
        # an interval for the gain, not a marginal interval for total A accuracy.
        ax.errorbar(bar_x,candidate,yerr=error,fmt='none',ecolor='#222222',capsize=3,
            elinewidth=1.1)
        for k in range(4):
            ax.text(bar_x[k],baseline[k]+interval[k,1]+.16,f'{gain[k]:+.2f}',
                ha='center',va='bottom',fontsize=8.5,color='#333333')
    ax.set_xticks(x,labels);ax.set_xlabel('Target offset from clean source')
    ax.set_ylim(*accuracy_limits);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,fontsize=9,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.02))
    ax.set_ylabel('MDM accuracy + A gain (%)')
    fig.suptitle(f'Clean-state probes · matched {step:,}-step backbones',fontsize=16,fontweight='bold')
    note=('Dark base: MDM · light cap: A gain · labels: gain (pp) · whiskers: 95% paired gain CI anchored at MDM mean\n'
          'ColorBrewer YlGn: low reveal darker, high reveal lighter · axis: 0.95 × minimum to 1.05 × maximum accuracy')
    save(fig,folder,'clean_neighbor_accuracy_bars',note)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
    for level,ax in enumerate(axes):
        records=s['paired'][level]['offsets'];point,interval,error=values(records,'accuracy_delta',100)
        ax.axhline(0,color='#777777',lw=1,ls='--')
        for k in range(4):
            uncertain=interval[k,0]<=0<=interval[k,1]
            ax.errorbar(x[k],point[k],yerr=error[:,k:k+1],fmt='o',color='#777777' if uncertain else '#173F66',
                capsize=4,markersize=7,elinewidth=1.6)
        ax.set_xticks(x,labels);ax.set_xlabel('Target offset from clean source')
        ax.set_title(f"Reveal {100*s['paired'][level]['reveal_rate']:.0f}%",fontweight='bold')
        ax.set_ylim(-.12,2.05);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Paired accuracy gain: A − MDM (percentage points)')
    fig.suptitle(f'Paired accuracy differences · {step:,}-step backbones',fontsize=16,fontweight='bold')
    save(fig,folder,'clean_neighbor_paired_accuracy_gain',
        '95% paired document-bootstrap intervals · above zero favors A · gray marks an interval that includes zero')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
    for level,ax in enumerate(axes):
        records=s['paired'][level]['offsets'];point,interval,error=values(records,'ce_delta')
        ax.axhline(0,color='#777777',lw=1,ls='--')
        for k in range(4):
            uncertain=interval[k,0]<=0<=interval[k,1]
            ax.errorbar(x[k],point[k],yerr=error[:,k:k+1],fmt='o',color='#777777' if uncertain else '#173F66',
                capsize=4,markersize=7,elinewidth=1.6)
        ax.set_xticks(x,labels);ax.set_xlabel('Target offset from clean source')
        ax.set_title(f"Reveal {100*s['paired'][level]['reveal_rate']:.0f}%",fontweight='bold')
        ax.set_ylim(-.21,.025);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Paired cross-entropy difference: A − MDM')
    fig.suptitle(f'Paired prediction-loss differences · {step:,}-step backbones',fontsize=16,fontweight='bold')
    save(fig,folder,'clean_neighbor_paired_ce_difference',
        '95% paired document-bootstrap intervals · below zero favors A · gray marks an interval that includes zero')
    (folder/'comparison_plot_receipt.json').write_text(json.dumps(dict(checkpoint_step=step,
        source=str(source.relative_to(ROOT)),source_sha256=digest(source),
        accuracy_bars='three reveal hues grouped at each offset;each bar stacks MDM accuracy and A-minus-MDM gain',
        accuracy_bar_whiskers='original paired accuracy_delta interval, translated vertically by MDM mean; not marginal A accuracy CI',
        accuracy_axis_limits_percent=accuracy_limits,
        accuracy_axis_rule='0.95*minimum and1.05*maximum observed MDM/A mean accuracy acrossall reveal levels',
        accuracy_palette='ColorBrewer YlGn;25% darkest,75% lightest;gain lighter than corresponding base',
        paired_plots='original paired differences and intervals from summary.json',
        confidence_intervals_modified=False,new_model_forwards=False,new_fitting=False,
        all_offset_results_displayed=True),indent=2)+'\n')
    print('Updated comparison plots:',folder)


if __name__=='__main__':
    for step in (5000,7500):refresh(RUN/f'step{step}')
