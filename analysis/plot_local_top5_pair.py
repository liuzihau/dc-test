"""Plot paired visibility grids and the two-level local error classification."""
import json
import os
from pathlib import Path
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/runtime/local-top5-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig,folder,name):
    fig.tight_layout(rect=(0,.07,1,.94))
    for extension in ('png','pdf'):fig.savefig(folder/(name+'.'+extension),dpi=180,facecolor='white')
    plt.close(fig)


def refresh(folder):
    s=json.loads((folder/'summary.json').read_text())
    note=f"{s['samples']} documents · {s['completed_seeds']}/{s['requested_seeds']} corruption seeds · reveal {100*s['reveal_probability']:.0f}%"
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for row,metric in enumerate(('accuracy','nll')):
        for column,label in enumerate(('MDM','A','A − MDM')):
            records=s['models'][label]['visibility'] if label in ('MDM','A') else s['paired']['visibility']
            key=metric if column<2 else metric+'_delta'
            values=np.array([r[key]['value'] if r[key]['value'] is not None else np.nan for r in records]).reshape(4,4)
            if metric=='accuracy':values*=100
            if column==2:
                limit=max(1e-9,np.nanmax(np.abs(values)))
                img=axes[row,column].imshow(values,cmap='RdBu' if metric=='accuracy' else 'RdBu_r',vmin=-limit,vmax=limit)
            else:img=axes[row,column].imshow(values,cmap='Blues' if metric=='accuracy' else 'Blues_r')
            ax=axes[row,column];ax.set_xticks(range(4),['MM','MR','RM','RR']);ax.set_yticks(range(4),['MM','MR','RM','RR'])
            ax.set_xlabel('Right (+1,+2)');ax.set_ylabel('Left (−2,−1)')
            ax.set_title(label+' · '+('accuracy (%)' if metric=='accuracy' else 'NLL'))
            for i in range(4):
                for j in range(4):
                    count=s['models']['MDM']['visibility'][4*i+j]['count']
                    ax.text(j,i,(f'{values[i,j]:.2f}' if np.isfinite(values[i,j]) else 'undefined')+f'\nn={count:,}',
                        ha='center',va='center',fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
            fig.colorbar(img,ax=ax,fraction=.046,pad=.04)
    fig.suptitle('Local visibility · main-only EMA, FP32',fontweight='bold')
    fig.text(.5,.02,note+' · M=masked, R=revealed',ha='center',fontsize=10)
    save(fig,folder,'visibility_grid')
    fig,axes=plt.subplots(1,2,figsize=(12,5.6))
    for ax,names,denom,title,labels in [
        (axes[0],('neighbor','other'),'among_wrong','All center errors',['Neighbor substitution','Other error']),
        (axes[1],('center_support','neighbor_only_support','no_local_support'),'among_neighbor',
         'Neighbor substitutions',['Center support','Neighbor-only\nsupport','No local support'])]:
        x=np.arange(len(names));width=.35
        for index,(model,color) in enumerate([('MDM','#247BB8'),('A','#D66034')]):
            records=[s['models'][model]['error_categories'][name][denom] for name in names]
            values=np.array([r['value'] if r['value'] is not None else np.nan for r in records])*100
            ci=np.array([r['ci95'] if r['ci95'] is not None else [np.nan,np.nan] for r in records])*100
            ax.bar(x+(index-.5)*width,values,width,label=model,color=color,
                yerr=np.maximum(np.stack([values-ci[:,0],ci[:,1]-values]),0),capsize=3)
        ax.set_xticks(x,labels);ax.set_ylabel('Percent of '+('wrong centers' if denom=='among_wrong' else 'neighbor substitutions'))
        ax.set_title(title,loc='left',fontweight='bold');ax.set_ylim(0,100);ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True);ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Two-level error classification',fontweight='bold')
    fig.text(.5,.02,note+' · three masked positions, distinct true IDs · 95% document-bootstrap intervals',ha='center',fontsize=9)
    save(fig,folder,'error_categories')
    table=np.asarray(s['paired']['transition_counts'])
    fig,ax=plt.subplots(figsize=(8,6.5));img=ax.imshow(table,cmap='Blues')
    labels=['Correct','Center support','Neighbor-only','No local support','Other error']
    ax.set_xticks(range(5),labels,rotation=25,ha='right');ax.set_yticks(range(5),labels)
    ax.set_xlabel('A outcome');ax.set_ylabel('MDM outcome')
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{table[i,j]:,}',ha='center',va='center',fontsize=10,
            bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
    fig.colorbar(img,ax=ax);fig.suptitle('Paired outcomes on the same eligible centers',fontweight='bold')
    fig.text(.5,.02,note,ha='center',fontsize=9);save(fig,folder,'paired_transitions')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    refresh(p.parse_args().folder)
