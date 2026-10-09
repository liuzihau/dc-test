"""Plot saved local-visibility and disjoint error summaries; no model inference."""
import argparse
import json
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/runtime/local-denoising-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def refresh(run):
    path=run/'summary.json'
    if not path.exists():
        progress=json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else {}
        print('Waiting for first complete corruption seed:',progress,flush=True)
        return
    data=json.loads(path.read_text())
    done=data.get('completed_corruption_seeds',data.get('corruption_seeds',0))
    requested=data.get('requested_corruption_seeds',data.get('corruption_seeds',0))
    stamp=f"{data['samples']} independent documents · corruption seeds {done}/{requested}"
    fig,axes=plt.subplots(1,2,figsize=(12,5.4))
    for ax,metric,title in zip(axes,('accuracy','nll'),('Masked-center accuracy (%)','Masked-center NLL')):
        values=np.array([r[metric]['value'] if r[metric]['value'] is not None else np.nan for r in data['visibility']]).reshape(4,4)
        if metric=='accuracy': values=values*100
        img=ax.imshow(values,cmap='Blues' if metric=='accuracy' else 'Blues_r')
        ax.set_xticks(range(4),['MM','MR','RM','RR'])
        ax.set_yticks(range(4),['MM','MR','RM','RR'])
        ax.set_xlabel('Right neighbors (+1, +2)'); ax.set_ylabel('Left neighbors (−2, −1)')
        ax.set_title(title)
        for row in range(4):
            for col in range(4):
                record=data['visibility'][4*row+col]
                text=f'{values[row,col]:.1f}' if metric=='accuracy' else f'{values[row,col]:.2f}'
                ax.text(col,row,text+f"\nn={record['count']:,}",ha='center',va='center',fontsize=9,
                        bbox=dict(facecolor='white',alpha=.75,edgecolor='none',pad=2))
        fig.colorbar(img,ax=ax,fraction=.046,pad=.04)
    fig.suptitle('BD3 OWT reference: local visibility',fontweight='bold')
    fig.text(.5,.015,stamp+' · M=masked; R=revealed',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(run/'local_visibility.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5.3))
    names=('swap','nonreciprocal','other')
    categories=[data['error_events'][name]['among_eligible'] for name in names]
    values=np.array([c['value'] or 0 for c in categories])*100
    ci=np.array([c['ci95'] if c['ci95'] is not None else [0,0] for c in categories])*100
    bars=ax.bar(['Reciprocal swap','Other neighbor\nsubstitution','Other error'],values,
        yerr=np.maximum(np.array([values-ci[:,0],ci[:,1]-values]),0),capsize=4,
        color=['#235a91','#508cbb','#a1b6c7'])
    for b,value in zip(bars,values):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.3,f'{value:.2f}%',ha='center')
    ax.set_ylabel('Percent of eligible centers (three masked, distinct true IDs)')
    ax.set_title('Disjoint local error categories',loc='left',fontweight='bold')
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.015,stamp+' · 95% document-bootstrap intervals',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(run/'local_errors.png',dpi=180);plt.close(fig)
    print('Updated local_visibility.png and local_errors.png:',stamp,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/analysis/owt-local-denoising-20261006/reference')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--interval',type=float,default=60)
    args=parser.parse_args()
    if not np.isfinite(args.interval) or args.interval<5: parser.error('Interval must be at least 5 seconds')
    try:
        while True:
            refresh(args.run)
            if not args.watch: return
            time.sleep(args.interval)
    except KeyboardInterrupt: pass


if __name__=='__main__': main()
