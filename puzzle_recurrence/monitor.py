"""Refresh live loss, generation accuracy, and three-state token-accuracy figures."""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import time
os.environ.setdefault('MPLCONFIGDIR','/tmp/puzzle-monitor-mpl')
from puzzle_recurrence.results import history,atomic_json,VARIANTS
from puzzle_recurrence.schedule import TASKS

ROOT=Path(__file__).resolve().parents[1]
COLORS={'trajectory_attention':'#247bb8','trajectory_recurrent':'#137f91'}
LABELS={'trajectory_attention':'Trajectory + attention','trajectory_recurrent':'Trajectory + attention + RM'}

def read_rows(path):
    if not path.exists():return []
    raw=path.read_bytes();raw=raw[:raw.rfind(b'\n')+1]
    rows=[]
    for r in csv.DictReader(io.StringIO(raw.decode())):
        if None in r or any(v is None for v in r.values()):continue
        rows.append(r)
    return rows

def points(rows,x,y):
    values={}
    for r in rows:
        try:a=float(r[x]);b=float(r[y])
        except (ValueError,TypeError,KeyError):continue
        if math.isfinite(a) and math.isfinite(b):values[a]=b
    return sorted(values.items())

def refresh(root,task,start_epoch=1.,smooth=128):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);root.mkdir(parents=True,exist_ok=True);generation=history(root)
    steps=TASKS[task]['steps_per_epoch'];frames={};snapshot=dict(task=task,variants={},generation=generation,
        model_forwards=0,source='Saved CSV metrics and completed generation records')
    for variant in VARIANTS:
        run=root/variant
        train=read_rows(run/'local_metrics/train.csv');valid=read_rows(run/'local_metrics/validation.csv')
        tokens=read_rows(run/'local_metrics/trajectory_validation.csv')
        frames[variant]=(train,valid,tokens)
        successful=[r for r in generation if r['variant']==variant]
        snapshot['variants'][variant]=dict(latest_training_step=int(float(train[-1]['optimizer_step'])) if train else None,
            latest_validation=valid[-1] if valid else None,latest_generation=successful[-1] if successful else None)
    def plot(ax,data,field,title,ylabel,smoothing=False,scale=1):
        for variant in VARIANTS:
            rows=data(variant);xy=points(rows,'optimizer_step',field)
            if not xy:continue
            import numpy as np
            x=np.array([p[0]/steps for p in xy]);y=np.array([p[1]*scale for p in xy])
            if smoothing:
                import pandas as pd
                y=pd.Series(y).rolling(smooth,min_periods=1).mean().to_numpy()
            keep=x>=start_epoch
            ax.plot(x[keep],y[keep],color=COLORS[variant],label=LABELS[variant],lw=2,
                marker=None if smoothing else 'o',markersize=4)
        ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Training epoch');ax.set_ylabel(ylabel)
        ax.set_xlim(left=max(0,start_epoch-.2));ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
        if ax.lines:ax.legend(fontsize=8)
    def generation_plot(ax,field,title):
        for variant in VARIANTS:
            rows=[r for r in generation if r['variant']==variant and r[field] is not None]
            if rows:ax.plot([r['epoch'] for r in rows],[100*float(r[field]) for r in rows],
                'o-',lw=2,color=COLORS[variant],label=LABELS[variant])
        ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Training epoch');ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0,100);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
        if ax.lines:ax.legend(fontsize=8)
        else:ax.text(.5,.5,'First generation evaluation: epoch 3',ha='center',transform=ax.transAxes,color='#607589')
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            temporary=root/(name+'.'+str(os.getpid())+'.partial.'+ext);fig.savefig(temporary,dpi=180,facecolor='white');temporary.replace(root/(name+'.'+ext))
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    plot(axes[0,0],lambda v:frames[v][0],'main_elbo','Main training objective','Weighted three-state objective',True)
    plot(axes[0,1],lambda v:frames[v][1],'val_nll','Main validation loss','Author single-canvas NLL')
    generation_plot(axes[1,0],'accuracy','Solved-puzzle generation accuracy')
    plot(axes[1,1],lambda v:frames[v][2],'val/trajectory_center_accuracy','Center-state token accuracy','Masked-token accuracy (%)',scale=100)
    save(fig,'performance')
    fig,ax=plt.subplots(figsize=(8,4.8));generation_plot(ax,'accuracy','Solved-puzzle generation accuracy');save(fig,'accuracy_vs_epoch')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    generation_plot(axes[0],'row_accuracy','Mean row accuracy');generation_plot(axes[1],'cell_accuracy','Mean cell accuracy');save(fig,'row_cell_accuracy')
    atomic_json(root/'snapshot.json',snapshot)
    print(task+': '+str(root/'performance.png'))
    for variant in VARIANTS:
        r=snapshot['variants'][variant];g=r['latest_generation']
        print(f'  {variant}: step={r["latest_training_step"]}; '+(f'epoch{g["epoch"]} solved={100*g["accuracy"]:.2f}%' if g else 'generation pending'))
    return snapshot

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',nargs='+',choices=list(TASKS),default=['sudoku','zebra'])
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs');p.add_argument('--watch',action='store_true')
    p.add_argument('--interval',type=int,default=60);p.add_argument('--start-epoch',type=float,default=1.)
    args=p.parse_args()
    if args.interval<5:p.error('Watch interval must be at least five seconds')
    try:
        while True:
            for task in args.tasks:refresh(args.output_root/task/'three-state-ablation',task,args.start_epoch)
            if not args.watch:return
            time.sleep(args.interval)
    except KeyboardInterrupt:pass

if __name__=='__main__':main()
