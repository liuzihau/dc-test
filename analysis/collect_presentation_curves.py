"""Collect recorded puzzle accuracy and main train/validation loss for presentation."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/runtime/presentation-curves-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A='mdm_np_zero_init_transformer_masked_source'
COLORS={'MDM':'#247BB8','MDM + NP':'#D66034','A: transformer NP':'#D66034'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':16,
    'axes.titleweight':'bold','axes.labelsize':12,'legend.fontsize':11,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})


def read(path,columns,sources):
    raw=path.read_bytes()
    # A running logger may be partway through appending its last row.
    raw=raw[:raw.rfind(b'\n')+1]
    frame=pd.read_csv(io.BytesIO(raw),usecols=columns)
    sources[str(path.relative_to(ROOT))]=dict(sha256=hashlib.sha256(raw).hexdigest(),
        complete_rows=len(frame),columns=columns)
    return frame


def loss_rows(path,field,sources):
    frame=read(path,['optimizer_step',field],sources)
    frame=frame.dropna().drop_duplicates('optimizer_step',keep='last').sort_values('optimizer_step')
    if not np.isfinite(frame.to_numpy()).all(): raise ValueError('Nonfinite metric: '+str(path))
    return frame


def owt_rows(original,variant,filename,field,sources):
    frame=loss_rows(original/'local_metrics'/filename,field,sources)
    continuation=ROOT/'outputs/owt/continuation-7500'/variant/'local_metrics'/filename
    if continuation.exists():
        extra=loss_rows(continuation,field,sources)
        common=frame.merge(extra,on='optimizer_step',suffixes=('_old','_new'))
        if not np.allclose(common[field+'_old'],common[field+'_new'],rtol=0,atol=0):
            raise ValueError('Continuation changed historical metrics')
        frame=pd.concat([frame,extra]).drop_duplicates('optimizer_step',keep='last').sort_values('optimizer_step')
    return frame


def style(ax,title,xlabel,ylabel):
    ax.set_title(title,loc='left',pad=13)
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
    ax.grid(axis='both',alpha=.17)
    ax.legend(frameon=False)


def accuracy(ax,task,frames):
    for label,frame in frames.items():
        ax.plot(frame.epoch,100*frame.accuracy,'o-',label=label,color=COLORS[label],lw=2.4,ms=5)
    style(ax,task+' · generation accuracy','Training epoch','Fully solved puzzles (%)')
    ax.set_ylim(0,100);ax.set_xlim(left=0)


def loss(ax,task,split,frames,steps_per_epoch=None):
    field='main_elbo' if split=='train' else 'val_nll'
    for label,frame in frames.items():
        x=frame.optimizer_step.to_numpy()
        y=frame[field].rolling(128 if steps_per_epoch else 32,min_periods=1).mean().to_numpy() if split=='train' else frame[field].to_numpy()
        if steps_per_epoch: x=x/steps_per_epoch
        if steps_per_epoch:
            keep=x>1 if split=='train' else np.ones(len(x),dtype=bool)
        else:
            keep=x>=1000
        ax.plot(x[keep],y[keep],label=label,color=COLORS[label],lw=2.2,
            marker='o' if split=='validation' else None,ms=4)
    title=task+(' · main training loss' if split=='train' else ' · main validation loss')
    style(ax,title,'Training epoch' if steps_per_epoch else 'Optimizer step',
        'Main ELBO' if split=='train' else 'NLL')
    if steps_per_epoch: ax.set_xlim(left=1 if split=='train' else 0)
    else: ax.set_xlim(left=1000,
        right=max(5100,max(f.optimizer_step.max() for f in frames.values())+100))


def save(fig,out,name,note=None):
    if note:
        fig.text(.10,.025,note,fontsize=9,color='#555555')
        fig.tight_layout(rect=(0,.065,1,1))
    else: fig.tight_layout()
    for extension in ('png','pdf'):
        fig.savefig(out/f'{name}.{extension}',dpi=220,facecolor='white')
    plt.close(fig)


def collect(out):
    out.mkdir(parents=True,exist_ok=True)
    sources={};puzzles={};endpoints={}
    for task,dirname in [('Sudoku','mdm-np-20ep'),('Zebra','mdm-np-40ep')]:
        root=ROOT/'outputs'/task.lower()/dirname
        contract=json.loads((root/'contract.json').read_text());steps=contract['steps_per_epoch']
        generation=read(root/'generation_history.csv',['variant','epoch','step','accuracy','correct','n'],sources)
        frames={};training={};validation={}
        for variant,label in [('mdm','MDM'),('mdm_np','MDM + NP')]:
            frame=generation[generation.variant==variant].sort_values('epoch').copy()
            if frame.empty or not np.allclose(frame.accuracy,frame.correct/frame.n):
                raise ValueError('Invalid recorded generation accuracy')
            frames[label]=frame
            training[label]=loss_rows(root/variant/'local_metrics/train.csv','main_elbo',sources)
            validation[label]=loss_rows(root/variant/'local_metrics/validation.csv','val_nll',sources)
        puzzles[task]=(frames,training,validation,steps)
        common=set(frames['MDM'].epoch)&set(frames['MDM + NP'].epoch)
        matched=max(common)
        endpoints[task]={label:float(frame[frame.epoch==matched].iloc[-1].accuracy) for label,frame in frames.items()}
        endpoints[task]['matched_epoch']=int(matched)
        fig,ax=plt.subplots(figsize=(8.5,5));accuracy(ax,task,frames)
        save(fig,out,task.lower()+'_accuracy','Recorded main-head generation · 1,280 evaluation puzzles')
        for split,data in [('train',training),('validation',validation)]:
            fig,ax=plt.subplots(figsize=(8.5,5));loss(ax,task,split,data,steps)
            save(fig,out,task.lower()+'_'+split+'_loss',
                'Trailing 128-update mean · main loss' if split=='train' else 'Recorded validation evaluations · no smoothing')
    owt={}
    for split,filename,field in [('train','train.csv','main_elbo'),('validation','validation.csv','val_nll')]:
        frames={}
        for group,variant,label in [('mdm-np-5k','mdm','MDM'),('transformer-np-5k',A,'A: transformer NP')]:
            frames[label]=owt_rows(ROOT/'outputs/owt'/group/variant,variant,filename,field,sources)
        owt[split]=frames
        endpoints['OWT_'+split]={label:dict(step=int(f.optimizer_step.iloc[-1]),value=float(f[field].iloc[-1])) for label,f in frames.items()}
        fig,ax=plt.subplots(figsize=(8.5,5));loss(ax,'OWT',split,frames)
        save(fig,out,'owt_'+split+'_loss','Trailing 32-update mean · main loss only' if split=='train'
            else 'EMA validation · recorded evaluations only · no smoothing')
    fig,axes=plt.subplots(2,2,figsize=(16,9))
    accuracy(axes[0,0],'Sudoku',puzzles['Sudoku'][0])
    accuracy(axes[0,1],'Zebra',puzzles['Zebra'][0])
    loss(axes[1,0],'OWT','train',owt['train'])
    loss(axes[1,1],'OWT','validation',owt['validation'])
    save(fig,out,'presentation_overview',
        'Puzzles: saved generation accuracy · OWT: MDM and A only · training loss: trailing 32-update mean')
    for split in ('train','validation'):
        fig,axes=plt.subplots(1,3,figsize=(18,5.5))
        for ax,task in zip(axes[:2],('Sudoku','Zebra')):
            _,training,validation,steps=puzzles[task]
            loss(ax,task,split,training if split=='train' else validation,steps)
        loss(axes[2],'OWT',split,owt[split])
        note=('Training: omit first epoch for Sudoku/Zebra; OWT starts at step 1,000 · trailing means: puzzles 128 updates, OWT 32'
            if split=='train' else 'Recorded validation evaluations · no smoothing · OWT: MDM and A only, from step 1,000')
        save(fig,out,'training_grid' if split=='train' else 'validation_grid',note)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for column,task in enumerate(('Sudoku','Zebra')):
        _,training,validation,steps=puzzles[task]
        loss(axes[0,column],task,'train',training,steps)
        loss(axes[1,column],task,'validation',validation,steps)
    save(fig,out,'sudoku_zebra_grid',
        'Top: training (first epoch omitted; trailing 128-update mean) · Bottom: validation (recorded evaluations)')
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    loss(axes[0],'OWT','train',owt['train'])
    loss(axes[1],'OWT','validation',owt['validation'])
    save(fig,out,'owt_grid',
        'MDM and A only · From step 1,000 · Training: trailing 32-update mean · Validation: EMA, no smoothing')
    receipt=dict(captured_at=datetime.now(ZoneInfo('Australia/Sydney')).isoformat(timespec='seconds'),
        source_metrics=sources,endpoints=endpoints,model_forwards=0,
        sudoku_zebra_accuracy='Recorded full-generation exact puzzle accuracy; no saved train/validation token-accuracy curves',
        puzzle_x_axis='optimizer_step / original steps_per_epoch (resumed Lightning epoch labels drift)',
        owt_models=['MDM','A'],owt_A='±1 transformer heads; both source and target masked; 0.25 per direction',
        training_cutoffs={'puzzles':'optimizer_step > steps_per_epoch','owt':'optimizer_step >= 1000'},
        owt_training_x_axis_start=1000,owt_validation_x_axis_start=1000,
        validation_cutoff={'puzzles':None,'owt':'optimizer_step >= 1000'},
        grids={'training_grid':'1×3 main training losses: Sudoku, Zebra, OWT',
            'validation_grid':'1×3 main validation losses: Sudoku, Zebra, OWT',
            'sudoku_zebra_grid':'2×2: columns Sudoku/Zebra, rows training/validation',
            'owt_grid':'1×2: OWT training and validation, MDM and A only'},
        smoothing={'puzzle_train':128,'owt_train':32,'validation':None},
        experimental_scope='one training seed per recipe; Zebra evaluation set overlaps training set')
    (out/'sources.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (out/'CONTENTS.txt').write_text(
        'Presentation figures (PNG for slides, PDF for vector export)\n\n'
        'sudoku_zebra_grid: puzzle losses; Sudoku/Zebra columns, training/validation rows.\n'
        'owt_grid: OWT training and validation side by side, from step 1,000.\n'
        'training_grid: Sudoku/Zebra/OWT main training loss, in one row of three panels.\n'
        'validation_grid: Sudoku/Zebra/OWT main validation loss, in one row of three panels.\n'
        'presentation_overview: Sudoku/Zebra exact generation accuracy and OWT main train/validation loss.\n'
        'sudoku_accuracy / zebra_accuracy: recorded main-head generation on 1,280 evaluation puzzles.\n'
        'sudoku_train_loss / sudoku_validation_loss: main training ELBO and validation NLL.\n'
        'zebra_train_loss / zebra_validation_loss: main training ELBO and validation NLL.\n'
        'owt_train_loss / owt_validation_loss: only MDM and A (both masked, ±1 transformer NP, 0.25 each).\n\n'
        'Puzzle train/validation token accuracy was not logged. Accuracy plots use saved generation evaluations.\n'
        'Zebra evaluation overlaps its training set; avoid calling this unseen-test accuracy.\n'
        'OWT training curves show the main objective, excluding auxiliary NP loss.\n'
        'Training smoothing: 128 updates for puzzles, 32 for OWT. Validation is unsmoothed.\n'
        'Puzzle training plots omit the first epoch. OWT training and validation plots start at step 1,000.\n'
        'sources.json records the captured input hashes, curve settings and latest matched endpoints.\n'
        'Refresh: /home/tliu0205/miniconda3/envs/dcache/bin/python analysis/collect_presentation_curves.py\n')
    print(json.dumps(dict(folder=str(out),endpoints=endpoints,figures=13,formats=['png','pdf']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/presentations/project-curves-20261006')
    collect(parser.parse_args().output.resolve())
