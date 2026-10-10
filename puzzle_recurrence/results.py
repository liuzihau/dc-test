"""Epoch-matched generation records, identical metrics to the old puzzle queues."""
import csv
import json
import os
from pathlib import Path

VARIANTS=('trajectory_attention','trajectory_recurrent')
HISTORY_COLUMNS=('variant','epoch','step','accuracy','row_accuracy','cell_accuracy','correct','n')

def atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.'+str(os.getpid())+'.partial')
    temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(path)

def record_generation(directory,epoch,step,checkpoint,expected=1280):
    directory=Path(directory);files=sorted(directory.glob('samples_*.json'))
    if not files:raise RuntimeError('Generation exited without a samples file: '+str(directory))
    payload=json.loads(files[-1].read_text());metrics=payload['eval_metrics']
    if int(metrics['n_total_puzzles'])!=expected:raise ValueError('Wrong generation sample count')
    if not 0<=float(metrics['puzzle_accuracy'])<=1:raise ValueError('Invalid puzzle accuracy')
    if abs(float(metrics['puzzle_accuracy'])-metrics['n_correct_puzzles']/expected)>1e-10:
        raise ValueError('Puzzle accuracy and correct count differ')
    atomic_json(directory/'complete.json',dict(epoch=epoch,step=step,checkpoint=str(Path(checkpoint).resolve()),
        samples=str(files[-1].resolve()),metrics=metrics))
    return metrics

def history(root):
    root=Path(root);rows=[]
    for variant in VARIANTS:
        for file in sorted((root/variant/'generation').glob('epoch-*/complete.json')):
            record=json.loads(file.read_text())
            metrics=record.get('metrics')
            if metrics is None:metrics=json.loads(Path(record['samples']).read_text())['eval_metrics']
            rows.append(dict(variant=variant,epoch=int(record['epoch']),step=int(record['step']),
                accuracy=float(metrics['puzzle_accuracy']),row_accuracy=metrics.get('mean_row_accuracy'),
                cell_accuracy=metrics.get('mean_cell_accuracy'),correct=int(metrics['n_correct_puzzles']),n=int(metrics['n_total_puzzles'])))
    rows.sort(key=lambda r:(r['variant'],r['epoch']))
    root.mkdir(parents=True,exist_ok=True)
    temporary=root/('generation_history.'+str(os.getpid())+'.csv.partial')
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=HISTORY_COLUMNS);writer.writeheader();writer.writerows(rows)
    temporary.replace(root/'generation_history.csv')
    return rows
