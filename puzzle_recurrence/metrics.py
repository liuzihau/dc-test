"""Save trajectory objectives separately from normal author ELBO monitoring."""
import csv
from pathlib import Path
import torch
from lightning.pytorch.callbacks import Callback

class TrajectoryMetrics(Callback):
    columns=('optimizer_step','high_loss','center_loss','low_loss','high_accuracy','center_accuracy','low_accuracy',
        'high_mask_ratio','center_mask_ratio','low_mask_ratio','gradient_edges','main_forwards','identity_forwards')
    def __init__(self,run):self.run=Path(run);self.pending=[];self.last_step=0
    def on_train_start(self,trainer,module):
        self.last_step=trainer.global_step
        if trainer.is_global_zero:
            path=self.run/'local_metrics/trajectory.csv'
            if path.exists():
                rows=list(csv.DictReader(path.open()));rows=[r for r in rows if int(r['optimizer_step'])<=trainer.global_step]
                with path.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=self.columns);writer.writeheader();writer.writerows(rows)
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        trace=getattr(module,'_last_trajectory',None)
        if trace is None:return
        values=torch.cat((trace['losses'],trace['accuracy'],trace['realized'].mean(0),
            trace['losses'].new_tensor([trace['gradient_edges'],trace['main_forwards'],trace['identity_forwards']])))
        self.pending.append(values)
        if trainer.global_step==self.last_step:return
        mean=trainer.strategy.reduce(torch.stack(self.pending).mean(0),reduce_op='mean')
        self.pending=[];self.last_step=trainer.global_step
        if trainer.is_global_zero:
            path=self.run/'local_metrics/trajectory.csv';path.parent.mkdir(parents=True,exist_ok=True)
            exists=path.exists()
            with path.open('a',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=self.columns)
                if not exists:writer.writeheader()
                writer.writerow(dict(zip(self.columns,[trainer.global_step,*mean.cpu().tolist()])))
    def on_validation_end(self,trainer,module):
        if not trainer.is_global_zero or trainer.sanity_checking:return
        names=['val/trajectory_objective']+[f'val/trajectory_{s}_accuracy' for s in ('high','center','low')]
        if names[0] not in trainer.callback_metrics:return
        path=self.run/'local_metrics/trajectory_validation.csv';path.parent.mkdir(parents=True,exist_ok=True)
        exists=path.exists()
        with path.open('a',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['optimizer_step',*names])
            if not exists:writer.writeheader()
            writer.writerow(dict(optimizer_step=trainer.global_step,**{k:float(trainer.callback_metrics[k]) for k in names}))
