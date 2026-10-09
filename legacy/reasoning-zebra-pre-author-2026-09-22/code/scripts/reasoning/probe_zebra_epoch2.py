#!/usr/bin/env python3
"""Bounded inference-only epoch1/2 Zebra diagnostics; no training changes.

Repeated test seeds use the existing primary protocol. All memory/decoding
interventions use the FIRST 256 validation records, never selected successes.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.utils.data import DataLoader, Subset
from reasoning.data import ReasoningDataset
from reasoning.evaluation import evaluate_corruption, evaluate_generation
from reasoning.model import ReasoningModel
from reasoning.runner import atomic_json, digest, load_checkpoint


class DropRoute(torch.nn.Module):
    def __init__(self, model, route):
        super().__init__()
        self.inner, self.config, self.route = model, model.config, route
        self.eval()

    def forward(self, *args, **kwargs):
        if self.route == 'no_final':
            kwargs['previous_final_hidden'] = None
        elif self.route == 'no_dcache':
            kwargs['previous_step_kv'] = None
        return self.inner(*args, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('results/generated/audits/reasoning-epoch2/probes'))
    p.add_argument('--device', default='cuda', choices=('cpu','cuda'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    data = Path('.cache/reasoning/zebra-benchmark-full-v1')
    datasets = {s: ReasoningDataset(data,s) for s in ('validation','test')}
    loaders = {s: DataLoader(Subset(d,range(256 if s=='validation' else 1000)),
                            batch_size=32,shuffle=False) for s,d in datasets.items()}
    for epoch,prefix in [(1,'full'),(2,'second')]:
        for variant in ('tt_ea_rm_np','tt_ea_np','tt'):
            if epoch == 1 and variant != 'tt_ea_rm_np':
                continue
            run = Path(f'outputs/reasoning/{prefix}-epoch-split-six-2x3090-mb32/zebra-benchmark')/variant
            saved = json.loads((run/'generation.json').read_text())
            checkpoint = load_checkpoint(saved['checkpoint'])
            assert checkpoint['contract']==saved['contract'] and checkpoint['step']==saved['step']
            assert checkpoint['contract']['data_sha256']==digest(data/'manifest.json')
            model = ReasoningModel(checkpoint['model_config']).to(device)
            model.load_state_dict(checkpoint['model'],strict=True)
            model.eval()
            del checkpoint
            jobs=[]
            if variant=='tt_ea_rm_np':
                jobs += [('test','correct',s,'paper',8,'generate') for s in (2026,2027,2028)]
                if epoch==2:
                    jobs += [('validation',c,2026,'paper',8,'generate') for c in
                             ('correct','none','no_final','no_dcache','shuffle_dcache','shuffle_final','shuffle_both')]
            if epoch==2:
                if variant!='tt_ea_rm_np':
                    jobs.append(('validation','correct',2026,'paper',8,'generate'))
                jobs += [('validation','correct',2026,'argmax',8,'generate'),
                         ('validation','correct',2026,'paper',384,'generate'),
                         ('validation','correct',2026,'argmax',384,'generate'),
                         ('validation','correct',2026,'paper',8,'cold')]
            for split,condition,seed,tokens,k,protocol in jobs:
                name=f'{variant}-e{epoch}-{split}-{condition}-s{seed}-{tokens}-k{k}-{protocol}'
                output=args.output/(name+'.json')
                if output.exists():
                    print('Already exists:',name,flush=True)
                    continue
                active = DropRoute(model,condition) if condition.startswith('no_') else model
                mem = 'correct' if condition.startswith('no_') else condition
                d=datasets[split]
                with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16):
                    common=dict(tokenizer=d.tokenizer,records=d.records,seed=seed,memory_condition=mem)
                    if protocol=='generate':
                        metrics,details=evaluate_generation(active,loaders[split],device,
                            token_selection=tokens,candidate_k=k,**common)
                    else:
                        metrics,details=evaluate_corruption(active,loaders[split],device,
                            ratios=(1.,.95,.9,.7),reset_each_ratio=True,**common)
                payload=dict(checkpoint=saved['checkpoint'],step=saved['step'],epoch=epoch,
                    variant=variant,contract=saved['contract'],split=split,condition=condition,
                    protocol=protocol,seed=seed,token_selection=tokens,candidate_k=k,
                    precision='bf16',device=str(device),metrics=metrics,examples=details,
                    note='Diagnostic only; official files untouched. Validation subset is first 256 records.')
                if split=='test' and seed==2026:
                    exact=all(a['id']==b['id'] and a['predicted_answer_ids']==b['predicted_answer_ids']
                              for a,b in zip(details,saved['examples'])) and len(details)==len(saved['examples'])
                    payload['historical_predictions_identical']=exact
                    if not exact:
                        raise ValueError('Historical seed reproduction differs; inspect before interpreting probes')
                atomic_json(output,payload)
                print(name,'solved',metrics.get('valid_solution'),'ratios',metrics.get('ratios'),flush=True)
            del model,active
            gc.collect()
            if device.type=='cuda':
                torch.cuda.empty_cache()
    atomic_json(args.output/'status.json',dict(status='complete'))


if __name__=='__main__':
    main()
