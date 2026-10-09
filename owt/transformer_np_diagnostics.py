"""Native transformer-head evaluation on the original frozen populations."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np
from owt.research import ROOT, atomic_write, read_json
from owt.source_pairing_diagnostics import (CONDITIONS, REFERENCES, prepare_references,
    verify_protocol as verify_reference_protocol, verify_saved_arrays, write_cell, sha256)
from owt.reveal_sweep import score_batch, digest
from analysis.owt_error_report import verify_pair, load_predictions

A = 'mdm_np_zero_init_transformer_masked_source'
B = 'mdm_np_zero_init_transformer_pair_count_control'
RUN_ROOT = ROOT / 'outputs/owt/transformer-np-5k'
LOCK = ROOT / 'outputs/owt/mdm-np-5k/.queue.lock'


def verify_protocol(protocol):
    if (protocol['variants'] != [A, B] or protocol['row_ids'] != list(range(1024))
            or protocol['conditions'] != [c[0] for c in CONDITIONS]
            or protocol['optimizer_step'] != 5000 or protocol['parameter_state'] != 'EMA'
            or protocol['physical_gpus'] != [2, 3]):
        raise ValueError('Native transformer population/settings changed')
    for name, expected in protocol['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Pinned native diagnostic dependency changed: ' + name)
    reference = read_json(ROOT / protocol['reference_protocol'])
    if sha256(ROOT / protocol['reference_protocol']) != protocol['reference_protocol_sha256']:
        raise ValueError('Saved reference protocol changed')
    verify_reference_protocol(reference)
    if protocol['bootstrap'] != reference['bootstrap'] or protocol['fixed_fusion'] != reference['fixed_fusion']:
        raise ValueError('Original bootstrap or fixed fusion changed')
    return reference


def verify_completed(variant):
    if variant not in (A, B):
        raise ValueError('Unknown native transformer arm')
    run = RUN_ROOT / variant
    if (read_json(run / 'complete.json') or {}).get('optimizer_step') != 5000:
        raise ValueError('Native arm has not completed5000')
    config = read_json(run / 'contract.json')
    if not config or config['mechanisms']['np']['weights'] != [.25, .25]:
        raise ValueError('Unexpected native transformer coefficients')
    expected = 'masked_source' if variant == A else 'matched_pair_count'
    if config['mechanisms']['np']['source_policy'] != expected:
        raise ValueError('Unexpected native transformer source policy')
    return run


def load_native_ema(variant):
    import gc
    from unittest.mock import patch
    import torch
    from omegaconf import OmegaConf
    from owt.checkpoint import checkpoint_provenance
    from owt.transformer_np_model import TransformerNPMDM
    run = verify_completed(variant)
    cls = TransformerNPMDM
    if variant == B:
        from owt.transformer_np_control import MatchedTransformerNPMDM
        from owt.transformer_np_control_schedule import verify_selection
        verify_selection(ROOT / 'outputs/research-notes/transformer_np_control_selection_20261004.json')
        cls = MatchedTransformerNPMDM
    config = OmegaConf.load(run / 'resolved_config.yaml')
    tokenizer = SimpleNamespace(vocab_size=50257, mask_token=None, all_special_ids=[50256])
    with patch('diffusion.metrics.Metrics', return_value=torch.nn.Module()):
        model = cls(config, tokenizer)
    provenance = checkpoint_provenance(run)
    payload = torch.load(provenance['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    if payload['global_step'] != 5000 or payload['transformer_np']['signature'] != model.resume_signature():
        raise ValueError('Checkpoint architecture/step changed')
    model.load_state_dict(payload['state_dict'], strict=True)
    model.ema.load_state_dict(payload['ema'])
    model.ema.copy_to(model._get_parameters())
    model.ema = None
    del payload
    gc.collect()
    model.eval()
    model.backbone.force_fp32_eval = True
    return model, provenance


class FeatureMap:
    """Index each branch's full-sequence features with the same target indices."""
    def __init__(self, features):
        import torch
        if set(features) != {-1, 1} or any(x.dtype != torch.float32 for x in features.values()):
            raise ValueError('Native branch features must be full FP32')
        self.features, self.dtype = features, torch.float32

    def __getitem__(self, index):
        return {offset: value[index] for offset, value in self.features.items()}


class OffsetProjection:
    def __init__(self, head, offset):
        self.head, self.offset = head, offset

    def __call__(self, features):
        return self.head(features[self.offset])


def native_score_batch(model, clean, canvas, masked, wrong, device):
    """Reuse established metric accounting, supplying the correct branch features."""
    import torch
    from owt.head_diagnostics import collect
    class Forward:
        backbone = model.backbone
        def __call__(self, inputs, sigma):
            self.logp, self.features = model.diagnostic_forward(inputs, sigma)
            return self.logp
    forward = Forward()
    rows = score_batch(forward, clean, canvas, masked, wrong, device)
    heads = model.backbone.neighbor_heads
    proxy = SimpleNamespace(np_config=model.np_config, boundary_ids=model.boundary_ids,
        mask_index=model.mask_index, backbone=SimpleNamespace(neighbor_heads=SimpleNamespace(
            offsets=heads.offsets, heads=[OffsetProjection(head, offset)
                for offset, head in zip(heads.offsets, heads.heads)])))
    with torch.inference_mode():
        predictions = collect(proxy, forward.logp, FeatureMap(forward.features), clean, masked, wrong, rows)
    return rows, predictions


def collect(args, protocol):
    reference = verify_protocol(protocol)
    run = verify_completed(args.variant)
    output = ROOT / protocol['outputs'][args.variant]['collection']
    if output.exists():
        raise ValueError('Refuse duplicate/partial native collection overwrite')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2,3':
        raise ValueError('Use GPUs2/3 only')
    reference_root = ROOT / reference['run_root']
    for label, _ in CONDITIONS:
        prepare_references(reference_root, reference, label)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.device_count() != 2:
        raise ValueError('Expected authorized GPU pair')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, provenance = load_native_ema(args.variant)
    model.to('cuda:0')
    def precision_hook(module, inputs, result):
        if inputs[0].dtype != torch.float32 or result.dtype != torch.float32 or torch.is_autocast_enabled('cuda'):
            raise ValueError('Native frozen block entered lower precision')
    handles = [block.register_forward_hook(precision_hook) for block in
               [*model.backbone.blocks, *[b.block for b in model.backbone.neighbor_branches]]]
    output.mkdir(parents=True)
    atomic_write(output / 'protocol.json', json.dumps(protocol, indent=2) + '\n')
    started = time.monotonic()
    observations, entries = [], []
    try:
        for label, ratio in CONDITIONS:
            clean, ids, canvas, masked, wrong, saved = prepare_references(reference_root, reference, label)
            batches, rows = [], []
            for start in range(0, len(ids), args.microbatch):
                end = min(start + args.microbatch, len(ids))
                values, predictions = native_score_batch(model, clean[start:end], canvas[start:end],
                    masked[start:end], wrong[start:end], 'cuda:0')
                batches.append(predictions)
                for i, value in zip(range(start, end), values):
                    rows.append(dict(variant=args.variant, condition=label, row_id=int(ids[i]),
                        mask_ratio=ratio, correct_fraction=1., clean_sha256=digest(clean[i]),
                        input_sha256=digest(canvas[i]), mask_sha256=digest(masked[i]), **value))
                atomic_write(output / 'progress.json', json.dumps(dict(condition=label,
                    finished_observations=len(observations)+len(rows),
                    elapsed_seconds=time.monotonic()-started)) + '\n')
            arrays = dict(row_ids=ids, **{key: np.concatenate([b[key] for b in batches]) for key in batches[0]})
            verify_saved_arrays(arrays, clean, ids, masked, wrong, {r['row_id']:r for r in rows})
            verify_pair(saved['mdm'], arrays)
            for direction in ('left', 'right'):
                if not np.array_equal(arrays[direction+'_source_state'], saved['mdm_np_zero_init'][direction+'_source_state']):
                    raise ValueError('Native diagnostic source eligibility changed')
            entries.append(write_cell(output, label, arrays, masked, rows))
            observations.extend(rows)
            print(label, '1024 native rows verified', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    with (output / 'observations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in observations for k in row}))
        writer.writeheader()
        writer.writerows(observations)
    result = dict(protocol_sha256=sha256(args.protocol), preflight=False, requested_variant=args.variant,
        evaluated_variant=args.variant, row_ids=ids.tolist(), optimizer_step=5000, parameter_state='EMA',
        precision=protocol['precision'], provenance=provenance, observations=len(observations),
        cells=entries, reference_model_forwards=0, new_sequence_evaluations=len(observations),
        native_transformer_features=True, block_fp32_assertions=True,
        elapsed_seconds=time.monotonic()-started)
    atomic_write(output / 'summary.json', json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--variant', choices=(A, B), required=True)
    parser.add_argument('--microbatch', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.microbatch <= 8:
        parser.error('Use microbatch1--8')
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collect(args, read_json(args.protocol))


if __name__ == '__main__':
    main()
