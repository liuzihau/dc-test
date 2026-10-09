#!/usr/bin/env python3
"""Bounded CPU-only checkpoint diagnostics; never updates model parameters.

Uses the first 64 validation examples, not test-set-selected puzzles. CPU FP32
results are diagnostic and must not be spliced into historical BF16 curves.
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
from reasoning.evaluation import evaluate_corruption
from reasoning.model import ReasoningModel
from reasoning.runner import load_checkpoint
from neighbor_prediction import neighbor_prediction_loss


def neighbor_gradients(model, loader):
    """Main/weighted-NP shared-parameter gradients on four fixed microbatches.

    Only the memory-free NP control is probed: its scalar objective can be
    decomposed without changing the adjacent-gradient surrogate. This is a
    final-checkpoint local measurement, not an explanation of training history.
    """
    assert not model.has_dcache and not model.has_final
    params = [p for name, p in model.named_parameters() if 'neighbor_heads' not in name]
    results = []
    for i, batch in enumerate(loader):
        if i >= 4:
            break
        trajectory = model.sample_trajectory(batch, torch.Generator().manual_seed(8721 + i))
        losses, auxiliaries = [], []
        for state, mask in zip(trajectory['states'], trajectory['masks']):
            out = model(state, batch['attention_mask'], return_memory=False)
            losses.append(model._masked_ce(out['logits'], batch['input_ids'], mask).mean())
            auxiliaries.append(neighbor_prediction_loss(
                model.backbone.neighbor_heads, out['final_hidden'], batch['input_ids'],
                state, batch['attention_mask'], model.mask_id,
                excluded_token_ids=model.config['special_ids'], checkpoint_chunks=False)['loss'])
        weights = model.config['weights']
        main = sum(w * loss for w, loss in zip(weights, losses)) / sum(weights)
        auxiliary = model.config['neighbor_weight'] * sum(
            w * loss for w, loss in zip(weights, auxiliaries)) / sum(weights)
        gmain = torch.autograd.grad(main, params, retain_graph=True, allow_unused=True)
        gaux = torch.autograd.grad(auxiliary, params, allow_unused=True)
        # Use shared parameters only; auxiliary-only heads cannot oppose the
        # backbone gradient and are excluded from the norm comparison.
        dot = sum(float((a.double() * b.double()).sum()) for a, b in zip(gmain, gaux)
                  if a is not None and b is not None)
        norm_main = sum(float(a.double().square().sum()) for a in gmain if a is not None)**.5
        norm_aux = sum(float(a.double().square().sum()) for a in gaux if a is not None)**.5
        results.append(dict(microbatch=i, examples=batch['input_ids'].shape[0],
            base_loss=float(main), weighted_auxiliary_loss=float(auxiliary),
            shared_main_gradient_norm=norm_main, shared_aux_gradient_norm=norm_aux,
            gradient_cosine=dot / max(norm_main * norm_aux, 1e-30)))
        del trajectory, losses, auxiliaries, out, main, auxiliary, gmain, gaux
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(
        'outputs/reasoning/full-epoch-split-six-2x3090-mb32/sudoku-benchmark'))
    p.add_argument('--data', type=Path, default=Path('.cache/reasoning/sudoku-benchmark-full-v1'))
    p.add_argument('--output', type=Path, default=Path(
        'results/generated/audits/sudoku-split-epoch1/cpu_probes.json'))
    p.add_argument('--examples', type=int, default=64)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    dataset = ReasoningDataset(args.data, 'validation')
    loader = DataLoader(Subset(dataset, range(min(args.examples, len(dataset)))),
                        batch_size=8, shuffle=False)
    result = dict(protocol='First validation examples; CPU FP32; cold, independent masks',
                  n_examples=min(args.examples, len(dataset)), evaluation_seed=2026,
                  warning='Small diagnostic subset, not new final generation results.', models={})
    for variant in ('mdm', 'tt', 'tt_ea', 'tt_ea_np', 'tt_ea_rm', 'tt_ea_rm_np'):
        saved = json.loads((args.root / variant / 'generation.json').read_text())
        checkpoint = load_checkpoint(saved['checkpoint'])
        assert checkpoint['contract'] == saved['contract'] and checkpoint['step'] == saved['step']
        model = ReasoningModel(checkpoint['model_config'])
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        del checkpoint
        # Build any backward graph before inference-mode RoPE caches are
        # created. Production training also initializes these in grad mode.
        record = {}
        if variant == 'tt_ea_np':
            record['neighbor_gradient_probe'] = neighbor_gradients(model, loader)
        metrics, _ = evaluate_corruption(model, loader, tokenizer=dataset.tokenizer,
            records=dataset.records, ratios=(1., .95, .9, .7), seed=2026, reset_each_ratio=True)
        record['cold_validation'] = metrics
        result['models'][variant] = record
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(variant, json.dumps(metrics['ratios']), flush=True)
        if 'neighbor_gradient_probe' in record:
            print('NP gradient diagnostics:', json.dumps(record['neighbor_gradient_probe']), flush=True)
        del model
        gc.collect()
    result['status'] = 'complete'
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
