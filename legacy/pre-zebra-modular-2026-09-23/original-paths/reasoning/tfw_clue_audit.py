"""Paired held-out clue sensitivity and clue-blind permutation baseline.

This is a diagnostic, not a solver. Corrupted clues can be inconsistent and
are out of distribution; sensitivity alone is not evidence of logical reasoning.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .data import ReasoningDataset
from .runner import atomic_json, digest, load_checkpoint, move_batch
from .tfw import PaperMDM


def shuffle_clue_roles(ids, target, tokenizer, seed):
    """Shuffle reference digits within roles; preserve layout and answer input."""
    tokens = tokenizer.decode(ids.tolist())
    stop = int(target.long().argmax()) - 1  # SEP is immediately before answer.
    groups = ([], [], [])  # attribute index, attribute value, explicit house.
    for pos in range(stop - 2):
        if tokens[pos] == 'c':
            groups[0].append(pos + 1)
            groups[1].append(pos + 2)
        elif tokens[pos] == 'n':
            groups[2].append(pos + 2)
    result = ids.clone()
    generator = torch.Generator().manual_seed(seed)
    for group in groups:
        if group:
            positions = torch.tensor(group, device=ids.device)
            order = torch.randperm(len(group), generator=generator).to(ids.device)
            result[positions] = ids[positions[order]]
    return result


def permutation_marginal(inputs, masked, target, houses, attributes, tokenizer):
    """Expected accuracy/NLL for uniform unused digits per answer row.

    Uses ONLY visible answer digits and public dimensions. Gold is not an input.
    Accuracy is the randomized tie expectation, not a cherry-picked tie breaker.
    """
    slots = target.nonzero().flatten()[:houses * attributes]
    nll = accuracy = count = 0
    digit_ids = set(tokenizer.encode(list(map(str, range(houses)))))
    for row in slots.reshape(attributes, houses):
        visible = row[~masked[row]]
        used = set(inputs[visible].tolist())
        if not used <= digit_ids or len(used) != len(visible):
            raise ValueError('Visible answer is not a partial permutation')
        remaining = houses - len(used)
        n = int(masked[row].sum())
        if n:
            nll += n * math.log(remaining)
            accuracy += n / remaining
            count += n
    return nll, accuracy, count


def run(args):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    dataset = ReasoningDataset(args.data_dir, 'validation')
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint['contract']['data_sha256'] != digest(Path(args.data_dir) / 'manifest.json'):
        raise ValueError('Checkpoint/data provenance mismatch')
    model = PaperMDM(checkpoint['model_config']).to(device).eval()
    if model.target_region != 'answer' or model.padding_attention != 'masked':
        raise ValueError('Audit requires the repaired answer-only model')
    model.load_state_dict(checkpoint['model'], strict=True)
    loader = DataLoader(Subset(dataset, range(min(args.examples, len(dataset)))), batch_size=32)
    rows = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                              enabled=device.type == 'cuda'):
        for ratio in (.1, .7, 1.):
            totals = {key: 0. for key in ('tokens', 'original_nll', 'shuffled_nll',
                'original_correct', 'shuffled_correct', 'blind_nll', 'blind_correct', 'changed')}
            deltas = []
            for batch in loader:
                gold, target = batch['input_ids'], batch['target_mask']
                masks, corruptions = [], []
                for i, index in enumerate(batch['record_index'].tolist()):
                    generator = torch.Generator().manual_seed(2026 + 7919 * index)
                    masked = (torch.rand(gold.shape[1], generator=generator) < ratio) & target[i]
                    masks.append(masked)
                    corruptions.append(shuffle_clue_roles(gold[i], target[i], dataset.tokenizer, 3011 + index))
                    meta = dataset.records[index]['metadata']
                    a, b, c = permutation_marginal(gold[i].masked_fill(masked, model.mask_id),
                        masked, target[i], meta['houses'], meta['attributes'], dataset.tokenizer)
                    totals['blind_nll'] += a; totals['blind_correct'] += b; totals['tokens'] += c
                masked = torch.stack(masks).to(device)
                altered = torch.stack(corruptions).to(device)
                batch = move_batch(batch, device)
                gold = batch['input_ids']
                content = masked & gold.ne(dataset.tokenizer.eos_id)
                totals['changed'] += float((altered != gold).any(-1).sum())
                per_condition = []
                for name, source in (('original', gold), ('shuffled', altered)):
                    logits = model(source.masked_fill(masked, model.mask_id),
                        attention_mask=batch['attention_mask'])['logits'].float()
                    logits[..., model.mask_id] = -torch.inf
                    ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), gold, reduction='none')
                    sums = ce.masked_fill(~content, 0).sum(-1)
                    totals[name + '_nll'] += float(sums.sum())
                    totals[name + '_correct'] += float(((logits.argmax(-1) == gold) & content).sum())
                    per_condition.append(sums / content.sum(-1).clamp_min(1))
                valid = content.any(-1)
                deltas.extend((per_condition[1] - per_condition[0])[valid].cpu().tolist())
            n = totals['tokens']
            row = dict(mask_ratio=ratio, masked_content_tokens=int(n),
                changed_prompt_fraction=totals['changed'] / min(args.examples, len(dataset)))
            for name in ('original', 'shuffled', 'blind'):
                row[name + '_nll'] = totals[name + '_nll'] / n
                row[name + '_accuracy'] = totals[name + '_correct'] / n
            delta = torch.tensor(deltas, dtype=torch.float64)
            row.update(paired_example_mean_nll_delta=float(delta.mean()),
                       paired_example_nll_delta_se=float(delta.std() / math.sqrt(len(delta))))
            rows.append(row)
    result = dict(step=checkpoint['step'], split='validation', examples=min(args.examples, len(dataset)),
        checkpoint_sha256=digest(args.checkpoint), data_sha256=checkpoint['contract']['data_sha256'], rows=rows,
        note='Corrupted clues may be inconsistent/OOD. Paired sensitivity is not a causal reasoning proof. '
             'Blind baseline has public dimensions and revealed answer digits, but sees no clues.')
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--examples', type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())
