#!/usr/bin/env python3
"""Read-only epoch-three Zebra audit; optional bounded CPU-only clue ablation.

Writes a separate diagnostic report. Never changes training/evaluation settings,
official generation files, or the running queue. CPU probes use the first 128
validation examples, fixed masks, FP32, and no cross-step memory. Hiding clue
keys is an OOD sensitivity diagnostic, not a new benchmark or causal proof.
"""
import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.reasoning.audit_sudoku_split import (
    csv_file, paired, read_json, sha256, training_rows)
from scripts.reasoning.audit_zebra_split import diagnose, parse_clues


def direct_slots(record):
    """Grid cells fixed by explicit '= attribute value, house number' clues."""
    h = record['metadata']['houses']
    result = set()
    for relation, refs in parse_clues(record['prompt']):
        if relation != '=' or len(refs) != 2:
            continue
        for left, right in (refs, refs[::-1]):
            kind, category, value = left
            if kind == 'c' and right[0] == 'n':
                slot = category*h + right[2]
                assert record['answer'][slot] == str(value)
                result.add(slot)
    return result


def saved_audit(output):
    data = Path('.cache/reasoning/zebra-benchmark-full-v1')
    manifest = read_json(data/'manifest.json')
    assert sha256(data/'test.jsonl') == manifest['splits']['test']['sha256']
    records = [json.loads(line) for line in (data/'test.jsonl').open()]
    cross_equality_chance = []
    for record in records:
        for relation, refs in parse_clues(record['prompt']):
            if relation == '=' and all(ref[0] == 'c' for ref in refs):
                cross_equality_chance.append(
                    float(refs[0][2] == refs[1][2]) if refs[0][1] == refs[1][1]
                    else 1/record['metadata']['houses'])
    (output/'clue_reference.json').write_text(json.dumps(dict(
        relation='cross-attribute equality', count=len(cross_equality_chance),
        expected_satisfaction=float(np.mean(cross_equality_chance)),
        reference='Independently uniform random valid permutation in each attribute row; ignores clues',
        caveat='An analytical reference, not a trained model or a whole-puzzle accuracy.'), indent=2)+'\n')
    summaries, details, sizes, clues, curves, training, checks = [], [], [], [], [], [], []
    solved, sources, previous = {}, {}, None
    for epoch, prefix in ((1, 'full'), (2, 'second'), (3, 'third')):
        run = Path(f'outputs/reasoning/{prefix}-epoch-split-six-2x3090-mb32/zebra-benchmark/tt_ea_rm_np')
        g = read_json(run/'generation.json')
        contract = read_json(run/'contract.json')
        assert contract == g['contract'] and contract['epochs'] == epoch
        assert contract['data_sha256'] == sha256(data/'manifest.json')
        assert g['step'] == epoch*math.ceil(contract['epoch_examples']/128)
        assert [r['id'] for r in records] == [e['id'] for e in g['examples']]
        for k, v in dict(seed=2026, candidate_k=8, policy='top_prob',
                         token_selection='paper', tokens_per_step=1, memory_condition='correct').items():
            assert g['metrics'][k] == v
        checkpoint = Path(g['checkpoint'])
        receipt = read_json(checkpoint.with_suffix('.pt.json'))
        assert receipt['size'] == checkpoint.stat().st_size
        assert receipt['sha256'] == sha256(checkpoint)
        if previous is not None:
            assert contract == dict(previous['contract'], epochs=epoch)
            provenance = read_json(run/'continuation_source.json')
            assert Path(provenance['source_checkpoint']).resolve() == Path(previous['checkpoint']).resolve()
            assert provenance['source_sha256'] == sha256(Path(previous['checkpoint']))
            assert provenance['source_examples'] == (epoch-1)*contract['epoch_examples']
        previous = g
        sources[epoch] = g
        checks.append(dict(epoch=epoch, step=g['step'], checkpoint=str(checkpoint),
                           sha256=receipt['sha256'], contract_and_resume_verified=True))
        selected = []
        relations = defaultdict(lambda: [0, 0])
        for record, example in zip(records, g['examples']):
            pred = example['predicted_answer_slots']
            d = diagnose(record, pred)
            assert bool(d['solved']) == example['scores']['valid_solution']
            assert bool(d['constraints']) == example['scores']['constraints_satisfied']
            assert bool(d['exact']) == example['scores']['exact_match']
            for (relation, ok), (parsed, refs) in zip(d.pop('clue_outcomes'), parse_clues(record['prompt'])):
                assert relation == parsed
                relations[relation][0] += int(ok)
                relations[relation][1] += 1
                if relation == '=':
                    subtype = '=absolute' if any(ref[0] == 'n' for ref in refs) else '=cross_attribute'
                    relations[subtype][0] += int(ok)
                    relations[subtype][1] += 1
            prefix_len = len(record['prompt'])+1
            assert sorted(example['decode_order']) == list(range(prefix_len, prefix_len+d['size']+1))
            correctness = [pred[p-prefix_len] == record['answer'][p-prefix_len]
                           for p in example['decode_order'] if p-prefix_len < d['size']]
            explicit = direct_slots(record)
            d.update(epoch=epoch, id=record['id'], first_content_correct=int(correctness[0]),
                     first_content_error=next((i+1 for i, ok in enumerate(correctness) if not ok), d['size']+1),
                     explicit_slots=len(explicit), explicit_correct=sum(pred[p] == record['answer'][p] for p in explicit))
            selected.append(d)
            details.append(d)
        solved[epoch] = np.array([d['solved'] for d in selected], dtype=bool)
        row = dict(epoch=epoch, step=g['step'], examples=len(selected), solved=int(solved[epoch].sum()),
                   content_accuracy=sum(d['content_correct'] for d in selected)/sum(d['size'] for d in selected),
                   first_content_accuracy=float(np.mean([d['first_content_correct'] for d in selected])),
                   mean_first_error=float(np.mean([d['first_content_error'] for d in selected])),
                   all_permutations=sum(d['all_permutations'] for d in selected),
                   format_success=sum(d['format_success'] for d in selected),
                   alternative_valid_grids=sum(d['constraints'] and not d['exact'] for d in selected),
                   explicit_accuracy=sum(d['explicit_correct'] for d in selected)/sum(d['explicit_slots'] for d in selected))
        logs = training_rows(run)
        tail = [r for s, r in logs.items() if s > g['step']-500]
        for key in ('lr', 'grad_norm', 'train/base_loss', 'train/neighbor_loss', 'train/identity_loss',
                    'train/content_accuracy_full', 'train/content_nll_full',
                    'train/mask_ratio_t0', 'train/mask_ratio_t1', 'train/mask_ratio_t2', 'train/mask_ratio_t3'):
            vals = [float(r[key]) for r in tail if r.get(key)]
            if vals:
                row['last500_'+key] = float(np.mean(vals))
                if key == 'grad_norm':
                    row['last500_grad_median'] = float(np.median(vals))
                    row['last500_grad_max'] = max(vals)
        final = read_json(run/f"validation/step-{g['step']:09d}.json")
        for label, val in (('cold', final), ('nested', final['nested_teacher_forced'])):
            for ratio, m in val['ratios'].items():
                row[f'{label}_acc_{ratio}'] = m['content_masked_token_accuracy']
                row[f'{label}_nll_{ratio}'] = m['content_conditional_nll']
        summaries.append(row)
        for h in range(3, 7):
            group = [r for r in selected if r['houses'] == h]
            sizes.append(dict(epoch=epoch, houses=h, examples=len(group), solved=sum(r['solved'] for r in group),
                              content_accuracy=sum(r['content_correct'] for r in group)/sum(r['size'] for r in group)))
        for relation, (correct, total) in relations.items():
            clues.append(dict(epoch=epoch, relation=relation, correct=correct, total=total, accuracy=correct/total))
        for file in sorted((run/'validation').glob('step-*.json')):
            val = read_json(file)
            for label, doc in (('cold', val), ('nested', val['nested_teacher_forced'])):
                for ratio, m in doc['ratios'].items():
                    curves.append(dict(epoch=epoch, step=int(file.stem.split('-')[-1]), protocol=label, ratio=ratio,
                                       nll=m['content_conditional_nll'], accuracy=m['content_masked_token_accuracy']))
        for step, r in logs.items():
            training.append(dict(epoch=epoch, step=step, **{k:v for k,v in r.items() if k != 'step'}))
    comparisons = [dict(reference_epoch=a, condition_epoch=b, **paired(solved[a], solved[b]))
                   for a,b in ((1,2), (2,3), (1,3))]
    for filename, rows in (('summary',summaries), ('paired',comparisons), ('by_houses',sizes),
                           ('clues',clues), ('validation',curves), ('training',training), ('examples',details)):
        csv_file(output/(filename+'.csv'), rows)
    (output/'integrity.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(dict(summary=summaries, paired=comparisons, by_houses=sizes), indent=2), flush=True)
    return sources


def cold_cpu_probes(output, sources, n=128):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from reasoning.data import ReasoningDataset
    from reasoning.evaluation import _generator
    from reasoning.model import ReasoningModel
    from reasoning.runner import load_checkpoint
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    dataset = ReasoningDataset('.cache/reasoning/zebra-benchmark-full-v1', 'validation')
    loader = DataLoader(Subset(dataset, range(n)), batch_size=8, shuffle=False)
    results = []
    for epoch in (2, 3):
        checkpoint = load_checkpoint(sources[epoch]['checkpoint'])
        model = ReasoningModel(checkpoint['model_config']).cpu().eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        accum = defaultdict(lambda: [0., 0, 0, 0, 0])
        with torch.inference_mode():
            for batch in loader:
                clean = batch['input_ids'].long()
                valid = batch['attention_mask'].bool()
                targets = batch['target_mask'].bool()
                direct = torch.zeros_like(targets)
                for b, index in enumerate(batch['record_index'].tolist()):
                    record = dataset.records[index]
                    offset = len(record['prompt'])+1
                    for p in direct_slots(record):
                        direct[b, offset+p] = True
                for ratio in (1., .7):
                    mask = torch.zeros_like(targets)
                    for b, index in enumerate(batch['record_index'].tolist()):
                        positions = torch.where(targets[b])[0]
                        order = torch.randperm(len(positions), generator=_generator(2026,index,'corruption'))
                        mask[b,positions[order[:math.ceil(ratio*len(positions))]]] = True
                    content = mask.clone()
                    for special in dataset.tokenizer.special_ids:
                        content &= clean.ne(special)
                    state = clean.masked_fill(mask, dataset.tokenizer.mask_id)
                    for condition in ('original', 'clue_keys_hidden'):
                        attention = valid.clone()
                        if condition == 'clue_keys_hidden':
                            for b, index in enumerate(batch['record_index'].tolist()):
                                attention[b,:len(dataset.records[index]['prompt'])] = False
                        out = model(state, attention, return_memory=False)
                        logits = out['logits']
                        correct = logits.argmax(-1).eq(clean)
                        entry = accum[ratio, condition]
                        entry[0] += float(F.cross_entropy(logits[content],clean[content],reduction='sum'))
                        entry[1] += int(correct[content].sum())
                        entry[2] += int(content.sum())
                        eligible = direct & content
                        entry[3] += int(correct[eligible].sum())
                        entry[4] += int(eligible.sum())
                        del out, logits
        for (ratio, condition), (loss, correct, count, dcorrect, dcount) in accum.items():
            results.append(dict(epoch=epoch, ratio=ratio, condition=condition, examples=n,
                                content_tokens=count, content_nll=loss/count, content_accuracy=correct/count,
                                direct_clue_tokens=dcount, direct_clue_accuracy=dcorrect/dcount if dcount else None))
        csv_file(output/'cpu_clue_probes.csv', results)
        print('CPU validation probes complete:', epoch, flush=True)
        del model
        gc.collect()
    (output/'cpu_probe_protocol.json').write_text(json.dumps(dict(
        examples=n, split='first validation records', seed=2026, device='cpu', precision='float32',
        memory='cold; no previous DCache or final feedback', masks='same corruption stream as saved validation',
        caveat='Hiding all clue keys is OOD; sensitivity is not proof of correct logical computation. '
               'This is not BF16 generation and must not overwrite official scores.'), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/generated/audits/zebra-epoch3-investigation'))
    parser.add_argument('--cpu-probes', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = saved_audit(args.output)
    if args.cpu_probes:
        cold_cpu_probes(args.output, sources)


if __name__ == '__main__':
    main()
