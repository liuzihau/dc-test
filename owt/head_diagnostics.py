"""Same-target head errors collected from an existing frozen forward pass."""
import numpy as np
import torch


def collect(model, logp, hidden, clean, masked, wrong, rows, chunk_size=128):
    truth = torch.as_tensor(clean, device=logp.device)
    scored = masked.copy()
    scored[:, 0] = False
    main_top_logp, main_index = logp.max(-1)
    main_prediction = main_index.cpu().numpy().astype(np.int32)
    main_logp = logp.gather(-1, truth[..., None]).squeeze(-1).cpu().numpy()
    saved = dict(main_prediction=main_prediction,
                 main_true_logp=np.where(scored, main_logp, np.nan).astype(np.float32),
                 main_top1_probability=np.where(scored, main_top_logp.exp().cpu().numpy(), np.nan).astype(np.float32),
                 scored=scored, clean=np.asarray(clean, dtype=np.int32))
    config = getattr(model, 'np_config', None)
    if not getattr(config, 'enabled', False):
        return saved
    if hidden is None:
        raise RuntimeError('NP hidden states were not captured')
    if hidden.dtype != torch.float32:
        raise RuntimeError('Head diagnostics require FP32 hidden states')
    content = np.ones_like(masked, dtype=bool)
    for token in set(model.boundary_ids) | {model.mask_index}:
        content &= clean != token
    heads = model.backbone.neighbor_heads
    directions = {}
    for offset, head in zip(heads.offsets, heads.heads):
        if offset not in (-1, 1):
            raise ValueError('This paired diagnostic requires adjacent NP heads')
        name = 'left' if offset == 1 else 'right'  # Source relative to target.
        target = slice(1, None) if offset == 1 else slice(None, -1)
        eligible = scored[:, target] & content[:, :-1] & content[:, 1:]
        batch, local_target = np.nonzero(eligible)
        target_index = local_target + (1 if offset == 1 else 0)
        source_index = target_index - offset
        predictions = np.full(clean.shape, -1, dtype=np.int32)
        true_logp = np.full(clean.shape, np.nan, dtype=np.float32)
        top1_probability = np.full(clean.shape, np.nan, dtype=np.float32)
        source_state = np.full(clean.shape, 255, dtype=np.uint8)
        for start in range(0, len(batch), chunk_size):
            b = batch[start:start + chunk_size]
            j = target_index[start:start + chunk_size]
            i = source_index[start:start + chunk_size]
            logits = head(hidden[b, i])
            if logits.dtype != torch.float32:
                raise RuntimeError('Auxiliary diagnostic entered lower precision')
            logits = logits.clone()
            logits[:, model.mask_index] = -torch.inf
            probabilities = logits.log_softmax(-1)
            values = probabilities.gather(-1, truth[b, j, None]).squeeze(-1)
            if not torch.isfinite(values).all():
                raise FloatingPointError('Nonfinite auxiliary true-token log probability')
            top_logp, top_index = probabilities.max(-1)
            predictions[b, j] = top_index.cpu().numpy()
            top1_probability[b, j] = top_logp.exp().cpu().numpy()
            true_logp[b, j] = values.cpu().numpy()
        source_state[batch, target_index] = np.where(
            masked[batch, source_index], 0,
            np.where(wrong[batch, source_index], 2, 1))
        saved[name + '_prediction'] = predictions
        saved[name + '_true_logp'] = true_logp
        saved[name + '_top1_probability'] = top1_probability
        saved[name + '_source_state'] = source_state
        directions[name] = predictions
        for row_index, row in enumerate(rows):
            for group, state in [('all', None), ('masked', 0), ('correct_revealed', 1), ('wrong_revealed', 2)]:
                choose = predictions[row_index] >= 0
                if state is not None:
                    choose &= source_state[row_index] == state
                main = main_prediction[row_index, choose]
                aux = predictions[row_index, choose]
                labels = clean[row_index, choose]
                mc, ac = main == labels, aux == labels
                prefix = f'head_{name}_{group}_'
                counts = dict(targets=int(choose.sum()),
                    both_correct=int((mc & ac).sum()),
                    rescue=int((~mc & ac).sum()),
                    main_only_correct=int((mc & ~ac).sum()),
                    both_wrong_same=int((~mc & ~ac & (main == aux)).sum()),
                    both_wrong_different=int((~mc & ~ac & (main != aux)).sum()),
                    auxiliary_true_probability_better=int((true_logp[row_index, choose] > main_logp[row_index, choose]).sum()))
                row.update({prefix + key: value for key, value in counts.items()})
                row[prefix + 'main_ce_sum'] = float(-main_logp[row_index, choose].astype(np.float64).sum())
                row[prefix + 'auxiliary_ce_sum'] = float(-true_logp[row_index, choose].astype(np.float64).sum())
    if set(directions) != {'left', 'right'}:
        raise ValueError('Both adjacent directions are required for union rescue')
    for k, row in enumerate(rows):
        choose = (directions['left'][k] >= 0) & (directions['right'][k] >= 0)
        correct = main_prediction[k] == clean[k]
        rescued = ~correct & ((directions['left'][k] == clean[k]) | (directions['right'][k] == clean[k]))
        row.update(head_union_targets=int(choose.sum()),
                   head_union_main_correct=int((correct & choose).sum()),
                   head_union_main_errors=int((~correct & choose).sum()),
                   head_union_rescue=int((rescued & choose).sum()))
    return saved


def summarize(rows):
    result = []
    cells = sorted({(r['variant'], r['mask_ratio'], r['correct_fraction']) for r in rows})
    for variant, mask_ratio, correctness in cells:
        cell = [r for r in rows if (r['variant'], r['mask_ratio'], r['correct_fraction']) == (variant, mask_ratio, correctness)]
        keys = {key for r in cell for key in r if key.startswith('head_')}
        if not keys:
            continue
        totals = {key: sum(r.get(key, 0) for r in cell) for key in keys}
        groups = {}
        for name in ['left', 'right']:
            for group in ['all', 'masked', 'correct_revealed', 'wrong_revealed']:
                prefix = f'head_{name}_{group}_'
                count = totals[prefix + 'targets']
                main_errors = totals[prefix + 'rescue'] + totals[prefix + 'both_wrong_same'] + totals[prefix + 'both_wrong_different']
                def rate(value):
                    return value / count if count else None
                groups[name + '_' + group] = dict(targets=count,
                    auxiliary_ce=rate(totals[prefix + 'auxiliary_ce_sum']),
                    main_ce_on_same_targets=rate(totals[prefix + 'main_ce_sum']),
                    main_accuracy=rate(totals[prefix + 'both_correct'] + totals[prefix + 'main_only_correct']),
                    auxiliary_accuracy=rate(totals[prefix + 'both_correct'] + totals[prefix + 'rescue']),
                    agreement=rate(totals[prefix + 'both_correct'] + totals[prefix + 'both_wrong_same']),
                    absolute_rescue=rate(totals[prefix + 'rescue']),
                    conditional_rescue=totals[prefix + 'rescue'] / main_errors if main_errors else None,
                    both_wrong=rate(totals[prefix + 'both_wrong_same'] + totals[prefix + 'both_wrong_different']),
                    auxiliary_true_probability_better=rate(totals[prefix + 'auxiliary_true_probability_better']))
        union = totals['head_union_targets']
        result.append(dict(variant=variant, mask_ratio=mask_ratio, correct_fraction=correctness,
            groups=groups, counts=totals,
            union_absolute_rescue=totals['head_union_rescue'] / union if union else None,
            union_conditional_rescue=totals['head_union_rescue'] / totals['head_union_main_errors'] if totals['head_union_main_errors'] else None,
            oracle_top1_selector_accuracy=(totals['head_union_main_correct'] + totals['head_union_rescue']) / union if union else None))
    return result
