"""Clue-blind calibration for the fixed five-category Zebra pilot.

This is not a solver. It knows the public answer schema (five permutations of
the house numbers followed by EOS) and uses only revealed answer digits. It
assigns every unresolved position in a category the uniform marginal over
that category's unused digits. Prompt tokens and masked ground-truth digits
never enter prediction.

The corruption assumption is teacher-forced masking of a clean answer. An
incorrect generated digit, particularly a duplicate, is outside this
calibration's contract. The baseline must not be reported as a learned model,
whole-puzzle solving accuracy, or an unconditional diffusion likelihood.
"""

import math

import torch


def _answer_slots(input_ids, target_mask, tokenizer):
    if getattr(tokenizer, "task", None) != "zebra":
        raise ValueError("Permutation shortcut requires the Zebra tokenizer")
    if (input_ids.ndim != 2 or target_mask.shape != input_ids.shape
            or target_mask.dtype != torch.bool):
        raise ValueError("Expected integer input_ids and boolean target_mask with shape [B,L]")
    if input_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64,
                              torch.uint8):
        raise ValueError("input_ids must contain integer token IDs")
    if input_ids.device != target_mask.device:
        raise ValueError("input_ids and target_mask must use the same device")
    slots = []
    for row in target_mask:
        selected = row.nonzero(as_tuple=False).flatten()
        if selected.numel() != 26:
            raise ValueError("Fixed Zebra layout requires 25 content slots and one EOS slot")
        if not torch.equal(selected, torch.arange(
                selected[0], selected[0] + 26, device=selected.device)):
            raise ValueError("Zebra answer slots must be contiguous and category-major")
        slots.append(selected)
    return slots


def permutation_shortcut_probabilities(input_ids, target_mask, tokenizer):
    """Return probabilities and eligibility without accepting clean targets.

    ``probabilities`` has shape [B,L,V]. Only currently masked *content*
    positions have a normalized distribution; all other rows are zero.
    ``eligible`` identifies precisely those positions. EOS is excluded even
    if masked. Prompt content and prompt length are not used beyond locating
    the task-fixed answer slots supplied in ``target_mask``.
    """
    slots = _answer_slots(input_ids, target_mask, tokenizer)
    digits = torch.tensor(tokenizer.encode(list("12345")),
                          dtype=torch.long, device=input_ids.device)
    probabilities = torch.zeros((*input_ids.shape, tokenizer.vocab_size),
                                dtype=torch.float32, device=input_ids.device)
    eligible = torch.zeros_like(target_mask)
    for row, positions in enumerate(slots):
        eos = input_ids[row, positions[-1]]
        if int(eos) not in (tokenizer.mask_id, tokenizer.eos_id):
            raise ValueError("Teacher-forced Zebra EOS slot must contain MASK or EOS")
        for category in range(5):
            category_slots = positions[5 * category:5 * (category + 1)]
            values = input_ids[row, category_slots]
            masked = values.eq(tokenizer.mask_id)
            observed = values[~masked]
            if observed.numel() and not bool(observed[:, None].eq(digits).any(-1).all()):
                raise ValueError("Revealed Zebra content must contain house digits only")
            if observed.unique().numel() != observed.numel():
                raise ValueError("Repeated revealed digits violate clean teacher-forced corruption")
            unused = digits[~digits[:, None].eq(observed[None, :]).any(-1)]
            unresolved = category_slots[masked]
            if unresolved.numel():
                # Every compatible permutation gives these same marginals.
                probabilities[row, unresolved[:, None], unused[None, :]] = 1.0 / unused.numel()
                eligible[row, unresolved] = True
    return {"probabilities": probabilities, "eligible": eligible}


def score_permutation_shortcut(input_ids, clean_ids, target_mask, tokenizer):
    """Score exact marginals on gold, using gold only after prediction.

    Totals can be pooled over batches. Top-1 uses the lowest vocabulary ID to
    break ties; ``expected_*`` accuracy instead averages uniform tie-breaking.
    Report content-only NLL/accuracy, not the zero-cost schema-known EOS.
    Empty masked-content sets have zero totals and ``None`` means.
    """
    prediction = permutation_shortcut_probabilities(input_ids, target_mask, tokenizer)
    if clean_ids.shape != input_ids.shape or clean_ids.device != input_ids.device:
        raise ValueError("clean_ids must match input_ids shape and device")
    if clean_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64,
                              torch.uint8):
        raise ValueError("clean_ids must contain integer token IDs")
    slots = _answer_slots(input_ids, target_mask, tokenizer)
    digits = torch.tensor(tokenizer.encode(list("12345")),
                          dtype=torch.long, device=input_ids.device).sort().values
    for row, positions in enumerate(slots):
        gold = clean_ids[row, positions]
        if int(gold[-1]) != tokenizer.eos_id:
            raise ValueError("Clean Zebra answer must end with EOS")
        for category in range(5):
            if not torch.equal(gold[category * 5:(category + 1) * 5].sort().values, digits):
                raise ValueError("Clean Zebra answer categories must be permutations of 1..5")
        revealed = input_ids[row, positions].ne(tokenizer.mask_id)
        if not torch.equal(input_ids[row, positions][revealed], gold[revealed]):
            raise ValueError("Revealed answer tokens must match clean teacher-forced targets")
    eligible = prediction["eligible"]
    distributions = prediction["probabilities"][eligible]
    targets = clean_ids[eligible].long()
    count = int(targets.numel())
    if count:
        assigned = distributions.gather(1, targets[:, None]).squeeze(1).double()
        nll_sum = float(-assigned.log().sum())
        correct = int(distributions.argmax(-1).eq(targets).sum())
        expected_correct = float(assigned.sum())
    else:
        nll_sum, correct, expected_correct = 0.0, 0, 0.0
    if not math.isfinite(nll_sum):
        raise ValueError("Gold answer is inconsistent with the revealed permutation")
    return {
        "baseline": "clue_blind_uniform_unused_house_digits",
        "assumption": "clean_teacher_forced_masking_fixed_5x5_category_major",
        "content_tokens": count,
        "content_nll_sum": nll_sum,
        "content_correct": correct,
        "expected_content_correct": expected_correct,
        "content_conditional_nll": nll_sum / count if count else None,
        "content_masked_token_accuracy": correct / count if count else None,
        "expected_content_masked_token_accuracy": expected_correct / count if count else None,
    }
