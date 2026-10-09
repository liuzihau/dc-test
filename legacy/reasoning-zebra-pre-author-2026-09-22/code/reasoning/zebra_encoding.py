"""Opt-in public coordinate features for Zebra, never solver/gold features.

The answer table uses attribute-major values per house. Its dimensions are
recovered from the visible category references and the public answer width.
The complete prepared datasets must pass audit_public_dimensions before use.
"""
import torch
from torch import nn


def public_features(ids, attention, *, sep_id, c_id, n_id, digit_ids, answer_base=384):
    if attention is None or attention.shape != ids.shape:
        raise ValueError('Coordinate encoding requires public attention layout')
    positions = torch.arange(ids.shape[1], device=ids.device)[None, :].expand_as(ids)
    sep = ids.eq(sep_id) & attention.bool()
    if not bool(sep.any(-1).all()):
        raise ValueError('A visible prompt SEP is required')
    # The original boundary is the FIRST SEP. A wrong generated SEP in an
    # answer slot must neither change layout nor crash autoregressive reveal.
    start = sep.long().argmax(-1) + 1
    prompt = positions < start[:, None] - 1
    target = (positions >= start[:, None]) & attention.bool()
    # Plain vocabulary indices are not assumed to coincide with digit values.
    digits = torch.full_like(ids, -1)
    for value, token_id in enumerate(digit_ids):
        digits = torch.where(ids.eq(token_id), value, digits)
    c_ref, n_ref = ids.eq(c_id) & prompt, ids.eq(n_id) & prompt
    next_digit = digits.roll(-1, 1)
    next_next = digits.roll(-2, 1)
    attributes = torch.where(c_ref, next_digit + 1, 0).amax(-1)
    width = target.sum(-1) - 1  # public content slots, excluding final EOS slot
    houses = width // attributes.clamp_min(1)
    valid = ((attributes >= 3) & (attributes <= 6) & (houses >= 3) & (houses <= 6)
             & (width == attributes * houses))
    bad_refs = ((c_ref & ((next_digit < 0) | (next_digit >= attributes[:, None])
                         | (next_next < 0) | (next_next >= houses[:, None])))
                | (n_ref & ((next_digit != 0) | (next_next < 0) | (next_next >= houses[:, None]))))
    if not bool(valid.all()) or bool(bad_refs.any()):
        raise ValueError('Public clue dimensions/references are invalid or ambiguous')
    offset = positions - start[:, None]
    content = target & (offset < width[:, None])
    position_ids = torch.where(target, answer_base + offset, positions)
    position_ids = torch.where(attention.bool(), position_ids, 0)
    # Reference features are repeated across the c/n marker and its two digits.
    # This explicitly binds a clue entity; it does not solve any relation.
    attr = torch.where(c_ref, next_digit + 1, 0)
    value = torch.where(c_ref, next_next + 1, 0)
    house = torch.where(n_ref, next_next + 1, 0)
    role = c_ref.long() + 2 * n_ref.long()
    attr, value, house, role = [base + base.roll(1, 1) + base.roll(2, 1)
                              for base in (attr, value, house, role)]
    attr = torch.where(content, offset.clamp_min(0) // houses[:, None] + 1, attr)
    house = torch.where(content, offset.clamp_min(0) % houses[:, None] + 1, house)
    # Crucial: only a CURRENTLY VISIBLE answer digit may get a value feature.
    # MASK (or a sampled non-digit) contributes zero; no clean target is read.
    value = torch.where(content, (digits + 1).clamp_min(0), value)
    role = torch.where(target, 3, role)
    return dict(position_ids=position_ids, attribute=attr, house=house, value=value,
                role=role, houses=houses, attributes=attributes, target_mask=target)


class TypedCoordinates(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attribute = nn.Embedding(7, hidden_size, padding_idx=0)
        self.house = nn.Embedding(7, hidden_size, padding_idx=0)
        self.value = nn.Embedding(7, hidden_size, padding_idx=0)
        self.role = nn.Embedding(4, hidden_size, padding_idx=0)
        for module in (self.attribute, self.house, self.value, self.role):
            nn.init.normal_(module.weight, std=.02)
            with torch.no_grad():
                module.weight[0].zero_()

    def forward(self, features):
        return (self.attribute(features['attribute']) + self.house(features['house'])
                + self.value(features['value']) + self.role(features['role']))


def audit_public_dimensions(dataset):
    """Check public-only dimension inference on ALL rows, without gold digits.

    For evaluation JSONL, compare against recorded public metadata. For packed
    train, check consistency with public layout and every clue reference; no
    gold solution tokens or solver traces are examined.
    """
    import numpy as np
    from .data import encode_record
    tok = dataset.tokenizer
    c_id, n_id = tok.encode(['c', 'n'])
    lookup = np.full(tok.vocab_size, -1, dtype=np.int64)
    for value, index in enumerate(tok.encode(list('012345'))):
        lookup[index] = value
    counts = {}
    for begin in range(0, len(dataset), 8192):
        end = min(len(dataset), begin + 8192)
        if dataset.packed:
            ids = np.asarray(dataset.tokens[begin:end])
            layout = np.asarray(dataset.layout[begin:end])
        else:
            encoded = [encode_record(r, tok, dataset.max_length) for r in dataset.records[begin:end]]
            ids = np.array([v[0] for v in encoded])
            layout = np.array([v[1:] for v in encoded])
        prompt = np.arange(ids.shape[1])[None, :] < layout[:, :1] - 1
        # Do not even decode digit identities in the answer region for audit.
        digits = lookup[np.where(prompt, ids, tok.pad_id)]
        c_ref, n_ref = (ids == c_id) & prompt, (ids == n_id) & prompt
        attrs = np.where(c_ref, np.roll(digits, -1, axis=1) + 1, 0).max(axis=1)
        width = layout[:, 1] - layout[:, 0] - 1
        houses = width // np.maximum(attrs, 1)
        if not (((attrs >= 3) & (attrs <= 6) & (houses >= 3) & (houses <= 6)
                 & (attrs * houses == width)).all()):
            raise ValueError('Cannot infer public dimensions for every training example')
        a, v = np.roll(digits, -1, axis=1), np.roll(digits, -2, axis=1)
        if ((c_ref & ((a < 0) | (a >= attrs[:, None]) | (v < 0) | (v >= houses[:, None])))
            | (n_ref & ((a != 0) | (v < 0) | (v >= houses[:, None])))).any():
            raise ValueError('Out-of-domain clue reference')
        if not dataset.packed:
            for i, record in enumerate(dataset.records[begin:end]):
                if (int(houses[i]), int(attrs[i])) != (record['metadata']['houses'], record['metadata']['attributes']):
                    raise ValueError('Inferred and public metadata dimensions differ')
        for h in range(3, 7):
            for a in range(3, 7):
                key = f'{h}x{a}'
                counts[key] = counts.get(key, 0) + int(((houses == h) & (attrs == a)).sum())
    return dict(rows=len(dataset), public_dimensions_valid=True, sizes=counts,
                gold_values_used=False, solver_trace_used=False)
