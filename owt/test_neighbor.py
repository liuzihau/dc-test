"""Numerical checks for masked-target alignment, weighting and chunk gradients."""
import copy
import torch
from torch.nn import functional as F
from owt.neighbor import NeighborHeads, neighbor_terms


def test_chunked_loss_and_gradients_match_explicit_token_sum():
    torch.manual_seed(34)
    heads = NeighborHeads(4, 7, [-1, 1]).double()
    reference = copy.deepcopy(heads)
    hidden = torch.randn(2, 6, 4, dtype=torch.float64, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    clean = torch.tensor([[1, 2, 3, 0, 4, 5], [2, 1, 5, 4, 3, 0]])
    noisy = torch.tensor([[1, 6, 3, 0, 6, 6], [6, 1, 6, 4, 3, 0]])
    valid = torch.ones_like(clean)
    weights = torch.tensor([[2.], [3.]], dtype=torch.float64)
    denom = valid.sum()
    actual, counts = neighbor_terms(heads, hidden, clean, noisy, valid, 6, [0], weights, denom, 1)
    expected = {}
    for offset, head in zip(reference.offsets, reference.heads):
        value = other.sum()*0
        n = 0
        for b in range(2):
            for source in range(6):
                target = source+offset
                if not 0 <= target < 6 or target == 0 or noisy[b, target] != 6:
                    continue
                if clean[b, source] == 0 or clean[b, target] == 0:
                    continue
                logits = head(other[b, source]).clone()
                logits[6] = -torch.inf
                value = value + F.cross_entropy(logits[None], clean[b, target:target+1]) * weights[b, 0]/denom
                n += 1
        expected[offset] = value
        assert counts[offset] == n
        torch.testing.assert_close(actual[offset], value)
    sum(actual.values()).backward()
    sum(expected.values()).backward()
    torch.testing.assert_close(hidden.grad, other.grad)
    for a, b in zip(heads.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


def test_empty_pairs_are_finite_and_keep_all_heads_in_graph():
    heads = NeighborHeads(4, 7, [-1, 1])
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    clean = torch.ones(2, 3, dtype=torch.long)
    values, counts = neighbor_terms(heads, hidden, clean, clean, torch.ones_like(clean),
                                    6, [0], torch.ones(2, 1), clean.numel()*torch.ones(()))
    loss = sum(values.values())
    assert loss == 0 and sum(counts.values()) == 0
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.parameters())
