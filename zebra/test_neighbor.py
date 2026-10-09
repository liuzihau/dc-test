import torch
import pytest
from torch import nn
from zebra.neighbor import NeighborHeads, neighbor_terms
from zebra.schedule import milestones


def terms(x0, xt, hidden=None, offsets=(-1, 1)):
    x0, xt = torch.tensor([x0]), torch.tensor([xt])
    hidden = torch.randn(1, x0.shape[1], 4, requires_grad=True) if hidden is None else hidden
    heads = NeighborHeads(4, 10, offsets)
    for p in heads.parameters():
        nn.init.zeros_(p)
    result, counts = neighbor_terms(heads, hidden, x0, xt, torch.ones_like(x0),
        torch.ones_like(x0), 9, [0, 8], torch.tensor([[2.]]), torch.tensor(5.))
    return result, counts, heads, hidden


def test_masked_target_clean_source_and_exact_scaling():
    result, counts, heads, hidden = terms([1,2,3,4,5], [1,2,9,4,5])
    assert counts[-1] == counts[1] == 1
    expected = 2 * torch.log(torch.tensor(9.)) / 5
    torch.testing.assert_close(result[-1], expected)
    torch.testing.assert_close(result[1], expected)
    (.25 * result[-1] + .25 * result[1]).backward()
    # Prev target is predicted from source index 3, next target from source 1.
    assert heads.heads[0][-1].bias.grad[3] < 0
    assert heads.heads[1][-1].bias.grad[3] < 0


def test_boundaries_edges_and_no_wrap():
    _, counts, _, _ = terms([1,8,3,4,0], [9,8,9,4,0])
    assert counts[-1] == 1  # source4 -> masked3
    assert counts[1] == 0


def test_no_pairs_zero_safe_and_all_heads_in_backward():
    losses, counts, heads, _ = terms([1,2,3], [1,2,3])
    assert all(x.item() == 0 for x in losses.values())
    sum(losses.values()).backward()
    assert all(p.grad is not None and not p.grad.any() for p in heads.parameters())


def test_offsets_cannot_cross_boundary():
    _, counts, _, _ = terms([1,8,3,4,5], [1,8,9,4,5], offsets=(-2,2))
    assert counts[2] == 0
    assert counts[-2] == 1


def test_no_three_epoch_overshoot():
    assert milestones(40) == list(range(3, 40, 3)) + [40]


def test_invalid_offsets():
    with pytest.raises(ValueError):
        NeighborHeads(4, 10, [-1, -1])
