"""The diagnostic source split must equal the actual NP objective and gradients."""
import torch
from owt.neighbor import NeighborHeads,neighbor_terms
from analysis.owt_gradient_profile import source_losses


def test_source_split_matches_production_loss_and_feature_gradients():
    torch.manual_seed(123)
    heads=NeighborHeads(4,7,[-1,1]).double()
    hidden=torch.randn(2,6,4,dtype=torch.float64,requires_grad=True)
    clean=torch.tensor([[1,2,3,0,4,5],[2,1,5,4,3,0]])
    noisy=torch.tensor([[1,6,3,0,6,6],[6,1,6,4,3,0]])
    valid=torch.ones_like(clean)
    factors=torch.tensor([[2.],[3.]],dtype=torch.float64)
    coefficients={-1:.25,1:.25};denominator=valid.sum()
    visible,masked,counts=source_losses(heads,hidden,clean,noisy,valid,6,[0],
                                      factors,denominator,coefficients,chunk_size=1)
    terms,reference_counts=neighbor_terms(heads,hidden,clean,noisy,valid,6,[0],
                                         factors,denominator,chunk_size=1)
    reference=sum(coefficients[offset]*value for offset,value in terms.items())
    torch.testing.assert_close(visible+masked,reference,rtol=1e-12,atol=1e-12)
    assert counts['visible']>0 and counts['masked']>0
    assert sum(counts.values())==sum(int(value) for value in reference_counts.values())
    expected=torch.autograd.grad(reference,hidden,retain_graph=True)[0]
    gv=torch.autograd.grad(visible,hidden,retain_graph=True)[0]
    gm=torch.autograd.grad(masked,hidden)[0]
    torch.testing.assert_close(gv+gm,expected,rtol=1e-12,atol=1e-12)
