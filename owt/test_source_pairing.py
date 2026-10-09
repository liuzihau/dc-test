"""Source eligibility, gradients, count controls, RNG and real-model checks."""
import copy
from types import SimpleNamespace
from unittest.mock import patch
import csv
import pytest
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf

from owt.neighbor import NeighborHeads, neighbor_terms
from owt.source_pairing import pairing_terms, private_pair_generator
from owt.source_pairing_model import SourcePairingMDM
from owt.source_pairing_metrics import SourcePairMetrics
from owt.test_initialization import config, model
from owt.model import OWTMDM


def example():
    torch.manual_seed(34)
    heads = NeighborHeads(4, 7, [-1, 1]).double()
    hidden = torch.randn(2, 6, 4, dtype=torch.float64, requires_grad=True)
    clean = torch.tensor([[1,2,3,4,5,1], [2,1,5,0,3,4]])
    noisy = torch.tensor([[1,6,6,4,6,6], [6,6,5,0,6,6]])
    return heads, hidden, clean, noisy, torch.ones_like(clean), torch.tensor([[2.],[3.]],dtype=torch.float64)


def test_masked_source_loss_and_gradients_match_explicit_sum():
    heads, hidden, clean, noisy, valid, weights = example()
    reference = copy.deepcopy(heads)
    other = hidden.detach().clone().requires_grad_()
    actual, stats = pairing_terms(heads,hidden,clean,noisy,valid,6,[0],weights,valid.sum(),'masked_source',chunk_size=1)
    expected = {}
    for offset, head in zip(reference.offsets, reference.heads):
        total = other.sum()*0
        count = 0
        for b in range(2):
            for source in range(6):
                target = source + offset
                if not 0 < target < 6 or noisy[b,source] != 6 or noisy[b,target] != 6:
                    continue
                if clean[b,source] == 0 or clean[b,target] == 0:
                    continue
                logits = head(other[b,source]).masked_fill(torch.arange(7).eq(6),-torch.inf)
                total = total + F.cross_entropy(logits[None],clean[b,target:target+1])*weights[b,0]/valid.sum()
                count += 1
        expected[offset] = total
        assert stats[offset][:,2].sum() == count
        assert stats[offset][:,2].sum() == stats[offset][:,3].sum()
        torch.testing.assert_close(actual[offset],total)
    sum(actual.values()).backward();sum(expected.values()).backward()
    torch.testing.assert_close(hidden.grad,other.grad)
    assert not hidden.grad[0,0].any() and not hidden.grad[0,3].any()
    for a,b in zip(heads.parameters(),reference.parameters()):
        torch.testing.assert_close(a.grad,b.grad)


def test_full_mask_matches_original_terms_and_gradients():
    heads, hidden, clean, _, valid, weights = example()
    original = copy.deepcopy(heads)
    other = hidden.detach().clone().requires_grad_()
    noisy = torch.full_like(clean,6)
    values, stats = pairing_terms(heads,hidden,clean,noisy,valid,6,[0],weights,valid.sum(),'masked_source',chunk_size=2)
    old, counts = neighbor_terms(original,other,clean,noisy,valid,6,[0],weights,valid.sum(),chunk_size=2)
    for offset in (-1,1):
        torch.testing.assert_close(values[offset],old[offset],rtol=0,atol=0)
        assert stats[offset][:,2].sum() == counts[offset]
    sum(values.values()).backward();sum(old.values()).backward()
    torch.testing.assert_close(hidden.grad,other.grad,rtol=0,atol=0)
    for a,b in zip(heads.parameters(),original.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


def test_empty_pairs_keep_all_heads_connected():
    heads, hidden, clean, _, valid, weights = example()
    # Isolated masked targets have only visible neighbours.
    noisy = clean.clone();noisy[:,2] = 6
    values, stats = pairing_terms(heads,hidden,clean,noisy,valid,6,[0],weights,valid.sum(),'masked_source')
    assert sum(values.values()) == 0
    assert sum(s[:,2].sum() for s in stats.values()) == 0
    sum(values.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.parameters())


def test_count_control_matches_row_mass_preserves_main_rng_and_uses_visible_sources():
    heads, hidden, clean, noisy, valid, weights = example()
    generator = private_pair_generator(271828,0,0)
    rng = torch.get_rng_state().clone()
    values, stats = pairing_terms(heads,hidden,clean,noisy,valid,6,[0],weights,valid.sum(),'matched_pair_count',generator)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    visible = 0
    for s in stats.values():
        torch.testing.assert_close(s[:,1],s[:,2],rtol=0,atol=0)
        torch.testing.assert_close(s[:,5],s[:,6],rtol=0,atol=0)
        visible += (s[:,2]-s[:,3]).sum()
    assert visible > 0
    replay, _ = pairing_terms(heads,hidden,clean,noisy,valid,6,[0],weights,valid.sum(),'matched_pair_count',private_pair_generator(271828,0,0))
    for offset in (-1,1):
        torch.testing.assert_close(values[offset],replay[offset],rtol=0,atol=0)


def test_count_control_rejects_unmatched_token_weights_and_invalid_offsets():
    heads, hidden, clean, noisy, valid, _ = example()
    with pytest.raises(ValueError,match='row-constant'):
        pairing_terms(heads,hidden,clean,noisy,valid,6,[0],torch.arange(6)[None],valid.sum(),'matched_pair_count',private_pair_generator(1,0,0))
    heads.offsets = (-2,2)
    with pytest.raises(ValueError,match='offsets'):
        pairing_terms(heads,hidden,clean,noisy,valid,6,[0],torch.ones(2,1),valid.sum(),'masked_source')


def source_model(variant):
    cfg = config(variant)
    tokenizer = SimpleNamespace(vocab_size=7,mask_token=None,all_special_ids=[])
    with patch('diffusion.metrics.Metrics',return_value=torch.nn.Module()):
        return SourcePairingMDM(cfg,tokenizer).train()


def test_actual_recipes_preserve_initialization_rng_main_loss_and_validation():
    torch.set_num_threads(2)
    torch.manual_seed(71);baseline = model(config('mdm_np_zero_init'));expected_rng = torch.get_rng_state()
    x = torch.tensor([[1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4]])
    valid = torch.ones_like(x)
    torch.manual_seed(42);baseline._loss(x,valid.clone());train_rng = torch.get_rng_state();main = baseline._last_components['main_elbo']
    baseline.eval();torch.manual_seed(42);validation = baseline._loss(x,valid.clone())
    for variant in ('mdm_np_zero_init_masked_source','mdm_np_zero_init_pair_count_control'):
        cfg = config(variant);cfg.mechanisms.np.pop('source_policy');cfg.mechanisms.np.pop('pair_selection_seed')
        assert OmegaConf.to_container(cfg,resolve=True) == OmegaConf.to_container(config('mdm_np_zero_init'),resolve=True)
        torch.manual_seed(71);new = source_model(variant)
        torch.testing.assert_close(torch.get_rng_state(),expected_rng,rtol=0,atol=0)
        for name,tensor in baseline.state_dict().items():
            torch.testing.assert_close(tensor,new.state_dict()[name],rtol=0,atol=0)
        torch.manual_seed(42);result = new._loss(x,valid.clone())
        torch.testing.assert_close(new._last_components['main_elbo'],main,rtol=0,atol=0)
        torch.testing.assert_close(torch.get_rng_state(),train_rng,rtol=0,atol=0)
        assert new.pair_calls == 1 and torch.isfinite(result.loss)
        result.loss.backward()
        assert all(p.grad is not None for p in new.backbone.neighbor_heads.parameters())
        new.eval();torch.manual_seed(42);actual = new._loss(x,valid.clone())
        torch.testing.assert_close(actual.loss,validation.loss,rtol=0,atol=0)
        assert new.pair_calls == 1


def test_resume_restores_private_selection_counter_and_rejects_changed_policy():
    new = source_model('mdm_np_zero_init_pair_count_control')
    state = {'source_pairing':{'calls':17,'policy':'matched_pair_count','seed':271828}}
    with patch.object(OWTMDM,'on_load_checkpoint'):
        new.on_load_checkpoint(state)
    assert new.pair_calls == 17
    a = torch.randperm(20,generator=private_pair_generator(271828,new.pair_calls,1))
    b = torch.randperm(20,generator=private_pair_generator(271828,17,1))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    state['source_pairing']['policy'] = 'masked_source'
    with pytest.raises(ValueError,match='identical source policy'):
        new.on_load_checkpoint(state)


def test_pair_callback_sums_accumulated_microbatches_and_ranks(tmp_path):
    callback = SourcePairMetrics(tmp_path)
    trainer = SimpleNamespace(global_step=0,is_global_zero=True,strategy=SimpleNamespace(reduce=lambda x,reduce_op:x*2))
    module = SimpleNamespace(_last_pair_statistics={-1:torch.ones(5,7),1:torch.ones(5,7)})
    callback.on_train_start(trainer,module)
    callback.on_train_batch_end(trainer,module,None,None,0)
    module._last_pair_statistics = {-1:torch.ones(5,7)*3,1:torch.ones(5,7)*3};trainer.global_step=1
    callback.on_train_batch_end(trainer,module,None,None,1)
    with (tmp_path/'local_metrics/source_pairs.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows)==1 and rows[0]['optimizer_step']=='1'
    assert float(rows[0]['prev_maskbin0_selected'])==8
