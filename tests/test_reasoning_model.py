"""CPU correctness for the isolated conditional reasoning control suite."""
import copy

import pytest
import torch

from reasoning.model import CacheState, ReasoningModel


@pytest.fixture(autouse=True)
def small_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(**kwargs):
    return dict(vocab_size=17, hidden_size=32, n_heads=4, n_layers=2,
                max_length=16, special_ids=[0, 1, 2, 3],
                **kwargs)


def batch():
    x = torch.tensor([[2, 4, 5, 3, 6, 7, 8, 9, 10, 11, 3, 0, 0, 0],
                      [2, 5, 4, 3, 9, 6, 7, 8, 11, 10, 3, 0, 0, 0]])
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[:, -1] = False
    target = valid.clone()
    target[:, :4] = False
    return dict(input_ids=x, attention_mask=valid, target_mask=target)


def nonzero_head(m):
    torch.nn.init.normal_(m.backbone.output_layer.linear.weight, std=.05)
    if m.has_final:
        m.backbone.dcachehooping_latent_norm.weight.data.fill_(.2)
    return m


@pytest.mark.parametrize("memory", ["none", "final", "dcache", "both"])
@pytest.mark.parametrize("neighbors", [False, True])
def test_controls_backward_and_checkpoint_roundtrip(memory, neighbors, tmp_path):
    m = ReasoningModel(config(memory_mode=memory, neighbors=neighbors))
    b = batch()
    loss, metrics = m.compute_loss(b, step=1500,
                                  generator=torch.Generator().manual_seed(31))
    assert torch.isfinite(loss)
    assert metrics["loss_weight_sum"].item() == pytest.approx(2.05)
    assert metrics["adjacent_edges"].item() == (4 if m.has_dcache else 0)
    loss.backward()
    assert m.backbone.output_layer.linear.weight.grad.abs().sum() > 0
    if neighbors:
        assert m.backbone.neighbor_heads.heads["next"][1].weight.grad.abs().sum() > 0
    if not m.has_dcache:
        assert m.backbone.dc_final_writer is None
        assert not any("step_memory_gate" in name for name, _ in m.named_parameters())
    file = tmp_path / "weights.pt"
    torch.save(m.state_dict(), file)
    clone = ReasoningModel(m.config)
    clone.load_state_dict(torch.load(file, weights_only=True), strict=True)
    m.eval(), clone.eval()
    torch.testing.assert_close(m(b["input_ids"], b["attention_mask"])["logits"],
                               clone(b["input_ids"], b["attention_mask"])["logits"])


def test_vanilla_reference_no_artificial_recurrent_parameters():
    m = ReasoningModel(config(attention_mode="vanilla", memory_mode="none", trajectory="single"))
    b = batch()
    loss, metrics = m.compute_loss(b, generator=torch.Generator().manual_seed(17))
    loss.backward()
    assert metrics["num_forwards"] == 1
    assert m.backbone.blocks[0].dc_qkv is None
    assert m.backbone.dcachehooping_latent_norm is None


def test_forward_count_includes_identity_reference():
    m = ReasoningModel(config(memory_mode='both', identity_probability=1.0,
                               final_dropout=0.0))
    _, metrics = m.compute_loss(batch(), generator=torch.Generator().manual_seed(77))
    assert metrics['trajectory_forwards'] == 5
    assert metrics['identity_forwards'] == 1
    assert metrics['num_forwards'] == 6


@pytest.mark.parametrize("attention", ["vanilla", "merged"])
def test_padding_keys_cannot_change_valid_logits(attention):
    m = nonzero_head(ReasoningModel(config(attention_mode=attention))).eval()
    b = batch()
    changed = b["input_ids"].clone()
    changed[:, -1] = 16
    with torch.no_grad():
        a = m(b["input_ids"], b["attention_mask"])
        z = m(changed, b["attention_mask"])
    torch.testing.assert_close(a["logits"][b["attention_mask"]], z["logits"][b["attention_mask"]], rtol=0, atol=0)
    assert a["final_hidden"][:, -1].abs().sum() == 0


def test_previous_cache_padding_is_excluded_not_just_zeroed():
    m = nonzero_head(ReasoningModel(config(memory_mode="both"))).eval()
    b = batch()
    with torch.no_grad():
        first = m(b["input_ids"], b["attention_mask"])
        assert all(entry[:, -1].abs().sum() == 0 for entry in first["step_kv"])
        poison = CacheState([entry.clone() for entry in first["step_kv"]], b["attention_mask"])
        for entry in poison:
            entry[:, -1].fill_(100000.)
        a = m(b["input_ids"], b["attention_mask"], first["step_kv"])
        z = m(b["input_ids"], b["attention_mask"], poison)
    torch.testing.assert_close(a["logits"], z["logits"], rtol=0, atol=0)
    perm = torch.tensor([1, 0])
    shuffled = poison.index_select_batch(perm)
    torch.testing.assert_close(shuffled.attention_mask, b["attention_mask"][perm])


def test_nested_trajectories_and_clues_protected_with_short_answers():
    m = ReasoningModel(config())
    for n in [1, 2, 3, 4, 5, 9]:
        b = batch()
        b["target_mask"][:, 4+n:] = False
        tr = m.sample_trajectory(b, torch.Generator().manual_seed(23))
        counts = [mask.sum(-1) for mask in tr["masks"]]
        for index, (state, mask) in enumerate(zip(tr["states"], tr["masks"])):
            assert torch.equal(state[~b["target_mask"]], b["input_ids"][~b["target_mask"]])
            assert mask.any(-1).all()
            if index:
                assert (counts[index] <= counts[index-1]).all()
                assert not (mask & ~tr["masks"][index-1]).any()
                if n >= 5:
                    assert (counts[index] < counts[index-1]).all()


def test_matched_corruption_rng_independent_of_memory_auxiliary():
    a = ReasoningModel(config(memory_mode="none", neighbors=False))
    b = ReasoningModel(config(memory_mode="both", neighbors=True, identity_probability=1.))
    ga, gb = torch.Generator().manual_seed(57), torch.Generator().manual_seed(57)
    for _ in range(2):
        _, am = a.compute_loss(batch(), generator=ga)
        _, bm = b.compute_loss(batch(), generator=gb)
        for name in ["full", "t0", "t1", "t2", "t3"]:
            torch.testing.assert_close(am["mask_ratio_"+name], bm["mask_ratio_"+name])
        assert torch.equal(ga.get_state(), gb.get_state())


def test_validation_deterministic_without_auxiliary_or_identity():
    m = ReasoningModel(config(memory_mode="both", neighbors=True, identity_probability=1.)).eval()
    with torch.no_grad():
        a, am = m.compute_loss(batch(), training=False, generator=torch.Generator().manual_seed(9))
        b, bm = m.compute_loss(batch(), training=False, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert am["identity_loss"] == am["neighbor_loss"] == am["final_dropout"] == 0
    assert am["base_loss"] == a


def test_detached_final_never_backpropagates_to_previous_hidden():
    m = nonzero_head(ReasoningModel(config(memory_mode="final")))
    b = batch()
    hidden = torch.randn(2, 14, 32, requires_grad=True)
    out = m(b["input_ids"], b["attention_mask"], previous_final_hidden=hidden)
    out["logits"][..., 4:].square().mean().backward()
    assert hidden.grad is None
    assert m.backbone.dcachehooping_latent_norm.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("target", [1, 2, 3, 4])
def test_actual_model_one_hop_only(target):
    weights = [float(index == target) for index in range(5)]
    m = nonzero_head(ReasoningModel(config(memory_mode="dcache", weights=weights,
        identity_probability=0., cache_only_probability=0., current_only_probability=0.)))
    states = []
    def retain(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden.retain_grad()
        states.append(hidden)
    handle = m.backbone.blocks[-1].register_forward_hook(retain)
    loss, _ = m.compute_loss(batch(), generator=torch.Generator().manual_seed(7))
    handle.remove()
    loss.backward()
    assert len(states) == 5
    for index, hidden in enumerate(states):
        nonzero = hidden.grad is not None and bool(hidden.grad.abs().sum() > 0)
        assert nonzero == (index in [target-1, target])


def test_source_dropout_keeps_clue_input_and_only_masks_answer_queries(monkeypatch):
    m = ReasoningModel(config(memory_mode="dcache", cache_only_probability=1., current_only_probability=0.))
    original = m.forward
    seen = []
    def tracked(inputs, valid, *args, **kwargs):
        seen.append((inputs.detach().clone(), kwargs.get("source_mask")))
        return original(inputs, valid, *args, **kwargs)
    monkeypatch.setattr(m, "forward", tracked)
    b = batch()
    m.compute_loss(b, step=1000, generator=torch.Generator().manual_seed(5))
    for state, source in seen:
        torch.testing.assert_close(state[:, :4], b["input_ids"][:, :4])
        if source is not None:
            assert not source[:, :4].any()
            assert not source[state.ne(m.mask_id)].any()


def test_no_neighbor_pairs_returns_safe_zero():
    m = ReasoningModel(config(neighbors=True))
    b = batch()
    b["input_ids"][:, 4:] = 0
    loss, metrics = m.compute_loss(b, generator=torch.Generator().manual_seed(77))
    assert torch.isfinite(loss)
    assert metrics["neighbor_loss"] == 0
    assert metrics["content_only_nll"] == 0
    loss.backward()


def test_unknown_configuration_key_rejected():
    with pytest.raises(ValueError, match="Unknown.*source_dropout_warmup"):
        ReasoningModel(config(source_dropout_warmup=99))


def test_shared_initialization_unchanged_by_auxiliary_or_memory():
    torch.manual_seed(981)
    baseline = ReasoningModel(config(memory_mode="none", neighbors=False))
    torch.manual_seed(981)
    extended = ReasoningModel(config(memory_mode="both", neighbors=True))
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, extended.state_dict()[name], rtol=0, atol=0)


def test_external_autocast_control_used_by_reasoning_only():
    m = nonzero_head(ReasoningModel(config()))
    b = batch()
    plain = m(b["input_ids"], b["attention_mask"])
    assert plain["logits"].dtype == torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = m(b["input_ids"], b["attention_mask"])
    assert torch.isfinite(mixed["logits"][..., 2:]).all()
    torch.testing.assert_close(plain["logits"][..., 2:], mixed["logits"][..., 2:], rtol=.1, atol=.02)


def test_no_unused_time_conditioning_parameters_in_reasoning():
    m = ReasoningModel(config(memory_mode="both"))
    assert not m.backbone.adaLN
    assert not hasattr(m.backbone, "sigma_map")
    assert not any("adaLN_modulation" in name for name, _ in m.named_parameters())
    b = batch()
    with pytest.raises(ValueError, match="without time conditioning"):
        m.backbone(b["input_ids"], sigma=torch.ones(2, 1), attention_mask=b["attention_mask"])
