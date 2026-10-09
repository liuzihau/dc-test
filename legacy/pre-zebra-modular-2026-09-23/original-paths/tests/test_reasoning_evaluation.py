"""CPU-only checks for leakage-free closed-loop and aligned task evaluation."""

import copy
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from reasoning.evaluation import _generator, evaluate_corruption, evaluate_generation


class TinyTokenizer:
    tokens = ["[PAD]", "[MASK]", "[BOS]", "[SEP]", "[EOS]", "a", "b", "c"]
    pad_id, mask_id, bos_id, sep_id, eos_id = range(5)
    special_ids = frozenset(range(5))
    vocab_size = 8

    def decode(self, values):
        return [self.tokens[int(value)] for value in values]


class TinyDataset(Dataset):
    def __init__(self, answers=None, tasks=None, size=4):
        self.tokenizer = TinyTokenizer()
        answers = answers or [[5, 4, 0, 0] for _ in range(size)]
        tasks = tasks or ["sudoku"] * len(answers)
        self.records = [dict(id=f"item-{i}", task=tasks[i], answer=self.tokenizer.decode(answer), metadata={})
                        for i, answer in enumerate(answers)]
        self.rows = []
        for i, answer in enumerate(answers):
            self.rows.append(dict(
                input_ids=torch.tensor([2, 5 + i % 3, 3] + answer + [0, 0]),
                attention_mask=torch.tensor([1] * 7 + [i % 2, 0], dtype=torch.bool),
                target_mask=torch.tensor([0] * 3 + [1] * 4 + [0, 0], dtype=torch.bool),
                record_index=torch.tensor(i),
            ))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class TinyCache(list):
    def __init__(self, entries, attention_mask):
        super().__init__(entries)
        self.attention_mask = attention_mask

    def index_select_batch(self, indices):
        return TinyCache([entry.index_select(0, indices) for entry in self],
                         self.attention_mask.index_select(0, indices))


class ToyModel(nn.Module):
    def __init__(self, mode="both", positions=False, answer_ids=None):
        super().__init__()
        self.config = {"memory_mode": mode}
        self.positions = positions
        self.answer_ids = answer_ids
        self.calls = []

    def forward(self, input_ids, attention_mask, previous_step_kv=None,
                previous_final_hidden=None, **kwargs):
        if self.config["memory_mode"] not in ("dcache", "both"):
            assert previous_step_kv is None
        if self.config["memory_mode"] not in ("final", "both"):
            assert previous_final_hidden is None
        assert kwargs["source_mask"] is None
        self.calls.append(dict(ids=input_ids.clone(), attention=attention_mask.clone(),
                               cache=copy.deepcopy(previous_step_kv),
                               final=copy.deepcopy(previous_final_hidden)))
        b, length = input_ids.shape
        logits = torch.zeros(b, length, 8)
        logits[..., 1] = 100  # The evaluator must never generate MASK.
        logits[..., 5] = torch.arange(length).float() + 1 if self.positions else 1
        if self.answer_ids is not None:
            logits[:, 3:7, :] = -100
            for position, value in enumerate(self.answer_ids, 3):
                logits[:, position, value] = 100
        tags = input_ids[:, 1, None, None].float().expand(b, length, 1).clone()
        return dict(logits=logits, final_hidden=tags + 20,
                    step_kv=TinyCache([tags], attention_mask.clone()))


def harmless_scorer(record, tokens):
    return {"exact_match": tokens == record["answer"], "valid_solution": False}


def run_generation(model, dataset=None, batch_size=2, **kwargs):
    dataset = dataset or TinyDataset()
    return evaluate_generation(model, DataLoader(dataset, batch_size=batch_size),
                               scorer=harmless_scorer, **kwargs)


def test_generation_masks_gold_eos_and_pad_and_preserves_prompt_and_outer_padding():
    first = TinyDataset(answers=[[5, 4, 0, 0], [5, 5, 4, 0]])
    second = TinyDataset(answers=[[6, 6, 6, 4], [7, 4, 0, 0]])
    before = copy.deepcopy(first.rows)
    left, right = ToyModel(), ToyModel()
    _, left_records = run_generation(left, first, seed=99)
    _, right_records = run_generation(right, second, seed=99)
    assert len(left.calls) == len(right.calls) == 4
    assert (left.calls[0]["ids"][:, 3:7] == 1).all()
    for a, b in zip(left.calls, right.calls):
        torch.testing.assert_close(a["ids"], b["ids"])
        assert a["ids"][:, 0].tolist() == [2, 2]
        assert a["ids"][:, 2].tolist() == [3, 3]
        assert (a["ids"][:, 7:] == 0).all()
        assert not a["attention"][:, -1].any()
    assert [r["predicted_answer_ids"] for r in left_records] == [r["predicted_answer_ids"] for r in right_records]
    for original, after in zip(before, first.rows):
        torch.testing.assert_close(original["input_ids"], after["input_ids"])


def test_top_prob_uses_first_k_of_tentative_order_then_preserves_remaining_order():
    model = ToyModel(positions=True)
    metrics, details = run_generation(model, TinyDataset(size=2), candidate_k=2, seed=41,
                                     token_selection="argmax")
    for index, item in enumerate(details):
        order = (torch.randperm(4, generator=_generator(41, index, "order")) + 3).tolist()
        expected = []
        while order:
            position = max(order[:2])  # Confidence grows monotonically with position.
            expected.append(position)
            order.remove(position)
        assert item["decode_order"] == expected
    assert metrics["candidate_k"] == 2
    assert metrics["nfe"] == 4
    assert metrics["mean_nfe_per_example"] == 4
    assert metrics["tokens_processed"] == 4 * 2 * 9
    assert "FLOPs" in metrics["compute_unit"]


def test_uniform_uses_seeded_order_and_paper_countdown_is_greedy():
    dataset = TinyDataset(size=2, tasks=["sudoku", "countdown"])
    metrics, details = run_generation(ToyModel(), dataset, policy="uniform", seed=9)
    for index, item in enumerate(details):
        expected = (torch.randperm(4, generator=_generator(9, index, "order")) + 3).tolist()
        assert item["decode_order"] == expected
        assert 1 not in item["predicted_answer_ids"]
    assert details[0]["token_selection"] == "sample"
    assert details[1]["token_selection"] == "argmax"
    assert details[1]["predicted_answer_ids"] == [5] * 4
    assert metrics["candidate_k"] is None


def test_seeded_generation_is_batching_invariant_for_memory_independent_model():
    _, left = run_generation(ToyModel(), batch_size=1, seed=102)
    _, right = run_generation(ToyModel(), batch_size=4, seed=102)
    assert [item["predicted_answer_ids"] for item in left] == [item["predicted_answer_ids"] for item in right]
    assert [item["decode_order"] for item in left] == [item["decode_order"] for item in right]


@pytest.mark.parametrize("condition", ["correct", "none", "shuffle_dcache", "shuffle_final", "shuffle_both"])
def test_memories_reset_between_batches_and_derangement_preserves_cache_metadata(condition):
    model = ToyModel()
    run_generation(model, memory_condition=condition)
    assert model.calls[0]["cache"] is None and model.calls[0]["final"] is None
    assert model.calls[4]["cache"] is None and model.calls[4]["final"] is None
    second = model.calls[1]
    if condition == "none":
        assert second["cache"] is None and second["final"] is None
        return
    cache_order = [1, 0] if condition in ("shuffle_dcache", "shuffle_both") else [0, 1]
    final_order = [1, 0] if condition in ("shuffle_final", "shuffle_both") else [0, 1]
    assert second["cache"][0][:, 0, 0].tolist() == [5 + i for i in cache_order]
    assert second["final"][:, 0, 0].tolist() == [25 + i for i in final_order]
    torch.testing.assert_close(second["cache"].attention_mask,
                               model.calls[0]["attention"][cache_order])


def test_singleton_shuffle_is_explicit_error_not_identity_shuffle():
    with pytest.raises(ValueError, match="batch size >= 2"):
        run_generation(ToyModel(), TinyDataset(size=1), memory_condition="shuffle_both")


@pytest.mark.parametrize("mode", ["none", "dcache", "final", "both"])
def test_returned_hidden_does_not_enable_absent_memory_paths(mode):
    model = ToyModel(mode=mode)
    run_generation(model)
    assert (model.calls[1]["cache"] is not None) == (mode in ("dcache", "both"))
    assert (model.calls[1]["final"] is not None) == (mode in ("final", "both"))


def test_eos_does_not_hide_unfinished_prefix_or_suffix_from_scorer():
    seen = []

    def scorer(record, tokens):
        seen.append(tokens)
        return {"valid_solution": tokens == ["a", "[EOS]", "[PAD]", "[PAD]"]}

    dataset = TinyDataset(size=2)
    model = ToyModel(answer_ids=[5, 4, 0, 7])
    metrics, details = evaluate_generation(model, DataLoader(dataset, batch_size=2), scorer=scorer,
                                          token_selection="argmax")
    assert len(model.calls) == 4  # EOS prediction is not a gold-length shortcut.
    assert seen == [["a", "[EOS]", "[PAD]", "c"]] * 2
    assert details[0]["predicted_answer"] == ["a"]
    assert details[0]["has_eos"]
    assert metrics["valid_solution"] == 0


def test_max_steps_is_gold_independent_budget_and_restores_training_mode():
    model = ToyModel()
    assert model.training
    metrics, details = run_generation(model, max_steps=2)
    assert model.training
    assert metrics["nfe"] == 4  # Two batches, each capped at two forwards.
    assert all(item["remaining_masked_slots"] == 2 for item in details)
    assert metrics["batch_completion_rate"] == 0


def test_fixed_corruption_is_cold_by_default_and_nested_masks_are_paired():
    dataset = TinyDataset(size=2)
    first, second = ToyModel(), ToyModel()
    metrics, details = evaluate_corruption(first, DataLoader(dataset, batch_size=2), seed=85)
    _, repeated = evaluate_corruption(second, DataLoader(dataset, batch_size=1), seed=85)
    assert metrics["protocol"] == "cold_independent"
    assert all(call["cache"] is None and call["final"] is None for call in first.calls)
    assert "valid_solution" not in metrics and "exact_match" not in metrics
    keyed = lambda values: {(item["record_index"], item["mask_ratio"]): item["masked_positions"] for item in values}
    assert keyed(details) == keyed(repeated)
    by_ratio = {item["mask_ratio"]: set(item["masked_positions"]) for item in details if item["record_index"] == 0}
    assert by_ratio[0.1] <= by_ratio[0.3] <= by_ratio[0.5] <= by_ratio[0.7]
    for call in first.calls:
        assert call["ids"][:, 0].tolist() == [2, 2]
        assert call["ids"][:, 2].tolist() == [3, 3]
        assert (call["ids"][:, 7:] == 0).all()


def test_fixed_corruption_nested_memory_is_explicit_and_resets_each_batch():
    model = ToyModel()
    metrics, _ = evaluate_corruption(model, DataLoader(TinyDataset(), batch_size=2),
                                    reset_each_ratio=False)
    assert metrics["protocol"] == "teacher_forced_nested"
    assert model.calls[0]["cache"] is None and model.calls[4]["cache"] is None
    assert model.calls[1]["cache"] is not None and model.calls[5]["cache"] is not None


def test_fixed_corruption_nll_has_separate_content_denominator():
    dataset = TinyDataset(size=2)
    metrics, _ = evaluate_corruption(ToyModel(), DataLoader(dataset, batch_size=2), ratios=(1.0,))
    result = metrics["ratios"]["1.0"]
    log_normalizer = math.log(math.e + 6)  # Seven legal symbols; MASK excluded.
    assert result["conditional_nll"] == pytest.approx(log_normalizer - 0.25)
    assert result["content_conditional_nll"] == pytest.approx(log_normalizer - 1)
    assert result["masked_tokens"] == 8 and result["content_tokens"] == 2
    assert result["masked_token_accuracy"] == 0.25
    assert result["content_masked_token_accuracy"] == 1


@pytest.mark.parametrize("kwargs", [dict(policy="top_p"), dict(candidate_k=0), dict(tokens_per_step=0), dict(max_steps=0)])
def test_generation_rejects_ambiguous_or_invalid_decoding_options(kwargs):
    with pytest.raises(ValueError):
        run_generation(ToyModel(), **kwargs)


@pytest.mark.parametrize("mode", ["none", "dcache", "final", "both"])
def test_real_reasoning_model_integrates_with_both_evaluators_on_cpu(mode):
    from reasoning.model import ReasoningModel

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = ReasoningModel(dict(vocab_size=8, hidden_size=16, n_heads=2,
                                    n_layers=1, max_length=9, memory_mode=mode))
        metrics, details = run_generation(model, TinyDataset(size=2), max_steps=2,
                                         memory_condition="shuffle_both")
        assert metrics["nfe"] == 2 and len(details) == 2
        corrupted, _ = evaluate_corruption(model, DataLoader(TinyDataset(size=2), batch_size=2),
                                          ratios=(0.3, 0.7), reset_each_ratio=False,
                                          memory_condition="shuffle_both")
        assert all(math.isfinite(value["conditional_nll"]) for value in corrupted["ratios"].values())
    finally:
        torch.set_num_threads(old_threads)
