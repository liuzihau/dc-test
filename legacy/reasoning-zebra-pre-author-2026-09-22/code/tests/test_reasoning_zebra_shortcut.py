"""The permutation calibration uses revealed answers, never clue/gold shortcuts."""

import math

import pytest
import torch

from reasoning.tasks import TaskTokenizer
from reasoning.zebra_shortcut import (permutation_shortcut_probabilities,
                                     score_permutation_shortcut)


def example(prefix=4):
    tokenizer = TaskTokenizer("zebra")
    answer = tokenizer.encode(list("12345") * 5) + [tokenizer.eos_id]
    clean = torch.tensor([[tokenizer.bos_id] * prefix + answer + [tokenizer.pad_id] * 3])
    target = torch.zeros_like(clean, dtype=torch.bool)
    target[:, prefix:prefix + 26] = True
    return tokenizer, clean, target


def test_fully_masked_uniform_house_digits_and_eos_excluded():
    tokenizer, clean, target = example()
    current = clean.masked_fill(target, tokenizer.mask_id)
    result = permutation_shortcut_probabilities(current, target, tokenizer)
    assert int(result["eligible"].sum()) == 25
    assert not result["eligible"][0, 29]
    expected = torch.zeros(tokenizer.vocab_size)
    expected[tokenizer.encode(list("12345"))] = .2
    torch.testing.assert_close(result["probabilities"][result["eligible"]], expected.expand(25, -1))
    assert not result["probabilities"][~result["eligible"]].any()
    metrics = score_permutation_shortcut(current, clean, target, tokenizer)
    assert metrics["content_conditional_nll"] == pytest.approx(math.log(5))
    assert metrics["content_tokens"] == 25
    assert metrics["content_correct"] == 5
    assert metrics["expected_content_masked_token_accuracy"] == pytest.approx(.2)


def test_only_unused_digits_receive_probability():
    tokenizer, clean, target = example()
    current = clean.clone()
    current[0, [5, 7]] = tokenizer.mask_id  # First category: missing 2 and 4.
    prediction = permutation_shortcut_probabilities(current, target, tokenizer)
    assert prediction["eligible"].sum() == 2
    expected = torch.zeros(tokenizer.vocab_size)
    expected[tokenizer.encode(["2", "4"])] = .5
    torch.testing.assert_close(prediction["probabilities"][0, 5], expected)
    torch.testing.assert_close(prediction["probabilities"][0, 7], expected)
    metrics = score_permutation_shortcut(current, clean, target, tokenizer)
    assert metrics["content_conditional_nll"] == pytest.approx(math.log(2))
    assert metrics["content_correct"] == 1
    assert metrics["expected_content_correct"] == 1


def test_prompt_change_does_not_change_predictions_or_scores():
    tokenizer, clean, target = example()
    current = clean.masked_fill(target, tokenizer.mask_id)
    changed = current.clone()
    changed[~target] = tokenizer.token_to_id["SAME"]
    first = permutation_shortcut_probabilities(current, target, tokenizer)
    second = permutation_shortcut_probabilities(changed, target, tokenizer)
    torch.testing.assert_close(first["probabilities"], second["probabilities"], rtol=0, atol=0)
    assert score_permutation_shortcut(current, clean, target, tokenizer) == score_permutation_shortcut(
        changed, clean, target, tokenizer)


def test_masked_gold_permutation_does_not_enter_predictions():
    tokenizer, clean, target = example()
    current = clean.masked_fill(target, tokenizer.mask_id)
    changed_gold = clean.clone()
    changed_gold[0, 4:9] = changed_gold[0, 4:9].roll(2)
    first = score_permutation_shortcut(current, clean, target, tokenizer)
    second = score_permutation_shortcut(current, changed_gold, target, tokenizer)
    assert first == second


def test_single_missing_digit_is_certain_and_eos_only_is_empty():
    tokenizer, clean, target = example()
    current = clean.clone()
    current[0, 4] = tokenizer.mask_id
    metrics = score_permutation_shortcut(current, clean, target, tokenizer)
    assert metrics["content_tokens"] == 1
    assert metrics["content_conditional_nll"] == 0
    assert metrics["expected_content_masked_token_accuracy"] == 1
    current = clean.clone()
    current[0, 29] = tokenizer.mask_id
    metrics = score_permutation_shortcut(current, clean, target, tokenizer)
    assert metrics["content_tokens"] == metrics["content_nll_sum"] == 0
    assert metrics["content_conditional_nll"] is None
    assert metrics["content_masked_token_accuracy"] is None


def test_variable_prefix_lengths_and_category_specific_mask_counts():
    tokenizer, first, first_target = example(prefix=4)
    _, second, second_target = example(prefix=8)
    first = torch.nn.functional.pad(first, (0, 4))
    first_target = torch.nn.functional.pad(first_target, (0, 4))
    clean = torch.cat((first, second))
    target = torch.cat((first_target, second_target))
    current = clean.clone()
    current[0, [4, 5, 6]] = tokenizer.mask_id
    current[1, [8, 13, 18, 23, 28]] = tokenizer.mask_id
    metrics = score_permutation_shortcut(current, clean, target, tokenizer)
    assert metrics["content_tokens"] == 8
    assert metrics["content_nll_sum"] == pytest.approx(3 * math.log(3))
    assert metrics["expected_content_correct"] == pytest.approx(6)


def test_duplicate_revealed_digits_rejected_not_repaired():
    tokenizer, clean, target = example()
    current = clean.clone()
    current[0, 5] = current[0, 4]
    with pytest.raises(ValueError, match="Repeated revealed"):
        permutation_shortcut_probabilities(current, target, tokenizer)


def test_wrong_revealed_token_rejected_by_scoring():
    tokenizer, clean, target = example()
    current = clean.clone()
    current[0, 4:9] = current[0, 4:9].roll(1)
    with pytest.raises(ValueError, match="Revealed answer tokens must match"):
        score_permutation_shortcut(current, clean, target, tokenizer)


def test_invalid_layout_and_special_content_rejected():
    tokenizer, clean, target = example()
    bad_target = target.clone()
    bad_target[0, 4] = False
    with pytest.raises(ValueError, match="requires 25 content"):
        permutation_shortcut_probabilities(clean, bad_target, tokenizer)
    bad_target[0, 1] = True
    with pytest.raises(ValueError, match="contiguous"):
        permutation_shortcut_probabilities(clean, bad_target, tokenizer)
    current = clean.clone()
    current[0, 4] = tokenizer.eos_id
    with pytest.raises(ValueError, match="house digits only"):
        permutation_shortcut_probabilities(current, target, tokenizer)


def test_non_zebra_tokenizer_rejected():
    _, clean, target = example()
    with pytest.raises(ValueError, match="Zebra tokenizer"):
        permutation_shortcut_probabilities(clean, target, TaskTokenizer("sudoku"))
