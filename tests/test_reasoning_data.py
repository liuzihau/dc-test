import copy
import json
import random

import pytest
import torch
from torch.utils.data import DataLoader

from reasoning.data import ReasoningDataset, prepare_dataset
from reasoning.tasks import (SPECIAL_TOKENS, TASK_LENGTHS, TaskTokenizer,
                             _countdown_chain, _parse_zebra_prompt,
                             _sudoku_solutions, _zebra_solutions,
                             generate_record, score_prediction,
                             task_answer_slots, task_identity, validate_record)


@pytest.mark.parametrize("task", ["sudoku", "zebra", "countdown"])
def test_generated_reference_solves_task_and_is_deterministic(task):
    first = generate_record(task, random.Random(82))
    second = generate_record(task, random.Random(82))
    assert first == second
    assert score_prediction(first, first["answer"])["valid_solution"]
    assert not score_prediction(first, ["[MASK]"]) ["valid_solution"]
    tokenizer = TaskTokenizer(task)
    assert tokenizer.decode(tokenizer.encode(first["prompt"])) == first["prompt"]
    assert tokenizer.decode(range(5)) == list(SPECIAL_TOKENS)
    with pytest.raises(ValueError, match="outside"):
        tokenizer.encode(["out_of_vocabulary"])
    with pytest.raises(ValueError, match="outside"):
        tokenizer.decode([-1])


@pytest.mark.parametrize("task", ["sudoku", "zebra", "countdown"])
def test_prepare_tensor_masks_batching_integrity_and_disjointness(tmp_path, task):
    output = tmp_path / task
    manifest = prepare_dataset(output, task, train_size=4, valid_size=2, test_size=2, seed=4)
    assert manifest["source"]["kind"] == "synthetic_pilot"
    assert manifest["source"]["benchmark_equivalence"] is False
    ids = []
    for split in ("train", "valid", "test"):
        dataset = ReasoningDataset(output, split)
        ids.extend(r["id"] for r in dataset.records)
        item = dataset[0]
        assert set(item) == {"input_ids", "attention_mask", "target_mask", "record_index"}
        assert item["input_ids"].shape == (TASK_LENGTHS[task],)
        assert item["target_mask"].sum() == task_answer_slots(task)
        assert torch.all(item["attention_mask"][item["target_mask"]])
        start = 2 + len(dataset.records[0]["prompt"])
        assert not item["target_mask"][:start].any()
        decoded = dataset.tokenizer.decode(item["input_ids"][item["target_mask"]])
        assert score_prediction(dataset.records[0], decoded)["valid_solution"]
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        assert batch["input_ids"].shape == (2, TASK_LENGTHS[task])
    assert len(ids) == len(set(ids))
    with pytest.raises(FileExistsError):
        prepare_dataset(output, task)
    path = output / "train.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        ReasoningDataset(output)


def test_countdown_fixed_slots_do_not_leak_reference_answer_length(tmp_path):
    output = tmp_path / "countdown"
    prepare_dataset(output, "countdown", train_size=2, valid_size=0, test_size=0)
    dataset = ReasoningDataset(output)
    record = {"task": "countdown", "prompt": list("1,2,3,4,5=15"),
              "answer": list("1+2=3,3+3=6,4+5=9,6+9=15")}
    alternate = copy.deepcopy(record)
    alternate["answer"] = list("1+5=6,6+4=10,10+3=13,13+2=15")
    dataset.records = [validate_record(record), validate_record(alternate)]
    a, b = dataset[0], dataset[1]
    assert len(record["answer"]) != len(alternate["answer"])
    assert torch.equal(a["target_mask"], b["target_mask"])
    assert torch.equal(a["attention_mask"], b["attention_mask"])
    assert torch.equal(a["input_ids"][~a["target_mask"]], b["input_ids"][~b["target_mask"]])
    # Trailing answer PAD is generated/supervised, not excluded by attention.
    assert (a["target_mask"] & (a["input_ids"] == dataset.tokenizer.pad_id)).any()


def test_sudoku_unique_solution_clues_and_units():
    record = generate_record("sudoku", random.Random(3))
    assert len(_sudoku_solutions([int(t) for t in record["prompt"]])) == 1
    wrong = list(record["answer"])
    wrong[0] = wrong[1]
    assert not score_prediction(record, wrong)["valid_solution"]
    # Digit permutation remains a valid Sudoku, but violates the input clues.
    permuted = [str(int(t) % 9 + 1) for t in record["answer"]]
    score = score_prediction(record, permuted)
    assert score["constraints_satisfied"] and not score["clues_preserved"]
    assert not score["valid_solution"]
    bad_prompt = copy.deepcopy(record); bad_prompt["prompt"] = ["0"] * 80
    with pytest.raises(ValueError, match="81"):
        validate_record(bad_prompt)


def test_zebra_unique_constraints_and_permutation_scoring():
    record = generate_record("zebra", random.Random(24))
    constraints = _parse_zebra_prompt(record["prompt"])
    assert any(c[0] in ("LEFT", "NEXT") for c in constraints)
    assert len(_zebra_solutions(constraints)) == 1
    assert not score_prediction(record, ["1"] * 25)["valid_solution"]
    changed = list(record["answer"])
    changed[0], changed[1] = changed[1], changed[0]
    assert not score_prediction(record, changed)["valid_solution"]
    with pytest.raises(ValueError, match="clue"):
        _parse_zebra_prompt(["AT", "C0", "V0", "1"])
    with pytest.raises(ValueError, match="clue"):
        _parse_zebra_prompt(["AT", "C0", "V0", "12", ";"])


def test_countdown_safe_exact_arithmetic_and_operand_multiplicity():
    valid = list("1+2=3,3+3=6,4+5=9,6+9=15")
    assert _countdown_chain(valid, [1, 2, 3, 4, 5], 15)
    assert not _countdown_chain(valid, [1, 2, 4, 4, 5], 15)
    assert not _countdown_chain(valid, [1, 2, 3, 4, 5], 14)
    assert not _countdown_chain(list("1/0=0,2+3=5,4+5=9,0+9=9"), [1, 0, 2, 3, 4], 9)
    assert not _countdown_chain(list("1/2=0,0+3=3,4+5=9,3+9=12"), [1, 2, 3, 4, 5], 12)
    assert not _countdown_chain(list("__import__('os').system('true')"), [1, 2, 3, 4, 5], 15)
    assert not _countdown_chain(list("1+1=2,2+2=4,3+4=7,4+7=11"), [1, 2, 3, 4, 5], 11)


def test_special_tokens_and_junk_after_eos_are_not_silently_removed():
    record = generate_record("countdown", random.Random(9))
    good = record["answer"] + ["[EOS]", "[PAD]"]
    assert score_prediction(record, good)["valid_solution"]
    assert not score_prediction(record, good + ["1"])["valid_solution"]
    assert not score_prediction(record, ["[PAD]"] + good)["valid_solution"]
    assert not score_prediction(record, record["answer"] + ["[MASK]"])["valid_solution"]


def test_import_rejects_duplicate_prompt_even_if_id_or_answer_differs(tmp_path):
    record = generate_record("countdown", random.Random(11))
    other = copy.deepcopy(record); other["id"] = "different-id"
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(record) + "\n" + json.dumps(other) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_dataset(tmp_path / "out", "countdown", train_size=1, valid_size=1, test_size=0, input_path=source)


def test_serialization_order_does_not_create_new_logical_tasks():
    first = {"task": "countdown", "prompt": list("1,2,3,4,5=15")}
    other = {"task": "countdown", "prompt": list("5,4,3,2,1=15")}
    assert task_identity(first) == task_identity(other)
    zebra_a = {"task": "zebra", "prompt": ["AT", "C0", "V0", "1", ";", "SAME", "C0", "V0", "C1", "V2", ";"]}
    zebra_b = {"task": "zebra", "prompt": ["SAME", "C1", "V2", "C0", "V0", ";", "AT", "C0", "V0", "1", ";"]}
    assert task_identity(zebra_a) == task_identity(zebra_b)


def test_explicit_import_preserves_splits_and_rejects_cross_split_leakage(tmp_path):
    rng = random.Random(19)
    paths = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / (split + ".jsonl")
        path.write_text(json.dumps(generate_record("countdown", rng)) + "\n")
        paths[split] = path
    output = tmp_path / "prepared"
    result = prepare_dataset(output, "countdown", input_path=paths)
    assert result["source"]["split_policy"] == "preserved_explicit_splits"
    assert len(ReasoningDataset(output)) == 1
    paths["validation"] = paths["train"]
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_dataset(tmp_path / "bad", "countdown", input_path=paths)


def test_import_rejects_invalid_labels_oov_and_overflow(tmp_path):
    record = generate_record("zebra", random.Random(2))
    bad = copy.deepcopy(record); bad["answer"] = ["1"] * 25
    with pytest.raises(ValueError, match="does not solve"):
        validate_record(bad)
    bad = copy.deepcopy(record); bad["prompt"] = record["prompt"] * 4
    with pytest.raises(ValueError, match="sequence length"):
        validate_record(bad)
    bad = copy.deepcopy(record); bad["prompt"][0] = "[MASK]"
    with pytest.raises(ValueError, match="special"):
        validate_record(bad)
    with pytest.raises(ValueError, match="does not match"):
        validate_record(record, "sudoku")


def test_empty_eval_splits_and_invalid_requested_sizes(tmp_path):
    prepared = tmp_path / "data"
    prepare_dataset(prepared, "sudoku", train_size=1, valid_size=0, test_size=0)
    assert len(ReasoningDataset(prepared, "validation")) == 0
    assert len(ReasoningDataset(prepared, "test")) == 0
    with pytest.raises(ValueError, match="positive"):
        prepare_dataset(tmp_path / "zero", "countdown", train_size=0)
    with pytest.raises(ValueError, match="nonnegative"):
        prepare_dataset(tmp_path / "negative", "countdown", valid_size=-1)


def test_manifest_layout_and_file_paths_are_validated(tmp_path):
    prepared = tmp_path / "data"
    prepare_dataset(prepared, "countdown", train_size=1, valid_size=0, test_size=0)
    path = prepared / "manifest.json"
    original = json.loads(path.read_text())
    bad = copy.deepcopy(original); bad["answer_slots"] -= 1
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="layout"):
        ReasoningDataset(prepared)
    bad = copy.deepcopy(original); bad["splits"]["train"]["filename"] = "../train.jsonl"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="inside"):
        ReasoningDataset(prepared)
