"""Versioned, paper-informed benchmark reconstruction (not author-verified).

Keep old pilot/source layouts readable. None of these conversions retokenizes
an existing checkpoint. Unspecified special-token conventions are recorded,
not silently advertised as an exact reproduction of arXiv:2602.03769.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

PROTOCOL = {
    "version": "latent-tokens-reconstruction-v2",
    "paper": "https://arxiv.org/abs/2602.03769v1",
    "author_verified_exact_reproduction": False,
    "generation": "fully masked answer; one sampled token/forward; top-prob candidate-8; no repair",
    "primary_metric": "complete valid solution grid; format success reported separately",
    "unverified_conventions": [
        "Sudoku blanks share MASK; 9 digits + PAD/MASK/BOS/SEP/EOS = 14",
        "Zebra clues SEP attribute-major grid EOS, no BOS; 23-symbol vocabulary",
        "Zebra public dimensions determine answer width, including one EOS slot",
        "Outer PAD is neither visible nor supervised; EOS has a separate format score",
        "Zebra strict reference table plus clue checks; Sudoku valid grid preserving givens",
    ],
    "checkpoint_selection": "final for first trial (paper selects best validation loss)",
}


def public_dimensions(record):
    meta = record.get("metadata", {})
    if not isinstance(meta, dict):
        raise ValueError("Missing public puzzle dimensions")
    h, a = meta.get("houses"), meta.get("attributes")
    if type(h) is not int or type(a) is not int or h not in range(3, 7) or a not in range(3, 7):
        raise ValueError("Public houses/attributes must be integers 3..6")
    return h, a


def legacy_zebra_record(record):
    h, a = public_dimensions(record)
    return dict(record, task="zebra-official",
                prompt=["HOUSES", str(h), "ATTRS", str(a)] + list(record["prompt"]))


def score_benchmark(record, tokens):
    """Fixed public grid positions; formatting cannot hide a wrong/missing cell.

    Never scan gold for output length. A wrong EOS is reported independently
    from grid validity; strict_sequence_success requires both.
    """
    from .tasks import score_prediction, record_answer_slots
    count = record_answer_slots(record) - 1
    tokens = list(tokens)
    content = tokens[:count]
    format_ok = len(tokens) == count + 1 and tokens[-1] == "[EOS]"
    if record["task"] == "sudoku-benchmark":
        legacy = dict(record, task="sudoku", prompt=[
            "0" if t == "[MASK]" else t for t in record["prompt"]])
    else:
        legacy = legacy_zebra_record(record)
    # Reject special symbols in any grid cell, including premature EOS/PAD.
    if len(content) != count or any(t.startswith("[") for t in content):
        result = dict(exact_match=False, valid_solution=False, constraints_satisfied=False)
    else:
        result = score_prediction(legacy, content)
    result.update(format_success=format_ok,
                  strict_sequence_success=bool(result["valid_solution"] and format_ok))
    return result


def convert_zebra(record):
    from .tasks import validate_record
    old = validate_record(record, "zebra-official")
    # IDs and split membership deliberately survive this representation change.
    return validate_record(dict(old, task="zebra-benchmark", prompt=old["prompt"][4:]))


def convert_sudoku(row, split, index):
    from .tasks import validate_record
    row = np.asarray(row)
    if row.shape != (325,) or row.dtype.kind not in "iu":
        raise ValueError("Released Sudoku row must be 325 integers")
    given = int(row[0])
    cells = row[1:].reshape(81, 4).astype(np.int64)
    if not 0 <= given <= 81 or (cells[:, :2] < 0).any() or (cells[:, :2] > 8).any():
        raise ValueError("Invalid given count or Sudoku coordinates")
    if (cells[:, 2] < 1).any() or (cells[:, 2] > 9).any():
        raise ValueError("Released Sudoku values must be 1..9")
    positions = cells[:, 0] * 9 + cells[:, 1]
    if len(set(positions.tolist())) != 81:
        raise ValueError("Repeated/missing Sudoku cell")
    puzzle = ["[MASK]"] * 81
    answer = [None] * 81
    for offset, (position, value) in enumerate(zip(positions, cells[:, 2])):
        answer[int(position)] = str(value)
        if offset < given:
            puzzle[int(position)] = str(value)
    # Solver strategy and order are discarded, not fed as a reasoning trace.
    return validate_record(dict(task="sudoku-benchmark", prompt=puzzle, answer=answer,
                                metadata=dict(source_split=split, source_index=int(index),
                                              givens=given, solver_order_used=False,
                                              strategy_tokens_used=False)))


def _publish(output, task, splits, seed, source):
    from .data import write_prepared_dataset
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to replace dataset: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix="." + output.name + ".prepare-", dir=output.parent))
    manifest = write_prepared_dataset(staged, task, splits, seed, source)
    staged.rename(output)
    return manifest


def migrate_zebra(source_dir, output):
    """Preserve exact selected released puzzles, IDs, split order and provenance."""
    from .data import ReasoningDataset, _sha256
    datasets = {s: ReasoningDataset(source_dir, s) for s in ("train", "validation", "test")}
    if any(d.task != "zebra-official" for d in datasets.values()):
        raise ValueError("Only released zebra-official data can be migrated")
    splits = {s: [convert_zebra(r) for r in d.records] for s, d in datasets.items()}
    original = datasets["train"].manifest
    source = dict(original["source"], benchmark_equivalence=False,
                  previous_manifest_sha256=_sha256(Path(source_dir) / "manifest.json"),
                  split_policy="preserve every selected ID and order from source manifest",
                  representation="headerless public-size grids; no synthetic PAD answer cells")
    source["layout"] = {"max_length": 384, "answer_slots": "houses * attributes + EOS"}
    source["protocol"] = PROTOCOL
    source["max_packed_tokens"] = max(len(r["prompt"]) + len(r["answer"]) + 2
                                       for rs in splits.values() for r in rs)
    return _publish(output, "zebra-benchmark", splits, original["seed"], source)


def _puzzle_keys(rows):
    """Source puzzle identities independent of solver order/strategy and answers."""
    # Chunking bounds temporary RAM for the 1.8M-row training archive.
    for start in range(0, len(rows), 4096):
        block = np.asarray(rows[start:start + 4096])
        cells = block[:, 1:].reshape(-1, 81, 4).astype(np.int64)
        given = block[:, 0]
        positions = cells[:, :, 0] * 9 + cells[:, :, 1]
        if ((given < 0) | (given > 81)).any() or (cells[:, :, :2] < 0).any() or (cells[:, :, :2] > 8).any():
            raise ValueError("Invalid coordinates/givens in source")
        if not np.all(np.sort(positions, axis=1) == np.arange(81)):
            raise ValueError("Repeated/missing source Sudoku cell")
        values = cells[:, :, 2]
        if ((values < 1) | (values > 9)).any():
            raise ValueError("Invalid source Sudoku digits")
        boards = np.zeros((len(block), 81), dtype=np.uint8)
        boards[np.arange(len(block))[:, None], positions] = np.where(
            np.arange(81)[None, :] < given[:, None], values, 0)
        for board in boards:
            yield hashlib.sha256(board.tobytes()).digest()


def import_sudoku(train_file, test_file, output, train_size=20000, valid_size=1000,
                  test_size=1000, seed=17):
    from .data import _sha256
    if train_size < 1 or valid_size < 1 or test_size < 1:
        raise ValueError("Require positive train/validation/test sizes")
    arrays = {s: np.load(p, allow_pickle=False, mmap_mode="r")
              for s, p in (("train", train_file), ("test", test_file))}
    for array in arrays.values():
        if array.ndim != 2 or array.shape[1] != 325 or array.dtype.kind not in "iu":
            raise ValueError("Expected official numeric Sudoku N x 325 source")
    rng = np.random.default_rng(seed)
    test_keys = list(_puzzle_keys(arrays["test"]))
    reserved_test = set(test_keys)  # ALL official test puzzles excluded from train.
    seen, selected_test = set(), []
    for i in rng.permutation(len(test_keys)):
        if test_keys[i] not in seen:
            selected_test.append(convert_sudoku(arrays["test"][i], "test", i))
            seen.add(test_keys[i])
        if len(selected_test) == test_size:
            break
    # Fixed shuffled selection; no filtering by difficulty or model performance.
    order = rng.permutation(len(arrays["train"]))
    selected_train = []
    for begin in range(0, len(order), 4096):
        indices = order[begin:begin + 4096]
        rows = arrays["train"][indices]
        for i, row, key in zip(indices, rows, _puzzle_keys(rows)):
            if key in reserved_test or key in seen:
                continue
            selected_train.append(convert_sudoku(row, "train", i))
            seen.add(key)
            if len(selected_train) == train_size + valid_size:
                break
        if len(selected_train) == train_size + valid_size:
            break
    if len(selected_test) != test_size or len(selected_train) != train_size + valid_size:
        raise ValueError("Insufficient distinct official puzzles after leakage exclusion")
    splits = dict(train=selected_train[:train_size], validation=selected_train[train_size:], test=selected_test)
    source = dict(kind="shah_released_sudoku_subset", benchmark_equivalence=False,
                  repository="https://github.com/kulinshah98/llm-reasoning-logic-puzzles",
                  protocol=PROTOCOL, solver_order_used=False, strategy_tokens_used=False,
                  split_policy="official heldout test; validation carved from official train",
                  integrity="source hashes; all test puzzle identities; selected solutions validated",
                  source_files={s: dict(path=str(Path(p).resolve()), sha256=_sha256(p),
                                        records=len(arrays[s])) for s, p in (("train", train_file), ("test", test_file))},
                  source_indices={s: [r["metadata"]["source_index"] for r in rs] for s, rs in splits.items()})
    return _publish(output, "sudoku-benchmark", splits, seed, source)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)
    zebra = sub.add_parser("zebra")
    zebra.add_argument("--source-dir", required=True)
    zebra.add_argument("--output", required=True)
    sudoku = sub.add_parser("sudoku")
    for field in ("train-file", "test-file", "output"):
        sudoku.add_argument("--" + field, required=True)
    for field, default in (("train-size", 20000), ("valid-size", 1000), ("test-size", 1000), ("seed", 17)):
        sudoku.add_argument("--" + field, type=int, default=default)
    args = vars(parser.parse_args(argv))
    task = args.pop("task")
    result = migrate_zebra(**args) if task == "zebra" else import_sudoku(**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
