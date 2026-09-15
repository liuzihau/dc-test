"""Immutable, checksummed task data preparation and fixed-layout tensor batches."""
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset

from .tasks import (TASK_LENGTHS, TaskTokenizer, generate_record, normalize_task,
                    task_answer_slots, task_identity, validate_record)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path, task):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(validate_record(json.loads(line), task))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError("%s:%d: %s" % (path, line_number, error)) from error
    return records


def _identity(record):
    # Prompt identity, not answer identity; alternate valid solutions cannot leak tasks.
    return task_identity(record)


def prepare_dataset(output_dir, task, train_size=256, valid_size=64, test_size=64,
                    seed=1, input_path=None):
    """Prepare pilot data or normalized JSONL; refuse to overwrite any output.

    input_path may be one JSONL (deterministically shuffled then split) or a dict
    mapping train/validation/test to JSONL paths.  Explicit splits are preserved
    in their entirety; their sizes are not overridden by the synthetic defaults.
    Imported data is not automatically attributed to an external benchmark.
    """
    task = normalize_task(task)
    output_dir = Path(output_dir)
    sizes = {"train": int(train_size), "validation": int(valid_size), "test": int(test_size)}
    if any(size < 0 for size in sizes.values()) or sizes["train"] == 0:
        raise ValueError("Training size must be positive and evaluation sizes nonnegative")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite nonempty dataset directory: %s" % output_dir)
    rng = random.Random(int(seed))
    source = {"kind": "synthetic_pilot", "benchmark_equivalence": False,
              "description": "Locally generated correctness/pilot tasks; not original paper data"}
    if isinstance(input_path, dict):
        paths = {("validation" if key == "valid" else key): value for key, value in input_path.items()}
        if set(paths) != {"train", "validation", "test"}:
            raise ValueError("Explicit source splits require train, validation, test")
        splits = {split: _read_jsonl(path, task) for split, path in paths.items()}
        sizes = {split: len(records) for split, records in splits.items()}
        if not sizes["train"]:
            raise ValueError("Imported train split must not be empty")
        source = {"kind": "normalized_jsonl_import", "benchmark_equivalence": False,
                  "split_policy": "preserved_explicit_splits",
                  "files": {split: {"path": str(Path(path).resolve()), "sha256": _sha256(path)} for split, path in paths.items()}}
    else:
        needed = sum(sizes.values())
        if input_path is None:
            records = []
            seen = set()
            for _ in range(max(100, 20 * needed)):
                record = generate_record(task, rng)
                if _identity(record) not in seen:
                    records.append(record); seen.add(_identity(record))
                if len(records) == needed:
                    break
            if len(records) != needed:
                raise RuntimeError("Failed to generate enough distinct tasks")
        else:
            records = _read_jsonl(input_path, task)
            if len(records) < needed:
                raise ValueError("Input has %d records but requested %d" % (len(records), needed))
            # Reject leakage even in the portion that would otherwise be discarded.
            if len({_identity(r) for r in records}) != len(records) or len({r["id"] for r in records}) != len(records):
                raise ValueError("Duplicate task prompts or IDs in imported JSONL")
            rng.shuffle(records)
            records = records[:needed]
            source = {"kind": "normalized_jsonl_import", "benchmark_equivalence": False,
                      "split_policy": "seeded_shuffle_then_requested_sizes",
                      "path": str(Path(input_path).resolve()), "sha256": _sha256(input_path)}
        splits = {}
        offset = 0
        for split, size in sizes.items():
            splits[split] = records[offset:offset + size]; offset += size
    all_records = [record for records in splits.values() for record in records]
    if len({_identity(r) for r in all_records}) != len(all_records) or len({r["id"] for r in all_records}) != len(all_records):
        raise ValueError("Duplicate task prompts or IDs within/across dataset splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, records in splits.items():
        path = output_dir / (split + ".jsonl")
        with path.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        files[split] = {"filename": path.name, "sha256": _sha256(path), "records": len(records)}
    tokenizer = TaskTokenizer(task)
    manifest = {"schema_version": 1, "task": task, "seed": int(seed), "max_length": TASK_LENGTHS[task],
                "answer_slots": task_answer_slots(task), "vocab": tokenizer.tokens,
                "target_layout": "fixed answer slots include supervised EOS/PAD; outer padding is excluded",
                "source": source, "splits": files}
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True); stream.write("\n")
    return manifest


class ReasoningDataset(Dataset):
    def __init__(self, directory, split="train", verify_checksum=True):
        directory = Path(directory)
        split = "validation" if split == "valid" else split
        with (directory / "manifest.json").open(encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported reasoning dataset schema")
        self.task = normalize_task(self.manifest["task"])
        self.tokenizer = TaskTokenizer(self.task)
        self.max_length = TASK_LENGTHS[self.task]
        if self.manifest["max_length"] != self.max_length or self.manifest["answer_slots"] != task_answer_slots(self.task) or self.manifest["vocab"] != self.tokenizer.tokens:
            raise ValueError("Dataset tokenizer/layout does not match current schema")
        entry = self.manifest["splits"][split]
        if Path(entry["filename"]).name != entry["filename"]:
            raise ValueError("Split filename must remain inside the dataset directory")
        path = directory / entry["filename"]
        if verify_checksum and _sha256(path) != entry["sha256"]:
            raise ValueError("Dataset checksum mismatch: %s" % path)
        self.records = _read_jsonl(path, self.task)
        if len(self.records) != entry["records"]:
            raise ValueError("Dataset record count does not match manifest")
        if len({_identity(r) for r in self.records}) != len(self.records):
            raise ValueError("Repeated prompt in dataset split")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        tok = self.tokenizer
        prefix = [tok.bos_id] + tok.encode(record["prompt"]) + [tok.sep_id]
        answer = tok.encode(record["answer"]) + [tok.eos_id]
        slots = task_answer_slots(self.task)
        answer += [tok.pad_id] * (slots - len(answer))
        clean = prefix + answer
        used = len(clean)
        ids = torch.tensor(clean + [tok.pad_id] * (self.max_length - used), dtype=torch.long)
        attention = torch.arange(self.max_length) < used
        targets = (torch.arange(self.max_length) >= len(prefix)) & attention
        return {"input_ids": ids, "attention_mask": attention, "target_mask": targets,
                "record_index": torch.tensor(index, dtype=torch.long)}
