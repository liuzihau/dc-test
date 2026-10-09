"""Source-data-aligned Shah Zebra puzzles; NOT an exact Latent Tokens reproduction.

The released data contains an answer trace inside the token sequence and a
NumPy solver-order array. Neither is a model input. Only symbolic clues before
ANSWER and public puzzle dimensions become the prompt. Output is the fixed
attribute-major table, not the source solver's adaptive answer order.
"""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import pickle
import random
import tempfile


SOURCE_URL = "https://github.com/kulinshah98/llm-reasoning-logic-puzzles"
SOURCE_COMMIT = "66a49af440cf52eef087793727f6bd41b8c19eb2"
DRIVE_FILES = {
    "train": "1UHoalXvhDtSYm3D3Dgm6YQgcTQKT7Ahf",
    "test": "1LCzeK8p5Z48AlOVn5BtPkj2Tqgvui7yH",
}
RELATIONS = ("=", "!=", "immediate-left", "nbr", "ends", "left-of", "inbetween")
SYMBOLS = tuple("0123456") + RELATIONS + ("LHS", "RHS", "c", "n", "CLUE_END", "HOUSES", "ATTRS")


class _DiscardedDtype:
    """Inert stand-in, not a call to numpy.dtype from an untrusted pickle."""

    def __init__(self, code, align=False, copy=True):
        if (code, align, copy) != ("i2", False, True):
            raise ValueError("Only the released int16 solver trace is supported")
        self.ready = False

    def __setstate__(self, state):
        if state != (3, "<", None, None, None, -1, -1, 0):
            raise ValueError("Unexpected solver-trace dtype state")
        self.ready = True


class _DiscardedTrace:
    """Validate a tiny numerical trace then discard bytes; never execute NumPy."""

    def __setstate__(self, state):
        if not isinstance(state, tuple) or len(state) != 5:
            raise ValueError("Unexpected solver-trace array state")
        version, shape, dtype, fortran, payload = state
        if (version != 1 or not isinstance(shape, tuple) or len(shape) != 2
                or any(type(x) is not int for x in shape)
                or not 9 <= shape[0] <= 36 or shape[1] != 2
                or type(dtype) is not _DiscardedDtype or not dtype.ready
                or fortran is not False or type(payload) is not bytes
                or len(payload) != 2 * shape[0] * shape[1]):
            raise ValueError("Invalid released solver trace")
        self.shape = shape


def _discarded_reconstruct(subtype, shape, dtype):
    if subtype is not _DiscardedTrace or shape != (0,) or dtype != b"b":
        raise ValueError("Unexpected solver-trace reconstruction request")
    return _DiscardedTrace()


class RestrictedZebraUnpickler(pickle.Unpickler):
    """Allow plain containers plus inert stand-ins for the observed trace format.

    No imported classes/functions can be called. This is not an unrestricted
    pickle loader or a general NumPy unpickler. Malformed/unrecognized revisions
    fail closed. As with other data parsers, use normal resource limits when
    inspecting arbitrary files (this does not protect against resource DoS).
    """

    def find_class(self, module, name):
        allowed = {
            ("numpy", "dtype"): _DiscardedDtype,
            ("numpy", "ndarray"): _DiscardedTrace,
            ("numpy.core.multiarray", "_reconstruct"): _discarded_reconstruct,
        }
        try:
            return allowed[module, name]
        except KeyError as error:
            raise ValueError("Forbidden pickle global: %s.%s" % (module, name)) from error

    def persistent_load(self, pid):
        raise ValueError("Persistent pickle references are forbidden")


def load_source(path):
    """Read the source release without importing/executing serialized objects."""
    path = Path(path)
    if path.stat().st_size > 2 * 1024 ** 3:
        raise ValueError("Source exceeds the 2 GiB release-size safety limit")
    with path.open("rb") as stream:
        records = RestrictedZebraUnpickler(stream).load()
        if stream.read(1):
            raise ValueError("Unexpected trailing content after source pickle")
    if type(records) is not list or not records:
        raise ValueError("Source must be a nonempty list of records")
    return records


def _integer(token, upper):
    if type(token) is not str or token not in tuple(str(x) for x in range(upper)):
        raise ValueError("Invalid zero-based Zebra index: %r" % token)
    return int(token)


def _reference(tokens, houses, attributes):
    if len(tokens) != 3 or tokens[0] not in ("c", "n"):
        raise ValueError("Malformed Zebra reference")
    kind, category, value = tokens
    category = _integer(category, attributes if kind == "c" else 1)
    value = _integer(value, houses)
    return kind, category, value


def parse_prompt(tokens):
    """Parse all seven released relations and strictly bounded public dimensions."""
    if (not isinstance(tokens, (tuple, list)) or len(tokens) < 5
            or tokens[0] != "HOUSES" or tokens[2] != "ATTRS"
            or tokens[1] not in ("3", "4", "5", "6")
            or tokens[3] not in ("3", "4", "5", "6")):
        raise ValueError("Official Zebra requires HOUSES/ATTRS dimensions")
    houses, attributes = int(tokens[1]), int(tokens[3])
    clues, chunk = [], []
    for token in tokens[4:]:
        if token != "CLUE_END":
            chunk.append(token)
            continue
        if (len(chunk) < 6 or chunk[0] not in RELATIONS
                or chunk[1] != "LHS" or chunk[5] != "RHS"):
            raise ValueError("Malformed official Zebra clue")
        kind = chunk[0]
        arity = 1 if kind == "ends" else 3 if kind == "inbetween" else 2
        if len(chunk) != 6 + 3 * (arity - 1):
            raise ValueError("Wrong arity for Zebra relation %s" % kind)
        refs = [_reference(chunk[2:5], houses, attributes)]
        refs.extend(_reference(chunk[i:i + 3], houses, attributes)
                    for i in range(6, len(chunk), 3))
        clues.append((kind, tuple(refs)))
        chunk = []
    if chunk or not clues:
        raise ValueError("Official clues must be nonempty and end in CLUE_END")
    return houses, attributes, clues


def relation_holds(kind, positions, houses):
    left = positions[0]
    if kind == "ends":
        return left in (0, houses - 1)
    right = positions[1]
    if kind == "=":
        return left == right
    if kind == "!=":
        return left != right
    if kind == "immediate-left":
        return left + 1 == right
    if kind == "nbr":
        return abs(left - right) == 1
    if kind == "left-of":
        return left < right
    if kind == "inbetween":
        # Released serialization convention: A LHS, then B,C RHS means
        # A < B < C (B is the middle item, NOT A). This ordered convention was
        # checked against all 139,082 such clues in the 100k official test file.
        # Published evaluation compares the complete reference table; we also
        # require reference equality for the primary valid_solution metric.
        return left < right < positions[2]
    raise ValueError("Unknown Zebra relation")


def score_answer(prompt, answer):
    houses, attributes, clues = parse_prompt(prompt)
    result = dict(constraints_satisfied=False, valid_solution=False)
    if (answer is None or len(answer) != houses * attributes
            or any(type(x) is not str or x not in tuple(map(str, range(houses))) for x in answer)):
        return result
    rows = [list(map(int, answer[i:i + houses])) for i in range(0, len(answer), houses)]
    if any(set(row) != set(range(houses)) for row in rows):
        return result
    positions = [{value: house for house, value in enumerate(row)} for row in rows]
    satisfied = all(relation_holds(kind, [v if k == "n" else positions[c][v]
                                         for k, c, v in refs], houses)
                    for kind, refs in clues)
    result.update(constraints_satisfied=satisfied, valid_solution=satisfied)
    return result


def logical_identity(prompt):
    houses, attributes, clues = parse_prompt(prompt)
    normalized = []
    for kind, refs in clues:
        if kind in ("=", "!=", "nbr"):
            refs = tuple(sorted(refs))
        normalized.append((kind, refs))
    # This canonicalizes duplicate clauses and symmetric/clue-order rewrites,
    # not arbitrary logically equivalent clue sets (an exponential task).
    return houses, attributes, sorted(set(normalized))


def convert_source_record(raw, source_split, source_index):
    """Discard both solver order and answer-trace tokens; verify the clean table."""
    if type(raw) is not list or len(raw) != 3:
        raise ValueError("Expected a three-part official Zebra record")
    sequence, table, trace = raw
    if (type(sequence) is not list or not all(type(x) is str for x in sequence)
            or sequence.count("ANSWER") != 1):
        raise ValueError("Source sequence must have exactly one ANSWER boundary")
    if (type(table) is not list or not 4 <= len(table) <= 7
            or any(type(row) is not list for row in table)):
        raise ValueError("Invalid source solution table")
    houses, attributes = len(table[0]), len(table) - 1
    if (not 3 <= houses <= 6 or table[0] != list(range(houses))
            or any(len(row) != houses or any(type(x) is not int for x in row)
                   or set(row) != set(range(houses)) for row in table)):
        raise ValueError("Source table must contain zero-based house permutations")
    if type(trace) is not _DiscardedTrace or getattr(trace, "shape", None) != (houses * attributes, 2):
        raise ValueError("Source solver trace shape differs from public puzzle size")
    prompt = ["HOUSES", str(houses), "ATTRS", str(attributes)] + sequence[:sequence.index("ANSWER")]
    answer = [str(x) for row in table[1:] for x in row]
    if not score_answer(prompt, answer)["valid_solution"]:
        raise ValueError("Source answer violates its clues")
    return dict(task="zebra-official", prompt=prompt, answer=answer,
                metadata=dict(source=SOURCE_URL, source_split=source_split,
                              source_index=int(source_index), houses=houses, attributes=attributes,
                              answer_layout="attribute-major values per house; zero based",
                              source_answer_trace_used=False, solver_order_used=False))


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def import_dataset(train_path, test_path, output_dir, *, train_size=20000,
                   valid_size=1000, test_size=1000, seed=17):
    """Use held-out source test only for test; validation is carved from train.

    Every source record is validated. Seeded reservoir samples are chosen before
    any model results are observed. All official test identities are excluded
    from both train/validation, including test examples not selected for eval.
    """
    from .data import write_prepared_dataset
    from .tasks import TASK_LENGTHS, validate_record

    sizes = dict(train=int(train_size), validation=int(valid_size), test=int(test_size))
    if any(x <= 0 for x in sizes.values()):
        raise ValueError("All three official subset sizes must be positive")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing to overwrite prepared data")
    rng = random.Random(int(seed))
    pools, seen, source_info = {}, set(), {}
    for split, path, count in (("test", test_path, sizes["test"]),
                               ("train", train_path, sizes["train"] + sizes["validation"])):
        raw_records = load_source(path)
        source_info[split] = dict(path=str(Path(path).resolve()), sha256=_digest(path),
                                 bytes=Path(path).stat().st_size, records=len(raw_records),
                                 drive_id=DRIVE_FILES[split])
        sample, eligible, duplicates, histogram = [], 0, 0, Counter()
        max_raw_prompt_tokens = max_packed_tokens = 0
        for index, raw in enumerate(raw_records):
            try:
                record = validate_record(convert_source_record(raw, split, index), "zebra-official")
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError("%s source record %d: %s" % (split, index, error)) from error
            max_raw_prompt_tokens = max(max_raw_prompt_tokens, len(record["prompt"]) - 4)
            max_packed_tokens = max(max_packed_tokens, len(record["prompt"]) + 37 + 2)
            identity = record["id"]
            if identity in seen:
                duplicates += 1
                continue
            seen.add(identity)
            histogram["%dx%d" % (record["metadata"]["houses"], record["metadata"]["attributes"])] += 1
            eligible += 1
            if len(sample) < count:
                sample.append(record)
            else:
                place = rng.randrange(eligible)
                if place < count:
                    sample[place] = record
            if (index + 1) % 100000 == 0:
                print("Validated %s source records: %d/%d" %
                      (split, index + 1, len(raw_records)), flush=True)
        del raw_records
        if eligible < count:
            raise ValueError("Not enough distinct %s records: %d < %d" % (split, eligible, count))
        rng.shuffle(sample)
        pools[split] = sample
        source_info[split].update(validated_records=source_info[split]["records"],
                                 distinct_eligible_records=eligible, duplicates_excluded=duplicates,
                                 eligible_size_histogram=dict(histogram),
                                 max_raw_prompt_tokens=max_raw_prompt_tokens,
                                 max_packed_tokens=max_packed_tokens)
    splits = dict(train=pools["train"][:sizes["train"]],
                  validation=pools["train"][sizes["train"]:], test=pools["test"])
    source = dict(kind="shah_official_zebra_subset", benchmark_equivalence=False,
                  description="Source-data-aligned; NOT exact reproduction of Latent Tokens 96.9%",
                  repository=SOURCE_URL, inspected_source_commit=SOURCE_COMMIT,
                  split_policy="official test reserved; seeded reservoir train/validation from official train",
                  duplicate_policy="exclude canonical clue identities globally, test source takes priority",
                  dimensions="all released 3..6 houses and 3..6 attributes; unfiltered",
                  source_files=source_info,
                  selected_size_histograms={s: dict(Counter("%dx%d" % (r["metadata"]["houses"], r["metadata"]["attributes"])
                                                           for r in rs)) for s, rs in splits.items()},
                  source_indices={s: [r["metadata"]["source_index"] for r in rs] for s, rs in splits.items()},
                  solver_trace_in_prompt=False, source_answer_trace_in_prompt=False)
    source["primary_metric"] = "complete reference-table equality with structural/clue checks"
    source["inbetween_convention"] = "serialized A,B,C satisfy A < B < C; verified on released source answers"
    source["layout"] = dict(max_length=TASK_LENGTHS["zebra-official"], answer_slots=37,
                            differs_from_latent_tokens_paper_384=True,
                            description="400 accommodates all released records; no length filtering or truncation")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["population", "split", "houses", "attributes", "records"])
    for split, stats in source_info.items():
        for dimensions, count in sorted(stats["eligible_size_histogram"].items()):
            writer.writerow(["source_distinct_eligible", split, *dimensions.split("x"), count])
    for split, histogram in source["selected_size_histograms"].items():
        for dimensions, count in sorted(histogram.items()):
            writer.writerow(["selected", split, *dimensions.split("x"), count])
    statistics = buffer.getvalue()
    source["statistics_file"] = dict(filename="source_statistics.csv",
                                     sha256=hashlib.sha256(statistics.encode()).hexdigest())
    # Cloud shutdown must not publish a half-written dataset. Keep unfinished
    # siblings recoverable; do not overwrite/delete them on a later retry.
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix="." + out.name + ".import-", dir=out.parent))
    try:
        manifest = write_prepared_dataset(staged, "zebra-official", splits, seed, source)
        with (staged / "source_statistics.csv").open("x", encoding="utf-8", newline="") as stream:
            stream.write(statistics)
        # A sibling is on the same filesystem. Existing nonempty directories
        # are never replaceable by rename, including a concurrent publication.
        staged.replace(out)
    except BaseException:
        print("Import was not published; recoverable staged files: %s" % staged, flush=True)
        raise
    return manifest
