"""Small, explicitly synthetic reasoning pilots and strict solution validators.

These generators are NOT the Latent Tokens paper's released datasets.  They
provide offline correctness tests while normalized benchmark records are sought.
All prompts and solutions use a fixed, task-specific symbolic vocabulary.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import random
import re


SPECIAL_TOKENS = ("[PAD]", "[MASK]", "[BOS]", "[SEP]", "[EOS]")
TASK_LENGTHS = {"sudoku": 192, "zebra": 384, "countdown": 64}
ANSWER_SLOTS = {"sudoku": 82, "zebra": 26, "countdown": 48}


def normalize_task(task):
    task = str(task).lower().replace("_", "-")
    task = {"sudoku-puzzle": "sudoku"}.get(task, task)
    if task not in TASK_LENGTHS:
        raise ValueError("Unknown reasoning task: %s" % task)
    return task


def task_answer_slots(task):
    """Fixed answer width, including EOS and any supervised PAD tokens."""
    return ANSWER_SLOTS[normalize_task(task)]


class TaskTokenizer:
    """No fitted vocabulary, external tokenizer, or unknown-token substitution."""

    def __init__(self, task):
        self.task = normalize_task(task)
        if self.task == "sudoku":
            symbols = list("0123456789")
        elif self.task == "countdown":
            symbols = list("0123456789+-*/=,")
        else:
            symbols = (list("12345") + ["C%d" % i for i in range(5)]
                       + ["V%d" % i for i in range(5)]
                       + ["AT", "SAME", "LEFT", "NEXT", ";"])
        self.tokens = list(SPECIAL_TOKENS) + symbols
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.pad_id, self.mask_id, self.bos_id, self.sep_id, self.eos_id = range(5)
        self.special_ids = frozenset(range(5))
        self.vocab_size = len(self.tokens)

    def encode(self, tokens):
        if not isinstance(tokens, (list, tuple)) or not all(isinstance(t, str) for t in tokens):
            raise ValueError("encode expects a list of symbolic token strings")
        try:
            return [self.token_to_id[token] for token in tokens]
        except KeyError as error:
            raise ValueError("Token outside %s vocabulary: %s" % (self.task, error.args[0])) from error

    def decode(self, ids):
        result = []
        for index in ids:
            index = int(index)
            if not 0 <= index < self.vocab_size:
                raise ValueError("Token id outside vocabulary: %s" % index)
            result.append(self.tokens[index])
        return result


def _strip_answer(tokens):
    """EOS is required only for generated slot outputs; raw answers omit it.

    A non-PAD token after EOS or interior special token makes a prediction invalid.
    Trailing PAD without EOS is tolerated for scoring mathematical task validity;
    evaluation reports sequence-token accuracy separately.
    """
    tokens = list(tokens)
    if "[EOS]" in tokens:
        end = tokens.index("[EOS]")
        if any(t != "[PAD]" for t in tokens[end + 1:]):
            return None
        tokens = tokens[:end]
    else:
        while tokens and tokens[-1] == "[PAD]":
            tokens.pop()
    if any(t in SPECIAL_TOKENS for t in tokens):
        return None
    return tokens


def _sudoku_solutions(board, limit=2, rng=None):
    board = list(board)
    answers = []
    rows, cols, boxes = [set() for _ in range(9)], [set() for _ in range(9)], [set() for _ in range(9)]
    for i, value in enumerate(board):
        if not value:
            continue
        r, c, b = i // 9, i % 9, (i // 27) * 3 + (i % 9) // 3
        if value not in range(1, 10) or value in rows[r] or value in cols[c] or value in boxes[b]:
            return []
        rows[r].add(value); cols[c].add(value); boxes[b].add(value)

    def search():
        if len(answers) >= limit:
            return
        best, choices = None, None
        for i, value in enumerate(board):
            if value:
                continue
            r, c, b = i // 9, i % 9, (i // 27) * 3 + (i % 9) // 3
            candidates = set(range(1, 10)) - rows[r] - cols[c] - boxes[b]
            if not candidates:
                return
            if choices is None or len(candidates) < len(choices):
                best, choices = i, candidates
                if len(choices) == 1:
                    break
        if best is None:
            answers.append(list(board)); return
        r, c, b = best // 9, best % 9, (best // 27) * 3 + (best % 9) // 3
        choices = sorted(choices)
        if rng is not None:
            rng.shuffle(choices)
        for value in choices:
            board[best] = value
            rows[r].add(value); cols[c].add(value); boxes[b].add(value)
            search()
            rows[r].remove(value); cols[c].remove(value); boxes[b].remove(value)
            board[best] = 0
            if len(answers) >= limit:
                return
    search()
    return answers


def _generate_sudoku(rng):
    # Randomized solver, not just symmetries of one canonical completed grid.
    solution = _sudoku_solutions([0] * 81, limit=1, rng=rng)[0]
    puzzle = list(solution)
    positions = list(range(81)); rng.shuffle(positions)
    wanted = rng.randint(35, 45)
    removed = 0
    for i in positions:
        old = puzzle[i]; puzzle[i] = 0
        if len(_sudoku_solutions(puzzle, limit=2)) != 1:
            puzzle[i] = old
        else:
            removed += 1
        if removed >= wanted:
            break
    return {"task": "sudoku", "prompt": [str(x) for x in puzzle],
            "answer": [str(x) for x in solution],
            "metadata": {"generator": "randomized_solver_unique_v1", "unique_solution": True,
                         "clues": 81 - removed}}


def _zebra_relation(kind, left, right):
    return {"SAME": left == right, "LEFT": left < right,
            "NEXT": abs(left - right) == 1}[kind]


def _parse_zebra_prompt(tokens):
    constraints = []
    chunk = []
    for token in tokens:
        if token != ";":
            chunk.append(token); continue
        if len(chunk) == 4 and chunk[0] == "AT":
            kind, cat, val, position = chunk
            if not re.fullmatch(r"C[0-4]", cat) or not re.fullmatch(r"V[0-4]", val) or position not in list("12345"):
                raise ValueError("Malformed Zebra AT clue")
            constraints.append((kind, int(cat[1]) * 5 + int(val[1]), int(position)))
        elif len(chunk) == 5 and chunk[0] in ("SAME", "LEFT", "NEXT"):
            kind, ca, va, cb, vb = chunk
            if any(not re.fullmatch(r"C[0-4]", c) for c in (ca, cb)) or any(not re.fullmatch(r"V[0-4]", v) for v in (va, vb)):
                raise ValueError("Malformed Zebra relational clue")
            constraints.append((kind, int(ca[1]) * 5 + int(va[1]), int(cb[1]) * 5 + int(vb[1])))
        else:
            raise ValueError("Malformed Zebra clue")
        chunk = []
    if chunk or not constraints:
        raise ValueError("Zebra clues must be nonempty and end in ;")
    return constraints


def _zebra_solutions(constraints, limit=2):
    answers = []

    def propagate(domains):
        changed = True
        while changed:
            changed = False
            for category in range(5):
                indices = list(range(5 * category, 5 * category + 5))
                fixed = [next(iter(domains[i])) for i in indices if len(domains[i]) == 1]
                if len(fixed) != len(set(fixed)):
                    return False
                for i in indices:
                    if len(domains[i]) > 1:
                        new = domains[i] - set(fixed)
                        if new != domains[i]: domains[i] = new; changed = True
            for kind, a, b in constraints:
                if kind == "AT":
                    new_a = domains[a] & {b}
                    if new_a != domains[a]: domains[a] = new_a; changed = True
                else:
                    new_a = {x for x in domains[a] if any(_zebra_relation(kind, x, y) for y in domains[b])}
                    new_b = {y for y in domains[b] if any(_zebra_relation(kind, x, y) for x in new_a)}
                    if new_a != domains[a]: domains[a] = new_a; changed = True
                    if new_b != domains[b]: domains[b] = new_b; changed = True
            if any(not domain for domain in domains):
                return False
        return True

    def search(domains):
        if len(answers) >= limit or not propagate(domains):
            return
        unresolved = [i for i, values in enumerate(domains) if len(values) > 1]
        if not unresolved:
            answers.append([next(iter(values)) for values in domains]); return
        i = min(unresolved, key=lambda j: len(domains[j]))
        for value in sorted(domains[i]):
            next_domains = [set(values) for values in domains]
            next_domains[i] = {value}; search(next_domains)
            if len(answers) >= limit:
                return
    search([set(range(1, 6)) for _ in range(25)])
    return answers


def _serialize_zebra(constraints):
    tokens = []
    for kind, a, b in constraints:
        tokens.extend([kind, "C%d" % (a // 5), "V%d" % (a % 5)])
        if kind == "AT":
            tokens.append(str(b))
        else:
            tokens.extend(["C%d" % (b // 5), "V%d" % (b % 5)])
        tokens.append(";")
    return tokens


def _generate_zebra(rng):
    positions = []
    for _ in range(5):
        row = list(range(1, 6)); rng.shuffle(row); positions.extend(row)
    constraints = [("AT", i, positions[i]) for i in range(5)]
    # Connect each new category to the preceding category, allowing multi-hop clues.
    for category in range(1, 5):
        for value in range(5):
            a = category * 5 + value
            b = (category - 1) * 5 + positions[(category - 1) * 5: category * 5].index(positions[a])
            constraints.append(("SAME", a, b))
    for _ in range(12):
        a, b = rng.sample(range(25), 2)
        if positions[a] < positions[b]: constraints.append(("LEFT", a, b))
        elif abs(positions[a] - positions[b]) == 1: constraints.append(("NEXT", a, b))
    # Prune selected direct clues only when relational constraints still imply uniqueness.
    order = list(range(len(constraints))); rng.shuffle(order)
    for index in sorted(order[:10], reverse=True):
        candidate = constraints[:index] + constraints[index + 1:]
        if len(_zebra_solutions(candidate)) == 1:
            constraints = candidate
    rng.shuffle(constraints)
    return {"task": "zebra", "prompt": _serialize_zebra(constraints),
            "answer": [str(x) for x in positions],
            "metadata": {"generator": "five_category_relational_unique_v1", "unique_solution": True,
                         "answer_layout": "category-major; each value maps to its house position"}}


def _countdown_inputs(prompt):
    string = "".join(prompt)
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+){4}=[0-9]+", string):
        raise ValueError("Countdown prompt must be five comma-separated nonnegative integers = target")
    numbers, target = string.split("=")
    return [int(value) for value in numbers.split(",")], int(target)


def _countdown_chain(tokens, numbers, target):
    remaining = Counter(Fraction(value) for value in numbers)
    steps = "".join(tokens).split(",")
    if len(steps) != len(numbers) - 1:
        return False
    for step in steps:
        match = re.fullmatch(r"([0-9]+)([+*/-])([0-9]+)=([0-9]+)", step)
        if not match:
            return False
        a, op, b, c = match.groups()
        a, b, c = Fraction(int(a)), Fraction(int(b)), Fraction(int(c))
        for operand in (a, b):
            if remaining[operand] <= 0:
                return False
            remaining[operand] -= 1
        if op == "/" and b == 0:
            return False
        value = {"+": lambda: a + b, "-": lambda: a - b,
                 "*": lambda: a * b, "/": lambda: a / b}[op]()
        if value != c:
            return False
        remaining[c] += 1
    return sum(remaining.values()) == 1 and remaining[Fraction(target)] == 1


def _generate_countdown(rng):
    numbers = [rng.randint(1, 9) for _ in range(5)]
    pool = list(numbers)
    steps = []
    while len(pool) > 1:
        ai, bi = sorted(rng.sample(range(len(pool)), 2), reverse=True)
        a, b = pool.pop(ai), pool.pop(bi)
        choices = [("+", a + b)]
        if a >= b: choices.append(("-", a - b))
        if a * b <= 99: choices.append(("*", a * b))
        if b and a % b == 0: choices.append(("/", a // b))
        op, value = rng.choice(choices)
        steps.append("%d%s%d=%d" % (a, op, b, value)); pool.append(value)
    return {"task": "countdown", "prompt": list(",".join(map(str, numbers)) + "=" + str(pool[0])),
            "answer": list(",".join(steps)),
            "metadata": {"generator": "five_single_digit_integer_chain_v1", "numbers": numbers,
                         "target": pool[0], "domain": "nonnegative integer intermediate results"}}


def score_prediction(record, predicted_answer_tokens):
    """Score by task rules; never require the reference chain/grid if alternatives work."""
    task = normalize_task(record["task"])
    answer = _strip_answer(predicted_answer_tokens)
    exact = answer is not None and answer == record["answer"]
    result = {"exact_match": bool(exact), "valid_solution": False}
    if task == "sudoku":
        result.update(clues_preserved=False, constraints_satisfied=False)
    elif task == "zebra":
        result.update(constraints_satisfied=False)
    if answer is None:
        return result
    if task == "sudoku":
        shaped = len(answer) == 81 and all(x in list("123456789") for x in answer)
        if not shaped:
            result.update(clues_preserved=False, constraints_satisfied=False); return result
        board = [int(x) for x in answer]
        clues = [int(x) for x in record["prompt"]]
        kept = all(not clue or board[i] == clue for i, clue in enumerate(clues))
        valid = bool(_sudoku_solutions(board, limit=1))
        result.update(clues_preserved=kept, constraints_satisfied=valid, valid_solution=kept and valid)
    elif task == "zebra":
        if len(answer) != 25 or any(x not in list("12345") for x in answer):
            return result
        positions = [int(x) for x in answer]
        permutations = all(set(positions[i:i + 5]) == set(range(1, 6)) for i in range(0, 25, 5))
        clues = _parse_zebra_prompt(record["prompt"])
        satisfied = all(positions[a] == b if kind == "AT" else _zebra_relation(kind, positions[a], positions[b])
                        for kind, a, b in clues)
        result.update(constraints_satisfied=satisfied, valid_solution=permutations and satisfied)
    else:
        numbers, target = _countdown_inputs(record["prompt"])
        result["valid_solution"] = _countdown_chain(answer, numbers, target)
    return result


def task_identity(record):
    """Logical input key; changing clue/operand serialization must not leak splits."""
    task = normalize_task(record["task"])
    if task == "countdown":
        numbers, target = _countdown_inputs(record["prompt"])
        content = [sorted(numbers), target]
    elif task == "zebra":
        # SAME/NEXT are symmetric; LEFT remains directed. Duplicate clauses are
        # semantically redundant and clue order must not create a new test task.
        clues = []
        for kind, a, b in _parse_zebra_prompt(record["prompt"]):
            if kind in ("SAME", "NEXT"):
                a, b = sorted((a, b))
            clues.append((kind, a, b))
        content = sorted(set(clues))
    else:
        content = record["prompt"]
    return json.dumps([task, content], separators=(",", ":"))


def validate_record(record, task=None):
    if not isinstance(record, dict) or not all(key in record for key in ("task", "prompt", "answer")):
        raise ValueError("Each record needs task, prompt and answer")
    actual = normalize_task(record["task"])
    if task is not None and actual != normalize_task(task):
        raise ValueError("Record task does not match dataset task")
    tokenizer = TaskTokenizer(actual)
    for field in ("prompt", "answer"):
        tokenizer.encode(record[field])
        if not record[field] or any(token in SPECIAL_TOKENS for token in record[field]):
            raise ValueError("Raw prompt/answer must be nonempty and contain no special tokens")
    if len(record["answer"]) >= task_answer_slots(actual):
        raise ValueError("Answer exceeds fixed target slots; never truncate")
    if len(record["prompt"]) + task_answer_slots(actual) + 2 > TASK_LENGTHS[actual]:
        raise ValueError("Prompt plus fixed answer slots exceeds task sequence length")
    if actual == "sudoku" and (len(record["prompt"]) != 81 or any(t not in list("0123456789") for t in record["prompt"])):
        raise ValueError("Sudoku prompt must have exactly 81 digit cells")
    if not isinstance(record.get("metadata", {}), dict):
        raise ValueError("metadata must be an object")
    if not score_prediction(record, record["answer"])["valid_solution"]:
        raise ValueError("Reference answer does not solve the supplied task")
    normalized = {"task": actual, "prompt": list(record["prompt"]), "answer": list(record["answer"]),
                  "metadata": dict(record.get("metadata", {}))}
    signature = hashlib.sha256(task_identity(normalized).encode()).hexdigest()
    normalized["id"] = str(record.get("id", signature))
    if not normalized["id"]:
        raise ValueError("Record ID must not be empty")
    return normalized


def generate_record(task, rng):
    task = normalize_task(task)
    generator = {"sudoku": _generate_sudoku, "zebra": _generate_zebra, "countdown": _generate_countdown}[task]
    # A rare long Countdown chain can exceed 64-token layout; resample, never truncate.
    for _ in range(100):
        record = generator(rng)
        if len(record["answer"]) < ANSWER_SLOTS[task] and len(record["prompt"]) + ANSWER_SLOTS[task] + 2 <= TASK_LENGTHS[task]:
            return validate_record(record)
    raise RuntimeError("Could not generate a record fitting the declared task layout")
