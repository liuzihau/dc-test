"""Baseline wrapper contracts, with no GPUs, tmux server or training jobs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "train_reasoning_mdm_aux_h100.sh"


@pytest.fixture
def launch(tmp_path):
    target = tmp_path / "scripts/train"
    target.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/train" / SCRIPT, target / SCRIPT)
    recorder = tmp_path / "record.py"
    recorder.write_text("import json, os, sys\nprint(json.dumps(dict(args=sys.argv[1:], env=dict(os.environ))))\n")
    # The wrapper must delegate all execution to the existing shared launcher.
    (target / "train_reasoning.sh").write_text(
        '#!/bin/bash\nexec "$DCACHE_PYTHON" "$BASELINE_TEST_RECORDER" "$@"\n')
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("REASONING_", "DCACHE_"))}
    env.update(DCACHE_PYTHON=sys.executable, BASELINE_TEST_RECORDER=str(recorder))

    def run(*args, **extra):
        result = subprocess.run(["/bin/bash", str(target / SCRIPT), *args],
                                env={**env, **extra}, capture_output=True,
                                text=True, timeout=20)
        call = json.loads(result.stdout.splitlines()[-1]) if result.returncode == 0 and args[:1] != ("--help",) else None
        return result, call

    return tmp_path, run


@pytest.mark.parametrize("task", ["sudoku", "zebra", "countdown"])
@pytest.mark.parametrize("action", ["prepare", "smoke", "train", "tmux", "evaluate", "plot"])
def test_fixed_baseline_delegates_all_actions(launch, task, action):
    root, run = launch
    result, call = run(task, action, REASONING_NO_ROBUSTNESS="0", REASONING_GRADIENT_MODE="adjacent")
    assert result.returncode == 0, result.stderr
    assert call["args"] == ["h100", task, "mdm_aux", action]
    env = call["env"]
    for key, value in dict(REASONING_GPU_IDS="0", REASONING_MICRO_BATCH="8",
                           REASONING_GLOBAL_BATCH="128", REASONING_MAX_STEPS="5000",
                           REASONING_TRAIN_EXAMPLES="20000", REASONING_VALID_EXAMPLES="1000",
                           REASONING_TEST_EXAMPLES="1000", REASONING_SEED="1",
                           REASONING_NO_ROBUSTNESS="1", REASONING_GRADIENT_MODE="detached").items():
        assert env[key] == value
    label = "pilot-v1-n20000-v1000-t1000"
    assert env["REASONING_DATA_DIR"] == str(root / f".cache/reasoning/{task}-{label}")
    assert env["REASONING_RUN_DIR"] == str(root / f"outputs/reasoning/{task}/mdm_aux-h100-{label}-seed1")


def test_custom_counts_are_labelled_and_explicit_paths_preserved(launch):
    _, run = launch
    result, call = run("sudoku", "prepare", REASONING_TRAIN_EXAMPLES="200",
                       REASONING_VALID_EXAMPLES="20", REASONING_TEST_EXAMPLES="30")
    assert result.returncode == 0, result.stderr
    assert call["env"]["REASONING_DATA_DIR"].endswith("sudoku-pilot-v1-n200-v20-t30")
    result, call = run("zebra", "train", REASONING_DATA_DIR="/persistent/shared/data",
                       REASONING_RUN_DIR="/persistent/runs/control", REASONING_SIZE="sminy",
                       REASONING_MICRO_BATCH="16", REASONING_SEED="3")
    assert result.returncode == 0, result.stderr
    assert call["env"]["REASONING_DATA_DIR"] == "/persistent/shared/data"
    assert call["env"]["REASONING_RUN_DIR"] == "/persistent/runs/control"
    assert call["env"]["REASONING_SIZE"] == "sminy"
    assert call["env"]["REASONING_MICRO_BATCH"] == "16"
    assert call["env"]["REASONING_SEED"] == "3"


@pytest.mark.parametrize("args", [[], ["sudoku"], ["wrong", "train"], ["sudoku", "wrong"],
                                  ["sudoku", "train", "both_aux"]])
def test_bad_arguments_fail_before_delegation(launch, args):
    _, run = launch
    result, call = run(*args)
    assert result.returncode == 2
    assert call is None


@pytest.mark.parametrize("variable,value", [("REASONING_MAX_STEPS", "0"),
                                          ("REASONING_TRAIN_EXAMPLES", "-1"),
                                          ("REASONING_VALID_EXAMPLES", "abc"),
                                          ("REASONING_SEED", "../oops")])
def test_bad_numeric_settings_fail_before_delegation(launch, variable, value):
    _, run = launch
    result, call = run("sudoku", "train", **{variable: value})
    assert result.returncode == 2
    assert variable in result.stderr
    assert call is None
