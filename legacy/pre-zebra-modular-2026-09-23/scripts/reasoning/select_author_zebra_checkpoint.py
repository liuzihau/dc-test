#!/usr/bin/env python3
"""Select the newest complete author-Zebra checkpoint by optimizer step."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import torch


STEP_PATTERN = re.compile(r"-(\d+)\.ckpt$")


def checkpoint_step(path: Path) -> Optional[int]:
  match = STEP_PATTERN.search(path.name)
  if match and path.name not in {"last.ckpt", "best.ckpt"}:
    return int(match.group(1))
  return None


def load_step(path: Path, author_root: Path) -> int:
  sys.path.insert(0, str(author_root))
  payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
  step = int(payload.get("global_step", -1))
  if step < 0:
    raise RuntimeError(f"Missing global_step in {path}")
  return step


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint_dir", type=Path)
  parser.add_argument("--author-root", type=Path, required=True)
  parser.add_argument("--show-step", action="store_true")
  args = parser.parse_args()

  candidates: list[tuple[int, Path]] = []
  for path in args.checkpoint_dir.glob("*.ckpt"):
    parsed = checkpoint_step(path)
    if parsed is not None:
      candidates.append((parsed, path.resolve()))

  # `last.ckpt` is the normal clean-segment checkpoint. A numbered periodic
  # checkpoint can be newer after an interrupted segment, so compare both.
  last = args.checkpoint_dir / "last.ckpt"
  if last.is_file():
    candidates.append((load_step(last, args.author_root), last.resolve()))

  if not candidates:
    raise SystemExit(f"No checkpoints found in {args.checkpoint_dir}")

  step, path = max(candidates, key=lambda item: (item[0], item[1].name == "last.ckpt"))
  # Verify the selected numbered checkpoint is readable and its filename is
  # truthful. This catches interrupted/non-atomic writes before resumption.
  verified_step = load_step(path, args.author_root)
  if verified_step != step:
    raise RuntimeError(f"Checkpoint step mismatch for {path}: {step} != {verified_step}")

  if args.show_step:
    print(f"{step}\t{path}")
  else:
    print(path)


if __name__ == "__main__":
  main()
