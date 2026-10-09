#!/usr/bin/env python3
"""Run the vendored author entrypoint with local runtime compatibility only.

Keeping this as ``sys.argv[0]`` ensures Lightning's spawned DDP rank also
installs the compatibility shim before importing the author model.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from reasoning.author_runtime import install_no_cudagraph_compile


def main() -> None:
  repo_root = Path(__file__).resolve().parents[2]
  author_main = repo_root / "third_party" / "reasoning_with_latent_tokens" / "main.py"
  # Match `python author_root/main.py`: its sibling imports (`models`,
  # `dataloader`, etc.) must win over similarly named modules in this repo.
  sys.path.insert(0, str(author_main.parent))
  if os.environ.get("DCACHE_AUTHOR_DISABLE_CUDAGRAPHS", "1") == "1":
    install_no_cudagraph_compile()
    print("AUTHOR COMPAT: compiled flex attention enabled; CUDA graphs disabled")
  runpy.run_path(str(author_main), run_name="__main__")


if __name__ == "__main__":
  main()
