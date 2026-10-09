#!/usr/bin/env python3
"""Import locally downloaded public Shah Zebra files without executing pickle code."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reasoning.zebra_official import import_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--valid-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    manifest = import_dataset(args.train_file, args.test_file, args.output_dir,
                              train_size=args.train_size, valid_size=args.valid_size,
                              test_size=args.test_size, seed=args.seed)
    print(json.dumps({"status": "PASS", "task": manifest["task"],
                      "source": manifest["source"]["description"],
                      "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
