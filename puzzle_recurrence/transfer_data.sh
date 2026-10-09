#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: bash puzzle_recurrence/transfer_data.sh USER@SOURCE_HOST" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
data_target="$repo_root/.cache/downloads/reasoning-puzzles/Reasoning puzzles public data"
source_repo="${DCACHE_SOURCE_REPO:-/share2/home/tliu0205/dc-test}"
source_path="${DCACHE_SOURCE_DATA_DIR:-$source_repo/.cache/downloads/reasoning-puzzles/Reasoning puzzles public data}"
mkdir -p "$data_target"
list_file="$(mktemp)"
trap 'rm -f "$list_file"' EXIT
python -c 'import json; from pathlib import Path; p=Path("puzzle_recurrence/data_manifest.json"); print("\n".join(x["name"] for x in json.loads(p.read_text())["files"]))' > "$list_file"
rsync -avP --protect-args --files-from="$list_file" "$1:$source_path/" "$data_target/"
for cache in author-sudoku author-zebra; do
  mkdir -p "$repo_root/.cache/$cache"
  rsync -avP --protect-args "$1:$source_repo/.cache/$cache/" "$repo_root/.cache/$cache/"
done
python -m puzzle_recurrence.data_setup --sha256
