"""Merge CSV attempts without retaining the abandoned tail after a restart."""
import json
from pathlib import Path


def load_history(path_string):
    import pandas as pd
    root = Path(path_string).expanduser()
    files = [root] if root.is_file() else list(root.rglob('metrics.csv'))
    if not files:
        raise FileNotFoundError(f'No metrics.csv found below: {root}')
    # Preserve legacy ordering where no explicit attempt metadata exists.
    legacy = sorted((p for p in files if not (p.parent / 'resume_attempt.json').exists()),
                    key=lambda p: (p.stat().st_mtime_ns, str(p)))
    records = ([root.parent / 'resume_attempt.json'] if root.is_file() else
               list(root.rglob('resume_attempt.json')))
    attempts = []
    for p in records:
        if not p.exists():
            continue
        meta = json.loads(p.read_text())
        if meta['version'] != 1 or type(meta['resume_step']) is not int or meta['resume_step'] < 0:
            raise ValueError(f'Invalid resume lineage: {p}')
        attempts.append((meta['started_ns'], meta['attempt_id'], meta['resume_step'], p.parent / 'metrics.csv'))
    events = [(None, p) for p in legacy] + [(s, p) for _, _, s, p in sorted(attempts)]
    frames = []
    for index, (cutoff, path) in enumerate(events):
        if cutoff is not None:
            # Discard the entire abandoned future, not merely exact duplicate rows.
            frames = [f.loc[pd.to_numeric(f['step'], errors='coerce') < cutoff].copy()
                      for f in frames]
        if not path.exists() or path.stat().st_size == 0:
            continue  # A restarted attempt can exist before its first CSV flush.
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if 'step' not in frame:
            continue
        frame['_file_index'], frame['_row_index'] = index, range(len(frame))
        frame['_source_csv'] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=['step', '_file_index', '_row_index', '_source_csv'])
    return pd.concat(frames, ignore_index=True, sort=False)
