"""CPU-only monitoring, factual experiment journal, and a living LaTeX report.

This module deliberately imports no tensor/GPU libraries. It records observations;
scientific theory revisions are authored separately and archived before replacement.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'outputs/owt/mdm-np-5k'
DOC = ROOT/'outputs/research-notes'
LABELS = {'mdm': 'MDM', 'mdm_np': 'Random NP',
          'mdm_np_zero_init': 'Zero NP',
          'mdm_np_zero_init_low_weight': 'Low-weight NP'}


def timestamp():
    return datetime.now(ZoneInfo('Australia/Sydney')).isoformat(timespec='seconds')


def tex_escape(text):
    substitutions = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
        '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(substitutions.get(c, c) for c in str(text))


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.partial')
    temporary.write_text(text)
    temporary.replace(path)


def record_event(identifier, title, body, details=None, doc=DOC):
    """Append once under a shared lock, preserving the original event facts."""
    doc = Path(doc)
    doc.mkdir(parents=True, exist_ok=True)
    path = doc/'research_events.jsonl'
    with (doc/'.journal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        if any(event['id'] == identifier for event in old):
            return False
        event = dict(id=identifier, time=timestamp(), title=title,
                     body=body, details=details or {})
        with path.open('a') as stream:
            stream.write(json.dumps(event, sort_keys=True)+'\n')
        with (doc/'research_activity_log.tex').open('a') as stream:
            stream.write('\n\\paragraph{'+tex_escape(event['time']+' — '+title)+'}\n'
                         +tex_escape(body)+'\n')
    return True


def read_csv(path):
    """Ignore an incomplete append; preserve only complete finite numeric rows."""
    path = Path(path)
    if not path.exists():
        return []
    rows = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                if None in row or any(value is None or value == '' for value in row.values()):
                    continue
                numbers = {key: float(value) for key, value in row.items()}
                step = numbers['optimizer_step']
                if not all(math.isfinite(value) for value in numbers.values()) or not step.is_integer():
                    continue
                numbers['optimizer_step'] = int(step)
                rows[int(step)] = numbers
            except (TypeError, ValueError, KeyError):
                continue
    return [rows[step] for step in sorted(rows)]


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def collect(root=RUNS):
    root = Path(root)
    snapshot = dict(observed_at=timestamp(), root=str(root), variants={}, queues={})
    for variant, label in LABELS.items():
        run = root/variant
        if not run.is_dir():
            continue
        train_path = run/'local_metrics/train.csv'
        train, valid = read_csv(train_path), read_csv(run/'local_metrics/validation.csv')
        contract = read_json(run/'contract.json')
        completion = read_json(run/'complete.json')
        item = dict(label=label, train_rows=len(train), validation_rows=valid,
                    latest_train=train[-1] if train else None,
                    latest_validation=valid[-1] if valid else None,
                    completion=completion, contract=contract)
        if train_path.exists():
            item['seconds_since_train_row'] = max(0, time.time()-train_path.stat().st_mtime)
        if train and contract:
            np = contract['mechanisms']['np']
            recent = train[-128:]
            main = sum(row['main_elbo'] for row in recent)/len(recent)
            item['trailing_main_128'] = main
            if np['enabled']:
                weights = dict(zip(np['offsets'], np['weights']))
                auxiliary = sum(weights[-1]*row['np_prev']+weights[1]*row['np_next']
                                for row in recent)/len(recent)
                item['weighted_aux_to_main_128'] = auxiliary/main if main else None
        snapshot['variants'][variant] = item
    for filename in ('current.json', 'zero_init_queue.json', 'low_weight_queue.json', 'reveal_sweep_queue.json'):
        data = read_json(root/filename)
        if data is not None:
            snapshot['queues'][filename] = data
    baseline = snapshot['variants'].get('mdm', {})
    controls = {row['optimizer_step']:row for row in baseline.get('validation_rows', [])}
    for variant, item in snapshot['variants'].items():
        row = item['latest_validation']
        if row and row['optimizer_step'] in controls:
            delta = row['val_nll']-controls[row['optimizer_step']]['val_nll']
            item['matched_validation_delta'] = delta
            item['matched_ppl_bound_change_percent'] = 100*math.expm1(delta)
    return snapshot


def status_tex(snapshot):
    variants = snapshot['variants']
    output = ['\\textbf{Observation time:} '+tex_escape(snapshot['observed_at'])+'.',
              'Lower validation main-head ELBO is better. Dashes denote unavailable results.',
              '\\begin{center}\\small',
              '\\begin{tabular}{r|rrrr}',
              'Update & MDM & Random NP & Zero NP & Low-weight NP \\\\ \\hline']
    lookup = {name:{row['optimizer_step']:row for row in item['validation_rows']}
              for name,item in variants.items()}
    steps = sorted({step for rows in lookup.values() for step in rows})
    for step in steps:
        cells = [f'{lookup[name][step]["val_nll"]:.4f}' if step in lookup.get(name,{}) else '--'
                 for name in LABELS]
        output.append(str(step)+' & '+' & '.join(cells)+r' \\')
    output += ['\\end{tabular}\\end{center}',
               '\\textbf{Live training progress at this snapshot.}']
    for variant, item in variants.items():
        if item['latest_train']:
            stage = 'complete' if item['completion'] else 'in progress'
            output.append(tex_escape(f'{item["label"]}: update '
                f'{item["latest_train"]["optimizer_step"]:,}, {stage}.')+'\\par')
    base, random = variants.get('mdm',{}), variants.get('mdm_np',{})
    if base.get('completion') and random.get('completion'):
        b, r = base['latest_validation'], random['latest_validation']
        delta = r['val_nll']-b['val_nll']
        output += [tex_escape(f'At 5,000 updates, random-init NP is worse by {delta:.6f} '
             f'nats/token, equivalent to {100*math.expm1(delta):.2f}% in the perplexity bound.')]
        if base.get('trailing_main_128') and random.get('trailing_main_128'):
            output += [tex_escape('Trailing 128-update training main losses: '
                f'MDM {base["trailing_main_128"]:.6f}; '
                f'random-init NP {random["trailing_main_128"]:.6f}.')]
        bh, rh = b['elapsed_seconds']/3600, r['elapsed_seconds']/3600
        output += [tex_escape(f'Observed training plus monitoring durations: MDM {bh:.2f} hours; '
                    f'random-init NP {rh:.2f} hours (ratio {rh/bh:.3f}). '
                    'This is measured elapsed time, not a FLOP-matched comparison.')]
    return '\n\n'.join(output)+'\n'


def archive_theory(source, rationale, doc=DOC):
    """Archive the old theory and its observed status before a new revision."""
    doc = Path(doc)
    metadata_path = doc/'theory_versions.json'
    with (doc/'.theory.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = read_json(metadata_path) or dict(current_version=0, revisions=[])
        old_version = metadata['current_version']
        theory = doc/'current_theory.tex'
        if theory.exists():
            history = doc/'history'
            history.mkdir(exist_ok=True)
            archived = history/f'theory-v{old_version:03d}.tex'
            if archived.exists():
                raise RuntimeError(f'Refusing to overwrite theory archive {archived}')
            text = theory.read_text()
            status = doc/'generated_status.tex'
            if status.exists():
                text = text.replace(r'\input{outputs/research-notes/generated_status.tex}', status.read_text())
            # Freeze figures too: later analyses must not silently alter old evidence.
            def freeze_figure(match):
                asset = Path(match.group(2))
                if not asset.is_absolute():
                    asset = ROOT/asset
                digest = hashlib.sha256(asset.read_bytes()).hexdigest()[:16]
                folder = history/f'theory-v{old_version:03d}-assets'
                folder.mkdir(exist_ok=True)
                frozen = folder/f'{digest}-{asset.name}'
                if not frozen.exists():
                    shutil.copyfile(asset, frozen)
                reference = str(frozen.relative_to(ROOT)) if frozen.is_relative_to(ROOT) else str(frozen)
                return match.group(1)+'{'+reference+'}'
            text = re.sub(r'(\\includegraphics(?:\[[^\]]*\])?)\{([^}]+)\}', freeze_figure, text)
            archived.write_text(text)
            archive_reference = str(archived.relative_to(ROOT)) if archived.is_relative_to(ROOT) else str(archived)
            with (doc/'theory_history.tex').open('a') as stream:
                stream.write('\n\\subsection{'+tex_escape(f'Theory version {old_version}')+'}\n'
                             +'\\begingroup\\let\\section\\subsubsection\\let\\subsection\\paragraph\n'
                             +'\\input{'+archive_reference+'}\n\\endgroup\n')
        version = old_version+1
        atomic_write(theory, Path(source).read_text())
        metadata['current_version'] = version
        metadata['revisions'].append(dict(version=version, time=timestamp(), rationale=rationale,
            sha256=hashlib.sha256(theory.read_bytes()).hexdigest()))
        atomic_write(metadata_path, json.dumps(metadata, indent=2)+'\n')
        record_event(f'theory_revision_{version}', f'Theory revision {version}', rationale, doc=doc)


def compile_pdf(doc=DOC):
    if shutil.which('latexmk') is None:
        raise RuntimeError('latexmk is unavailable')
    build = ROOT/'.cache/runtime/research-latex'
    build.mkdir(parents=True, exist_ok=True)
    command = ['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
               '-file-line-error', f'-outdir={build}',
               str(Path(doc)/'neighbor-prediction-research-assessment.tex')]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=120)
    atomic_write(Path(doc)/'latex_build.log', result.stdout)
    if result.returncode:
        raise RuntimeError('LaTeX compilation failed; see outputs/research-notes/latex_build.log')
    source = build/'neighbor-prediction-research-assessment.pdf'
    destination = Path(doc)/source.name
    temporary = destination.with_suffix('.partial.pdf')
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    atomic_write(Path(doc)/'last_successful_build.json',
                 json.dumps(dict(time=timestamp(), signature=build_signature(doc)), indent=2)+'\n')


def build_signature(doc=DOC):
    digest = hashlib.sha256()
    for filename in ('neighbor-prediction-research-assessment.tex', 'current_theory.tex',
                     'generated_status.tex', 'research_activity_log.tex', 'theory_history.tex'):
        path = Path(doc)/filename
        if path.exists():
            digest.update(filename.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def observe(root=RUNS, doc=DOC, build=True):
    snapshot = collect(root)
    changed = False
    for variant, item in snapshot['variants'].items():
        row = item['latest_validation']
        if row:
            body = (f'{item["label"]}, update {row["optimizer_step"]:,}: '
                    f'EMA main-head validation ELBO {row["val_nll"]:.6f} nats/token '
                    'on the fixed 1,024-row monitoring subset.')
            if 'matched_validation_delta' in item:
                body += (f' Difference from MDM at the same update: '
                         f'{item["matched_validation_delta"]:+.6f} nats/token. '
                         'This checkpoint is not an independent training replicate.')
            digest = hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:12]
            changed |= record_event(f'validation_{variant}_{row["optimizer_step"]}_{digest}',
                                    'Validation observation', body, row, doc)
        if item['completion']:
            changed |= record_event(f'complete_{variant}', 'Training completed',
                f'{item["label"]} completed {item["completion"]["optimizer_step"]:,} optimizer updates. '
                'A completion marker and saved checkpoint are present.', item['completion'], doc)
        elif item.get('seconds_since_train_row',0) > 900:
            step = item['latest_train']['optimizer_step'] if item['latest_train'] else 0
            changed |= record_event(f'stall_{variant}_{step}', 'Progress requires inspection',
                f'{item["label"]} has not appended a training row for more than 15 minutes. '
                'Inspect the controller and training log; no automatic restart or termination was performed.',
                doc=doc)
    atomic_write(Path(doc)/'research_monitor.json', json.dumps(snapshot,indent=2)+'\n')
    if changed or not (Path(doc)/'generated_status.tex').exists():
        atomic_write(Path(doc)/'generated_status.tex', status_tex(snapshot))
    successful = read_json(Path(doc)/'last_successful_build.json') or {}
    if build and successful.get('signature') != build_signature(doc):
        compile_pdf(doc)
    return snapshot, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['watch','snapshot','revise'])
    parser.add_argument('--root', type=Path, default=RUNS)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--rationale')
    parser.add_argument('--no-build', action='store_true')
    args = parser.parse_args()
    if args.action == 'revise':
        if not args.source or not args.rationale:
            parser.error('revise requires --source and --rationale')
        archive_theory(args.source, args.rationale)
        compile_pdf()
        return
    if not 1 <= args.interval <= 60:
        parser.error('interval must be between 1 and 60 seconds')
    DOC.mkdir(parents=True, exist_ok=True)
    with (DOC/'.monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                snapshot, changed = observe(args.root, build=not args.no_build)
                if changed or args.action == 'snapshot':
                    print(json.dumps(dict(time=snapshot['observed_at'], changed=changed,
                        steps={name:item['latest_train']['optimizer_step']
                            for name,item in snapshot['variants'].items() if item['latest_train']})), flush=True)
            except Exception as exc:
                print(f'{timestamp()} monitor error: {exc}', file=sys.stderr, flush=True)
                if args.action == 'snapshot':
                    raise
            if args.action == 'snapshot':
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
