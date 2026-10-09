#!/usr/bin/env python3
"""Read-only audit of exact and canonical duplication in released Zebra data."""
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reasoning.zebra_official import load_source, logical_identity


def key(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).digest()


def main():
    data = Path('.cache/reasoning/zebra-benchmark-full-v1')
    manifest = json.loads((data/'manifest.json').read_text())
    output = Path('results/generated/audits/zebra-source-duplicates')
    output.mkdir(parents=True, exist_ok=True)
    validation = {}
    for line in (data/'validation.jsonl').open():
        r = json.loads(line)
        h, a = r['metadata']['houses'], r['metadata']['attributes']
        prompt = ['HOUSES', str(h), 'ATTRS', str(a)] + r['prompt']
        validation[key(['zebra-official', logical_identity(prompt)])] = True
    packed_ids = (data/'train.identities.bin').read_bytes()
    packed_keys = {packed_ids[i:i+32] for i in range(0,len(packed_ids),32)}
    sources, test_exact, test_canonical, examples = {}, set(), set(), []
    retained, counters, distributions = set(), Counter(), {}
    for split in ('test', 'train'):
        source = manifest['source']['source_files'][split]
        path = Path(source['path'])
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024), b''):
                digest.update(block)
        assert digest.hexdigest() == source['sha256']
        raw = load_source(path)
        exact_counts, canonical_counts, sequence_table_counts = Counter(), Counter(), Counter()
        first, answers, histogram = {}, {}, Counter()
        conflicts = exact_overlap = canonical_overlap = 0
        for i, (sequence,table,trace) in enumerate(raw):
            h, a = len(table[0]), len(table)-1
            clues = sequence[:sequence.index('ANSWER')]
            prompt = ['HOUSES',str(h),'ATTRS',str(a)]+clues
            exact = key([h,a,clues])
            canonical = key(['zebra-official', logical_identity(prompt)])
            combined = key([sequence,table])  # Solver-order ndarray excluded.
            answer = key(table)
            if canonical in answers and answers[canonical] != answer:
                conflicts += 1
            answers.setdefault(canonical, answer)
            histogram[f'{h}x{a}:raw'] += 1
            if not canonical_counts[canonical]:
                histogram[f'{h}x{a}:canonical_unique'] += 1
            if split == 'train':
                exact_overlap += int(exact in test_exact)
                canonical_overlap += int(canonical in test_canonical)
                if canonical in test_canonical:
                    category = 'excluded_test'
                elif canonical in validation:
                    category = 'excluded_validation'
                elif canonical in retained:
                    category = 'duplicate_train'
                else:
                    category = 'retained_train'
                    retained.add(canonical)
                    histogram[f'{h}x{a}:retained'] += 1
                counters[category] += 1
                if canonical in first and len(examples) < 8:
                    old_index, old_exact = first[canonical]
                    examples.append(dict(first_index=old_index, repeat_index=i,
                        exact_visible_prompt=old_exact==exact, canonical_sha256=canonical.hex(),
                        houses=h, attributes=a))
                first.setdefault(canonical, (i,exact))
            exact_counts[exact] += 1
            canonical_counts[canonical] += 1
            sequence_table_counts[combined] += 1
            if (i+1)%100000 == 0:
                print(split, i+1, 'exact_unique',len(exact_counts),'canonical_unique',len(canonical_counts),flush=True)
        sources[split] = dict(path=str(path),sha256=digest.hexdigest(),rows=len(raw),
            unique_exact_visible_prompts=len(exact_counts), unique_canonical_puzzles=len(canonical_counts),
            unique_sequence_plus_table=len(sequence_table_counts),
            exact_prompt_repeat_rows=len(raw)-len(exact_counts),
            extra_unique_prompts_collapsed_by_canonicalization=len(exact_counts)-len(canonical_counts),
            canonical_repeat_rows=len(raw)-len(canonical_counts),
            conflicting_answer_rows=conflicts,
            canonical_multiplicity_histogram=dict(sorted(Counter(canonical_counts.values()).items())),
            train_rows_matching_test_exact=exact_overlap,
            train_rows_matching_test_canonical=canonical_overlap,
            unique_canonical_train_test_overlap=len(set(canonical_counts)&test_canonical) if split=='train' else None)
        distributions[split] = dict(sorted(histogram.items()))
        if split == 'test':
            test_exact, test_canonical = set(exact_counts), set(canonical_counts)
        del raw, exact_counts, canonical_counts, sequence_table_counts, first, answers
        gc.collect()
    actual = dict(raw_train=sources['train']['rows'],**counters)
    assert actual == manifest['source']['counters'], (actual,manifest['source']['counters'])
    assert retained == packed_keys, 'Prepared identities differ from independently selected source puzzles'
    report = dict(sources=sources, filter_counters=actual, distributions=distributions,
        duplicate_examples=examples, retained_matches_packed=True,
        identity_definition='Dimensions plus clue set; ignores clue order/repeated clauses; symmetric =/!=/nbr operands sorted. Does NOT deduplicate by solution grid.',
        limit='Full byte-pickle-record equality not claimed: sequence+table hash excludes solver-order ndarray. No assertion about unpublished paper preprocessing.')
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='distributions'},indent=2),flush=True)


if __name__=='__main__':
    main()
