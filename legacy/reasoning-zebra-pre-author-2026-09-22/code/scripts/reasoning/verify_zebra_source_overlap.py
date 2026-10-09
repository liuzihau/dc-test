#!/usr/bin/env python3
"""Independent exact-token train/test overlap check, no canonical clue parser."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from reasoning.zebra_official import load_source


def exact_key(raw):
    sequence, table, _ = raw
    return len(table[0]), len(table)-1, tuple(sequence[:sequence.index('ANSWER')])


def main():
    data = Path('.cache/reasoning/zebra-benchmark-full-v1')
    m = json.loads((data/'manifest.json').read_text())
    frozen = [json.loads(line) for line in (data/'test.jsonl').open()]
    keys = {(r['metadata']['houses'],r['metadata']['attributes'],tuple(r['prompt'])):r['id'] for r in frozen}
    test = load_source(m['source']['source_files']['test']['path'])
    index = {}
    for i,r in enumerate(test):
        index.setdefault(exact_key(r),i)
    train = load_source(m['source']['source_files']['train']['path'])
    seen, selected, examples = set(),set(),[]
    rows = 0
    for i,r in enumerate(train):
        k=exact_key(r)
        if k in index:
            rows+=1;seen.add(k)
            if len(examples)<6:
                j=index[k]
                assert r[1]==test[j][1]
                examples.append(dict(train_index=i,test_index=j,houses=k[0],attributes=k[1],
                    prompt_tokens=list(k[2]),same_solution_table=True,
                    same_full_token_sequence=r[0]==test[j][0]))
        if k in keys:
            selected.add(keys[k])
    report=dict(method='Exact tuple equality of dimensions and every clue token, not logical normalization',
        raw_train_rows=len(train),raw_test_rows=len(test),
        train_rows_matching_test=rows,unique_test_prompts_in_train=len(seen),
        unique_test_prompts=len(index),raw_test_rows_in_train=sum(exact_key(r) in seen for r in test),
        frozen_1000_test_puzzles_in_raw_train=len(selected),frozen_test_ids=sorted(selected),
        examples=examples,
        caveat='Public raw-source overlap is not proof that a paper retained this overlap after its preprocessing.')
    output=Path('results/generated/audits/zebra-source-duplicates/exact_overlap.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('examples','frozen_test_ids')},indent=2))


if __name__=='__main__':main()
