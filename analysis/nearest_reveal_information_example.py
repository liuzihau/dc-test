"""Exact information example for clean-source nearest-token replacements.

This is an illustrative discrete distribution, not an OWT model evaluation.
The actual corruption helper runs on both equally likely clean sequences.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from owt.reveal_corruption import make_canvas, nearest_different_sources


def conditional_entropy(observations):
    """Exact H(Y|canvas) for a finite list of equally probable states."""
    groups=defaultdict(Counter)
    for target,canvas in observations:
        groups[tuple(canvas)][target]+=1
    total=len(observations)
    entropy=0.
    for counts in groups.values():
        n=sum(counts.values())
        entropy+=n/total*sum(-(v/n)*math.log(v/n) for v in counts.values())
    return entropy


def example(seed=20261004):
    # All positions are known token A=2 except position 12, a fair bit Y=0/1.
    # Choose a mask draw independently of Y that masks Y and leaves position 0
    # revealed. There are exactly 5 scored masked and 20 revealed positions.
    clean=np.full(25,2,dtype=np.int64);clean[12]=0
    sources=nearest_different_sources(clean,[3])
    for row_id in range(10000):
        _,masked,_,_=make_canvas(clean,.2,1.,seed,row_id,None,3,nearest_sources=sources)
        if masked[12] and not masked[0]:
            break
    else:
        raise RuntimeError('No eligible mask draw found')
    mask_indices=np.flatnonzero(masked).tolist()
    summary=[];states=[]
    for correctness in [1.,.8,.6]:
        observations=[]
        for target in [0,1]:
            truth=clean.copy();truth[12]=target
            sources=nearest_different_sources(truth,[3])
            canvas,current_mask,wrong,info=make_canvas(truth,.2,correctness,seed,row_id,None,3,
                nearest_sources=sources)
            assert np.array_equal(current_mask,masked)
            assert info['actual_mask_ratio']==.2
            assert math.isclose(info['actual_correct_fraction'],correctness)
            assert info['masked_positions']==5 and info['reliability_population']==20
            # Every potential wrong revealed position copies the hidden bit.
            assert np.all(sources[~masked]==12)
            assert np.all(canvas[wrong]==target)
            observations.append((target,canvas.tolist()))
            states.append(dict(correct_fraction=correctness,target=target,
                clean=truth.tolist(),canvas=canvas.tolist(),
                wrong_positions=np.flatnonzero(wrong).tolist(),metadata=info))
        entropy=conditional_entropy(observations)
        expected=math.log(2) if correctness==1 else 0.
        assert math.isclose(entropy,expected,abs_tol=1e-12)
        summary.append(dict(mask_ratio=.2,correct_fraction=correctness,
            wrong_revealed_positions=round((1-correctness)*20),
            target_conditional_entropy_nats=entropy,
            pooled_masked_bayes_ce=entropy/5,
            extra_information_about_target_vs_correct_context_nats=math.log(2)-entropy))
    return dict(kind='exact illustrative information counterexample',seed=seed,row_id=row_id,
        clean_distribution='Y at position 12 is a fair bit 0/1; every other position is constant token A=2',
        mask_indices=mask_indices,mask_token=3,summary=summary,states=states,
        conclusion='Nearest clean-token substitutions are not a garbling of the fully correct revealed canvas: they can add information about masked targets.',
        limits=['The distribution is a constructed example, not natural language or a trained neural model.',
                'This demonstrates absence of a universal monotonic Bayes-risk guarantee, not an OWT robustness outcome.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/analysis/nearest-reveal-information-example/summary.json'))
    args=parser.parse_args()
    result=example()
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['corruption_source_sha256']=hashlib.sha256(Path('owt/reveal_corruption.py').read_bytes()).hexdigest()
    if args.output.exists():
        if json.loads(args.output.read_text())!=result:
            raise RuntimeError('Refusing to overwrite different example evidence')
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':
    main()
