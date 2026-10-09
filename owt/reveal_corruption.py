"""Paired corruption for masking and reliability sweeps; no global RNG use."""
import numpy as np
import hashlib


def nearest_different_sources(clean, forbidden_ids=()):
    """Closest allowed clean-token position with a different ID; ties go left."""
    clean=np.asarray(clean,dtype=np.int64)
    allowed=~np.isin(clean,list(forbidden_ids))
    sources=np.full(len(clean),-1,dtype=np.int64)
    for index,token in enumerate(clean):
        for distance in range(1,len(clean)):
            left=index-distance;right=index+distance
            if left>=0 and allowed[left] and clean[left]!=token:
                sources[index]=left;break
            if right<len(clean) and allowed[right] and clean[right]!=token:
                sources[index]=right;break
    return sources


def make_canvas(clean, mask_ratio, correct_fraction, seed, row_id, wrong_pool,
                mask_id, forbidden_ids=(), keep_first=False, nearest_sources=None):
    clean=np.asarray(clean,dtype=np.int64)
    if clean.ndim!=1 or len(clean)<2:
        raise ValueError('Requires a complete token sequence')
    if not 0<mask_ratio<=1 or not 0<=correct_fraction<=1:
        raise ValueError('Invalid mask ratio or revealed-token correctness')
    if nearest_sources is None:
        pool=np.asarray(wrong_pool,dtype=np.int64)
        allowed=(pool>=0)&(pool<mask_id)
        for token in forbidden_ids:
            allowed &= pool!=token
        pool=pool[allowed]
        if not len(pool) or not np.any(pool!=pool[0]):
            raise ValueError('Wrong-token pool needs at least two allowed token values')
    generators=[np.random.default_rng(np.random.SeedSequence([seed,row_id,k])) for k in range(3)]
    eligible=np.arange(1 if keep_first else 0,len(clean))
    order=generators[0].permutation(eligible)
    masked=np.zeros(len(clean),dtype=bool)
    masked[order[:round(mask_ratio*len(order))]]=True
    visible=~masked
    if keep_first:
        visible[0]=False  # Fixed anchor is outside the reliability population.
    unreliable_order=generators[1].permutation(eligible)
    unreliable_order=unreliable_order[visible[unreliable_order]]
    wrong=np.zeros(len(clean),dtype=bool)
    wrong[unreliable_order[:round((1-correct_fraction)*len(unreliable_order))]]=True
    if nearest_sources is not None:
        sources=np.asarray(nearest_sources,dtype=np.int64)
        if sources.shape!=clean.shape or np.any(sources[wrong]<0):
            raise ValueError('A requested wrong position has no different allowed clean source')
        replacements=clean[np.maximum(sources,0)]
    else:
        replacements=generators[2].choice(pool,size=len(clean))
        for _ in range(32):
            collisions=replacements==clean
            if not collisions.any():
                break
            replacements[collisions]=generators[2].choice(pool,size=int(collisions.sum()))
        # Avoid a pathological concentrated pool causing an unbounded rejection loop.
        for index in np.flatnonzero(replacements==clean):
            replacements[index]=pool[np.flatnonzero(pool!=clean[index])[0]]
    canvas=clean.copy()
    canvas[masked]=mask_id
    canvas[wrong]=replacements[wrong]
    assert not (wrong&masked).any()
    assert np.all(canvas[wrong]!=clean[wrong])
    assert np.all(canvas[visible&~wrong]==clean[visible&~wrong])
    info=dict(masked_positions=int(masked.sum()),
        reliability_population=int(visible.sum()),wrong_revealed_positions=int(wrong.sum()),
        actual_mask_ratio=float(masked[eligible].mean()),
        actual_correct_fraction=(float(1-wrong.sum()/visible.sum()) if visible.any() else None))
    if nearest_sources is not None:
        used=sources[wrong]
        targets=np.unique(used[masked[used]])
        info.update(copied_masked_source_occurrences=int(masked[used].sum()),
            distinct_copied_masked_targets=int((targets!=0).sum()),
            replacement_source_sha256=hashlib.sha256(sources.tobytes()).hexdigest())
    return canvas,masked,wrong,info
