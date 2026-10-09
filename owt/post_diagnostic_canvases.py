"""Pure construction of paired post-5k diagnostic inputs; no global RNG."""
import numpy as np

from owt.reveal_corruption import make_canvas, nearest_different_sources


def clean_context(clean, ratio, row_id, seed=20261004, mask_id=50257,
                  boundaries=(50256,)):
    return make_canvas(clean, ratio, 1., seed, row_id, None, mask_id,
                       boundaries, nearest_sources=nearest_different_sources(
                           clean, (*boundaries, mask_id)))


def anchor_pair(clean, ratio, row_id, seed=20261004, mask_id=50257,
                boundaries=(50256,)):
    original, masked, wrong, _ = clean_context(
        clean, ratio, row_id, seed, mask_id, boundaries)
    anchored = original.copy()
    anchored[0] = clean[0]
    anchored_mask = masked.copy()
    anchored_mask[0] = False
    assert np.array_equal(original[1:], anchored[1:])
    assert np.array_equal(masked[1:], anchored_mask[1:])
    return (original, masked, wrong), (anchored, anchored_mask, wrong.copy())


def source_pairs(clean, ratio, row_id, seed=20261004,
                 selection_seed=20261005, mask_id=50257, boundaries=(50256,)):
    """Same masked target for both directions; source zero is never toggled."""
    _, (background, masked, _) = anchor_pair(
        clean, ratio, row_id, seed, mask_id, boundaries)
    content = ~np.isin(clean, (*boundaries, mask_id))
    eligible = masked[1:-1] & content[:-2] & content[1:-1] & content[2:]
    candidates = np.flatnonzero(eligible) + 1
    candidates = candidates[candidates >= 2]  # Keep input zero clean in both states.
    if not len(candidates):
        return []
    rng = np.random.default_rng(np.random.SeedSequence(
        [selection_seed, row_id, round(ratio * 100)]))
    target = int(rng.choice(candidates))
    result = []
    for direction in (-1, 1):  # Source index relative to target.
        source = target + direction
        latent, visible = background.copy(), background.copy()
        latent[source] = mask_id
        visible[source] = clean[source]
        other = np.arange(len(clean)) != source
        assert np.array_equal(latent[other], visible[other])
        assert latent[target] == visible[target] == mask_id
        assert latent[0] == visible[0] == clean[0]
        for state, canvas in [('masked', latent), ('revealed', visible)]:
            result.append(dict(row_id=int(row_id), mask_ratio=ratio,
                direction=direction, target_index=target, source_index=source,
                source_state=state, canvas=canvas,
                background_masked_positions=int(masked.sum()),
                actual_masked_positions=int((canvas == mask_id).sum())))
    return result
