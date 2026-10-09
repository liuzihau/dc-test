"""Explicit names for the split-attention suite; legacy IDs remain loadable."""

SPLIT_VARIANTS = ('mdm', 'tt', 'tt_ea', 'tt_ea_np', 'tt_ea_rm', 'tt_ea_rm_np')
SPLIT_LABELS = dict(zip(SPLIT_VARIANTS, (
    'MDM', 'TT', 'TT + EA', 'TT + EA + NP',
    'TT + EA + RM', 'TT + EA + RM + NP')))


def split_variant_config(variant):
    if variant not in SPLIT_VARIANTS:
        raise ValueError('Unknown split-suite variant: ' + variant)
    ea = '_ea' in variant
    rm = '_rm' in variant
    return dict(attention_mode='separate' if ea else 'vanilla',
                memory_mode='both' if rm else 'none',
                neighbors=variant.endswith('_np'),
                trajectory='single' if variant == 'mdm' else 'five')
