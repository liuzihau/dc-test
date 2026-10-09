"""Two-level local error taxonomy and paired document-bootstrap estimates."""
import numpy as np
from analysis.local_denoising_metrics import error_events,bootstrap_ratio

FIELDS=('eligible','wrong','neighbor','other','center_support','neighbor_only_support','no_local_support','swap')
CLASSES=('correct','center_support','neighbor_only_support','no_local_support','other')


def events(clean,masked,prediction,top5,**kwargs):
    clean=np.asarray(clean);top5=np.asarray(top5)
    if top5.shape!=(*clean.shape,5): raise ValueError('Require five candidate IDs per position')
    original=error_events(clean,masked,prediction,**kwargs)
    result={k:original[k] for k in ('eligible','wrong','other','swap')}
    neighbor=original['out_left']|original['out_right']
    center=(top5==clean[...,None]).any(-1)
    adjacent=np.zeros_like(center)
    adjacent[:,1:-1]=((top5[:,:-2]==clean[:,1:-1,None]).any(-1)|
                       (top5[:,2:]==clean[:,1:-1,None]).any(-1))
    result.update(neighbor=neighbor,center_support=neighbor&center,
        neighbor_only_support=neighbor&~center&adjacent,
        no_local_support=neighbor&~center&~adjacent)
    if not np.array_equal(result['neighbor']|result['other'],result['wrong']):
        raise ValueError('Outer error partition is incomplete')
    if not np.array_equal(sum(result[k].astype(int) for k in CLASSES[1:4]),neighbor.astype(int)):
        raise ValueError('Top-five subclasses overlap or omit substitutions')
    if np.any(result['swap']&result['no_local_support']):
        raise ValueError('A reciprocal swap must have local top-five support')
    return result


def statistics(clean,masked,prediction,top5,**kwargs):
    e=events(clean,masked,prediction,top5,**kwargs)
    return np.stack([e[k].sum(1) for k in FIELDS],axis=1)


def transition_statistics(clean,masked,baseline,baseline_top5,candidate,candidate_top5,**kwargs):
    left=events(clean,masked,baseline,baseline_top5,**kwargs)
    right=events(clean,masked,candidate,candidate_top5,**kwargs)
    if not np.array_equal(left['eligible'],right['eligible']): raise ValueError('Paired populations differ')
    left['correct']=left['eligible']&~left['wrong'];right['correct']=right['eligible']&~right['wrong']
    return np.stack([np.stack([(left[a]&right[b]).sum(1) for b in CLASSES],axis=1) for a in CLASSES],axis=1)


def summaries(visibility,counts,draws):
    # Preserve all corruption seeds within each document before resampling.
    vis=np.asarray(visibility).sum(0);cnt=np.asarray(counts).sum(0)
    e={name:cnt[:,index] for index,name in enumerate(FIELDS)}
    patterns=[]
    for c in range(16):
        patterns.append(dict(pattern=format(c,'04b'),count=int(vis[:,c,0].sum()),
            accuracy=bootstrap_ratio(vis[:,c,1],vis[:,c,0],draws),
            nll=bootstrap_ratio(vis[:,c,2],vis[:,c,0],draws)))
    categories={}
    for name in FIELDS[2:]:
        record=dict(count=int(e[name].sum()),
            among_wrong=bootstrap_ratio(e[name],e['wrong'],draws),
            among_eligible=bootstrap_ratio(e[name],e['eligible'],draws))
        if name in CLASSES[1:4]: record['among_neighbor']=bootstrap_ratio(e[name],e['neighbor'],draws)
        categories[name]=record
    return dict(visibility=patterns,
        overall_accuracy=bootstrap_ratio(vis[:,:,1].sum(1),vis[:,:,0].sum(1),draws),
        overall_nll=bootstrap_ratio(vis[:,:,2].sum(1),vis[:,:,0].sum(1),draws),
        eligible_centers=int(e['eligible'].sum()),wrong_centers=int(e['wrong'].sum()),
        error_rate=bootstrap_ratio(e['wrong'],e['eligible'],draws),error_categories=categories)


def paired_difference(numerator_a,numerator_b,denominator,draws):
    a=np.asarray(numerator_a);b=np.asarray(numerator_b);d=np.asarray(denominator)
    return bootstrap_ratio(a-b,d,draws)


def paired_summary(baseline_visibility,candidate_visibility,transitions,draws):
    b=np.asarray(baseline_visibility).sum(0);a=np.asarray(candidate_visibility).sum(0)
    if not np.array_equal(a[:,:,0],b[:,:,0]): raise ValueError('Visibility denominators differ')
    table=np.asarray(transitions).sum(0)
    denominator=table.sum((1,2))
    gains={}
    for k,name in enumerate(CLASSES[1:],1):
        gains[name]=dict(corrected=int(table[:,k,0].sum()),introduced=int(table[:,0,k].sum()),
            net_gain=paired_difference(table[:,k,0],table[:,0,k],denominator,draws))
    return dict(classes=list(CLASSES),transition_counts=table.sum(0).astype(int).tolist(),
        gain_accounting=gains,
        accuracy_gain=paired_difference(table[:,1:,0].sum(1),table[:,0,1:].sum(1),denominator,draws),
        visibility=[dict(pattern=format(c,'04b'),
            accuracy_delta=paired_difference(a[:,c,1],b[:,c,1],b[:,c,0],draws),
            nll_delta=paired_difference(a[:,c,2],b[:,c,2],b[:,c,0],draws)) for c in range(16)])
