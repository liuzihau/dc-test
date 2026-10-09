import numpy as np
import pytest
from analysis.local_top5_metrics import events,statistics,transition_statistics,paired_summary


def example():
    clean=np.array([[99,10,11,12,13,14,99]])
    masked=np.ones_like(clean,dtype=bool)
    prediction=clean.copy();prediction[0,3]=13
    top5=np.tile(np.array([70,71,72,73,74]),(*clean.shape,1))
    top5[...,0]=prediction
    return clean,masked,prediction,top5


@pytest.mark.parametrize('case',['center_support','neighbor_only_support','no_local_support'])
def test_substitution_subclasses_are_disjoint(case):
    c,m,p,t=example()
    if case=='center_support': t[0,3,1]=12;t[0,2,2]=12
    elif case=='neighbor_only_support':t[0,4,4]=12
    e=events(c,m,p,t,block_size=7,special_ids=(99,))
    assert e['neighbor'][0,3] and e[case][0,3]
    assert sum(e[k][0,3] for k in ('center_support','neighbor_only_support','no_local_support'))==1


def test_other_error_with_top5_support_stays_other():
    c,m,p,t=example();p[0,3]=80;t[0,3,0]=80;t[0,3,1]=12
    e=events(c,m,p,t,block_size=7,special_ids=(99,))
    assert e['other'][0,3] and not e['neighbor'][0,3] and not e['center_support'][0,3]


def test_reciprocal_swap_has_neighbor_support():
    c,m,p,t=example();p[0,4]=12;t[0,4,0]=12
    e=events(c,m,p,t,block_size=7,special_ids=(99,))
    assert e['swap'][0,3] and e['neighbor_only_support'][0,3]


@pytest.mark.parametrize('invalid',['reveal','repeated','special'])
def test_eligibility_constraints(invalid):
    c,m,p,t=example()
    if invalid=='reveal':m[0,2]=False
    elif invalid=='repeated':c[0,2]=c[0,4]
    else:c[0,2]=99
    assert not events(c,m,p,t,block_size=7,special_ids=(99,))['eligible'][0,3]


def test_wrong_to_wrong_transitions_do_not_count_as_accuracy_gain():
    c,m,p,t=example();a=p.copy();a[0,3]=80;at=t.copy();at[0,3,0]=80
    table=transition_statistics(c,m,p,t,a,at,block_size=7,special_ids=(99,))
    assert table[0,3,4]==1
    vis=np.ones((1,1,16,3));vis[:,:,:,0]=2
    result=paired_summary(vis,vis,table[None],np.zeros((5,1),dtype=int))
    assert result['accuracy_gain']['value']==0


def test_correcting_substitution_has_positive_net_gain():
    c,m,p,t=example();a=c.copy();at=t.copy();at[...,0]=a
    table=transition_statistics(c,m,p,t,a,at,block_size=7,special_ids=(99,))
    assert table[0,3,0]==1
    count=statistics(c,m,p,t,block_size=7,special_ids=(99,))
    assert count[0,2]==count[0,6]==1
