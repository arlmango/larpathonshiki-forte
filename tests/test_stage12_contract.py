"""Synthetic numeric tests only; real stage12 training never uses these fixtures."""
import copy
import time
import numpy as np
from tools import stage12_mil as M


def units():
    rng = np.random.default_rng(12)
    result = []
    for i in range(8):
        y = int(i < 3)
        z = rng.normal(0, .15, (12+i, 6))
        if y:
            z[3:6, 0] += 2.
        t = np.arange(len(z), dtype=float)+.5
        aux = z[3:5].copy() if i == 0 else np.empty((0, 6))
        result.append({'y':y,'z':z,'t':t,'dur':float(len(z)),
                       'aux_z':aux,'aux_t':t[3:5] if i==0 else np.empty(0),
                       'aux_w':np.ones(len(aux)),'ivs':[(3.,5.)] if i==0 else [],'endpoints':[5.] if y else []})
    return result


def test_analytic_gradient_matches_finite_difference():
    u=units(); m=M.representation(u,list(range(len(u))))
    bags=[M.features(v,m) for v in u]; aux=[M.features(v,m,True) for v in u]
    f=lambda b:M.objective(b,bags,[v['y'] for v in u],aux,[v['aux_w'] for v in u])
    theta=np.linspace(-.05,.05,bags[0].shape[1]+1)
    _,gradient=f(theta)
    numeric=np.zeros_like(theta)
    for j in range(len(theta)):
        step=np.zeros_like(theta);step[j]=1e-6
        numeric[j]=(f(theta+step)[0]-f(theta-step)[0])/2e-6
    np.testing.assert_allclose(gradient,numeric,rtol=1e-5,atol=1e-7)


def test_heldout_data_and_labels_do_not_change_fit():
    u=units();ix=list(range(1,len(u)))
    first,_=M.train(u,ix,time.monotonic()+60)
    altered=copy.deepcopy(u)
    altered[0]['z'][:]=1e9;altered[0]['aux_z'][:]=-1e9
    altered[0]['y']=0;altered[0]['ivs']=[(99,100)]
    second,_=M.train(altered,ix,time.monotonic()+60)
    assert all(np.array_equal(first[k],second[k]) for k in first)
    assert M.inference(u[0],first)[0] == M.inference(u[0],second)[0]


def test_unknown_time_positive_contributes_bag_gradient_and_balance():
    u=units();m=M.representation(u,list(range(len(u))))
    bags=[M.features(v,m) for v in u];aux=[M.features(v,m,True) for v in u]
    y=[v['y'] for v in u]; aw=[v['aux_w'] for v in u]
    theta=np.zeros(bags[0].shape[1]+1)
    before=M.objective(theta,bags,y,aux,aw)[1]
    changed=copy.deepcopy(bags);changed[1]=changed[1]+2.
    after=M.objective(theta,changed,y,aux,aw)[1]
    assert not np.allclose(before,after)
    assert len(aux[1]) == len(aux[2]) == 0
    weights=M.balanced_weights(y)
    assert np.isclose(weights[np.array(y)==1].sum(),.5)
    assert np.isclose(weights[np.array(y)==0].sum(),.5)


def test_pooling_replication_and_reload(tmp_path):
    logits=np.array([-1.,2.,.3])
    assert np.isclose(M.pooling(logits)[0],M.pooling(np.tile(logits,3))[0])
    u=units();model,info=M.train(u,list(range(len(u))),time.monotonic()+60)
    assert info['converged'] and info['gradient_max_abs']<1e-6
    p=tmp_path/'synthetic.npz';np.savez(p,**model)
    with np.load(p,allow_pickle=False) as z: restored=dict(z)
    for v in u:
        np.testing.assert_array_equal(M.inference(v,model)[2],M.inference(v,restored)[2])
    assert M.metrics([v['y'] for v in u],[M.inference(v,model)[0]>=0 for v in u])['f1']>.8
