"""Synthetic numerical, train-fold isolation and inference/reload checks."""
import time
import numpy as np
from tools import stage16_persistence as P


def test_summary_constant_and_temporal_order():
    x=np.tile(np.arange(10.)[:,None],(1,4))
    f=P.summarize(x)
    np.testing.assert_allclose(f[:4],8.1)
    np.testing.assert_allclose(f[4:8],.8)
    np.testing.assert_allclose(f[8:],1.)
    c=P.summarize(np.zeros((10,4)))
    assert np.array_equal(c,np.zeros(12))
    shuffled=x[[0,9,1,8,2,7,3,6,4,5]]
    s=P.summarize(shuffled)
    np.testing.assert_array_equal(f[:8],s[:8])
    assert np.all(s[8:]<f[8:])


def test_balanced_gradient_and_heldout_isolation():
    rng=np.random.default_rng(16);F=rng.normal(size=(8,12));y=np.array([1,1,1,0,0,0,0,0])
    ix=list(range(1,8));a,info=P.train(F,y,ix,time.monotonic()+30)
    assert info['converged']
    X=(F[ix]-a['mean'])/a['sd'];w=P.M.balanced_weights(y[ix])
    assert np.isclose(w[y[ix]==1].sum(),.5) and np.isclose(w[y[ix]==0].sum(),.5)
    theta=np.linspace(-.1,.1,13);_,g=P.objective(theta,X,y[ix],w)
    numeric=[]
    for j in range(13):
        e=np.zeros(13);e[j]=1e-6
        numeric.append((P.objective(theta+e,X,y[ix],w)[0]-P.objective(theta-e,X,y[ix],w)[0])/2e-6)
    np.testing.assert_allclose(g,numeric,atol=1e-7)
    F[0]=1e9;y[0]=0;b,_=P.train(F,y,ix,time.monotonic()+30)
    assert all(np.array_equal(a[k],b[k]) for k in a)


def test_synthetic_bundle_reload_and_research_http(tmp_path):
    from fastapi.testclient import TestClient
    from tools.stage15_api import create_research_app
    rng=np.random.default_rng(17);t=np.arange(12)+.5;z=rng.normal(size=(12,6))
    cls=dict(mean=np.zeros(12),sd=np.ones(12),beta=np.zeros(12),bias=np.asarray(1.))
    loc=dict(mean=np.zeros(8),sd=np.ones(8),beta=np.ones(8)*.1,bias=np.asarray(0.))
    np.savez(tmp_path/'weights.npz',**{kind+'_'+n:v for kind,m in [('classifier',cls),('localizer',loc)] for n,v in m.items()})
    (tmp_path/'artifact_manifest.json').write_bytes(P.M.pack({'weights.npz':P.M.A.fsha(tmp_path/'weights.npz')}))
    a,b=P.load_bundle(tmp_path);before=P.predict(t,z,cls,loc);after=P.predict(t,z,a,b)
    assert before[:2]==after[:2] and np.array_equal(before[2],after[2])
    from tools.research_provider import ModelProvider
    provider=ModelProvider(a,b,{'synthetic':(t,z)},stage=16)
    with TestClient(create_research_app(provider,api_key="fixture-only"),headers={"X-API-Key":"fixture-only"}) as client:
        r=client.post('/research/stage15/predict',json={'sample_id':'synthetic'})
        assert r.status_code==200 and r.json()['production_enabled'] is False
        assert r.json()['candidate_times_sec']==[before[1]]
        assert not {'verdict','type','norm'}&r.json().keys()
        assert client.post('/research/stage15/predict',json={'sample_id':'missing'}).status_code==503
