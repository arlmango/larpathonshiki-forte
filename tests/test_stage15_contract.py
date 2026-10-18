"""Synthetic stage15 leakage, matching and numerical contracts."""
import copy,time
import numpy as np
from tools import stage15_two_component as L,stage12_mil as M


def fixture():
    rng=np.random.default_rng(15);U=[]
    for i in range(8):
        t=np.arange(12)+.5;z=rng.normal(size=(12,6));y=int(i<4)
        U.append({'t':t,'z':z,'y':y,'endpoints':[5.] if y else [],'ivs':[]})
    return U,[L.local_features(u['t'],u['z']) for u in U]


def test_one_to_one_matching_counts_missed_and_mistimed_events():
    assert L.match_one([10,12],[11])==(1,0,1,1.)
    assert L.match_one([10],[15])==(1,0,0,5.)
    assert L.match_one([10],[15.01])[:3]==(0,1,1)
    assert L.match_one([10,12],[])[:3]==(0,0,2)
    assert L.match_one([],[10])[:3]==(0,1,0)


def test_gate_independent_localization_and_negative_false_alarms():
    U=[{'y':1,'endpoints':[10,12]},{'y':1,'endpoints':[20]},{'y':0,'endpoints':[]}]
    r=L.evaluate_events(U,[-1,1,1],[10,40,10])
    assert r['localization_all_positive_clips']==2 and r['localization_hits_within_5s']==1
    assert r['positive_gate_miss_clips']==1 and r['gate_missed_gt_events']==2
    assert r['false_alarm_negative_clips']==1
    assert r['end_to_end_events']==L.event_metrics(0,2,3)


def test_rows_balance_masks_and_gradient():
    U,F=fixture();X,y,w=L.training_rows(F,U,list(range(8)),np.zeros(8),np.ones(8))
    assert np.isclose(w[y==1].sum(),.5) and np.isclose(w[y==0].sum(),.5)
    assert int(y.sum())==16 and int((y==0).sum())==48
    b=np.linspace(-.1,.1,9);f=lambda v:L.objective(v,X,y,w)
    _,g=f(b);numeric=[]
    for j in range(9):
        d=np.zeros(9);d[j]=1e-6;numeric.append((f(b+d)[0]-f(b-d)[0])/2e-6)
    np.testing.assert_allclose(g,numeric,atol=1e-7)


def test_heldout_features_and_vendor_labels_do_not_change_localizer():
    U,F=fixture();ix=list(range(1,8));a,_=L.train(F,U,ix,time.monotonic()+30)
    U2=copy.deepcopy(U);F2=copy.deepcopy(F)
    U2[0]['y']=0;U2[0]['endpoints']=[999];F2[0][:]=1e9
    b,_=L.train(F2,U2,ix,time.monotonic()+30)
    assert all(np.array_equal(a[k],b[k]) for k in a)


def test_synthetic_saved_bundle_provider_and_research_http(tmp_path):
    from tools.stage15_api import create_research_app
    from fastapi.testclient import TestClient
    U,F=fixture();loc,_=L.train(F,U,list(range(8)),time.monotonic()+30)
    cls={'mean':np.zeros(4),'sd':np.ones(4),'beta':np.zeros(4),'bias':np.asarray(1.)}
    arrays={**{'classifier_'+k:v for k,v in cls.items()},**{'localizer_'+k:v for k,v in loc.items()}}
    np.savez(tmp_path/'weights.npz',**arrays)
    (tmp_path/'artifact_manifest.json').write_bytes(M.pack({'weights.npz':M.A.fsha(tmp_path/'weights.npz')}))
    from tools.research_provider import make_model_provider
    provider=make_model_provider(tmp_path,{'synthetic':(U[0]['t'],U[0]['z'])})
    with TestClient(create_research_app(provider,api_key="fixture-only"),headers={"X-API-Key":"fixture-only"}) as client:
        r=client.post('/research/stage15/predict',json={'sample_id':'synthetic'})
        assert r.status_code==200 and r.json()['status']=='not_validated'
        assert r.json()['production_enabled'] is False and len(r.json()['candidate_times_sec'])==1
        assert not {'verdict','violations','type','norm'}&r.json().keys()
        assert client.post('/research/stage15/predict',json={'sample_id':'missing'}).status_code==503
