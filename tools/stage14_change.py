"""One fixed four-dimensional temporal-change MIL head; aggregates only, no appearance vectors."""
import io,json,time
from pathlib import Path
import numpy as np
from tools import stage12_mil as M,stage13_all40 as S
from tools.local_safe import run_guarded

L2=1.


def change_features(u):
    z=u['z']-np.median(u['z'],axis=0)
    delta=np.diff(z,axis=0,prepend=z[:1])
    acceleration=np.diff(delta,axis=0,prepend=delta[:1])
    context=[]
    for i,t in enumerate(u['t']):
        ix=(u['t']<=t)&(u['t']>t-2.)
        context.append(np.linalg.norm(z[i]-z[ix].mean(0)))
    raw=np.column_stack([np.linalg.norm(z,axis=1),np.linalg.norm(delta,axis=1),
                         np.linalg.norm(acceleration,axis=1),context])
    median=np.median(raw,axis=0)
    scale=np.maximum(np.maximum(1.4826*np.median(abs(raw-median),axis=0),.1*raw.std(axis=0)),1e-6)
    return np.clip((raw-median)/scale,-5.,5.)


def objective(theta,bags,y):
    beta,bias=theta[:-1],theta[-1]
    loss=.5*L2*(beta@beta);g=np.r_[L2*beta,0.]
    for x,target,w in zip(bags,y,M.balanced_weights(y)):
        score,attention=M.pooling(x@beta+bias)
        loss+=w*(np.logaddexp(0.,score)-target*score)
        rowg=w*(M.sigmoid(score)-target)*attention
        g+=np.r_[x.T@rowg,rowg.sum()]
    return float(loss),g


def train(features,y,indices,deadline):
    selected=[features[i] for i in indices]
    mean=np.mean([x.mean(0) for x in selected],axis=0)
    sd=np.maximum(np.sqrt(np.mean([((x-mean)**2).mean(0) for x in selected],axis=0)),1e-4)
    bags=[(x-mean)/sd for x in selected]
    theta,info=M.minimise(lambda b:objective(b,bags,[y[i] for i in indices]),5,deadline)
    return {'mean':mean,'sd':sd,'beta':theta[:-1],'bias':np.asarray(theta[-1])},info


def predict(x,t,model):
    logits=((x-model['mean'])/model['sd'])@model['beta']+model['bias']
    return M.pooling(logits)[0],float(t[np.argmax(logits)]),logits


def metrics(y,p,scores=None):
    r=M.metrics(y,p,scores)
    neg=M.metrics(1-np.asarray(y),~np.asarray(p,dtype=bool))
    r['macro_f1']=.5*(r['f1']+neg['f1'])
    return r


def main(_=None):
    start=time.monotonic();out=Path(M.A.lp())/'research_models/stage14_change'
    out.mkdir(mode=0o700,parents=True,exist_ok=True)
    M.save(out/'attempt.json',M.pack({'started_unix':time.time(),'one_fixed_attempt':True}))
    U,inputs,_=S.load();y=[u['y'] for u in U];folds=S.stratified_folds(y)
    for rel in ['tools/stage14_change.py','tests/test_stage14_contract.py','docs/PROTOCOL_STAGE14.md',
                'tools/stage13_all40.py','tools/stage12_mil.py']:
        inputs[str(M.ROOT/rel)]=M.A.fsha(M.ROOT/rel)
    M.save(out/'input_manifest.json',M.pack(inputs))
    F=[change_features(u) for u in U];assert all(x.shape==(len(u['t']),4) and np.isfinite(x).all() for x,u in zip(F,U))
    scores=np.zeros(40);peaks=np.zeros(40);baseline=np.zeros(40,bool);models=[];infos=[]
    for held in folds:
        ix=[i for i in range(40) if i not in held]
        model,info=train(F,y,ix,start+120);models.append(model);infos.append(info)
        for i in held:scores[i],peaks[i],_=predict(F[i],U[i]['t'],model)
        baseline[held]=S.duration_prediction(U,ix,held)
    final,final_info=train(F,y,list(range(40)),start+120)
    for name,arrays in [('weights.npz',final),('cv_weights.npz',{f'fold_{k}_{n}':v for k,m in enumerate(models) for n,v in m.items()})]:
        buffer=io.BytesIO();np.savez(buffer,**arrays);M.save(out/name,buffer.getvalue())
    with np.load(out/'weights.npz',allow_pickle=False) as z:restored=dict(z)
    assert all(np.array_equal(predict(f,u['t'],final)[2],predict(f,u['t'],restored)[2]) for f,u in zip(F,U))
    with np.load(out/'cv_weights.npz',allow_pickle=False) as z:cv=dict(z)
    for k,held in enumerate(folds):
        loaded={n:cv[f'fold_{k}_{n}'] for n in final}
        assert all(predict(F[i],U[i]['t'],loaded)[0]==scores[i] for i in held)
    # Recompute stage13 OOF on the identical fixed folds for a paired descriptive comparison.
    stage13=Path(M.A.lp())/'research_models/stage13_all40'
    with np.load(stage13/'cv_weights.npz',allow_pickle=False) as z:oldcv=dict(z)
    oldscores=np.zeros(40)
    for k,held in enumerate(folds):
        oldmodel={n:oldcv[f'fold_{k}_{n}'] for n in ['projection','mean','sd','beta','bias']}
        for i in held:oldscores[i]=M.inference(U[i],oldmodel)[0]
    pos=[i for i,u in enumerate(U) if u['y']];exact=[i for i,u in enumerate(U) if u['ivs']]
    hit=lambda i,t:any(abs(t-v)<=5 for v in U[i]['endpoints'])
    oldsummary=json.loads((stage13/'summary.json').read_text())
    result={'coverage':{'unique_clips':40,'positive':21,'negative':19,'cached_steps':sum(len(f) for f in F),'feature_dims':4},
            'stage14_oof':metrics(y,scores>=0,scores),'stage13_same_fold_oof':metrics(y,oldscores>=0,oldscores),
            'duration_baseline_same_folds':metrics(y,baseline),'always_positive':metrics(y,np.ones(40)),
            'endpoint_hits_within_5s':sum(hit(i,peaks[i]) for i in pos),'endpoint_positive_clips':len(pos),
            'stage13_endpoint_hits_within_5s':oldsummary['endpoint_hits_within_5s'],
            'random_candidate_endpoint_hit_rate':oldsummary['random_candidate_endpoint_hit_rate'],
            'exact_interval_hits':sum(any(a<=peaks[i]<=b for a,b in U[i]['ivs']) for i in exact),'exact_interval_clips':len(exact),
            'all_cv_converged':all(i['converged'] for i in infos),'final_fit':final_info,
            'final_resubstitution':metrics(y,[predict(f,u['t'],final)[0]>=0 for f,u in zip(F,U)]),
            'reload_exact_final_and_cv':True,'weights_sha256':M.A.fsha(out/'weights.npz'),
            'runtime_s':round(time.monotonic()-start,3),'production_enabled':False,'clean_holdout_remaining':False,
            'independent_validation':False,'outcome':'EXPLORATORY_ONLY_NO_PRODUCTION_GO'}
    config={'method':'four scalar temporal novelty channels, clip-relative robust background, strongly regularized MIL',
            'channels':['residual_norm','step_change_norm','second_difference_norm','trailing2s_deviation_norm'],
            'appearance_vectors_in_head':False,'physical_optical_flow':False,'filename_or_duration_features':False,
            'per_clip_preprocessing':'median/MAD with .1std and1e-6 floors; clip +/-5; offline full clip',
            'fold_scaler':'equal clip mean/std fit on train only, floor1e-4','l2':L2,'threshold':0.,'temperature':1.,
            'auxiliary_frame_supervision':False,'cv':'same stage13 stratified5fold seed13 unique clips',
            'selection':'one fixed method, no tuning','production_enabled':False,'deployment_threshold':None}
    assert all(M.A.fsha(p)==h for p,h in inputs.items())
    M.save(out/'config.json',M.pack(config));M.save(out/'summary.json',M.pack(result))
    M.save(out/'artifact_manifest.json',M.pack({p.name:M.A.fsha(p) for p in out.iterdir() if p.is_file()}))
    print('STAGE14 '+json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(run_guarded(main))
