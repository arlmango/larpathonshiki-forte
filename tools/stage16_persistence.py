"""Fixed 12-dimensional distribution/persistence classifier, frozen stage15 localizer."""
import io,json,time
from pathlib import Path
import numpy as np
from tools import stage15_two_component as L
from tools.local_safe import run_guarded
M,S,C=L.M,L.S,L.C
L2=1.
NAMES=('mean','sd','beta','bias')


def summarize(f):
    f=np.asarray(f,float)
    M.require(f.ndim==2 and f.shape[1]==4 and len(f)>=3 and np.isfinite(f).all(),'invalid change features')
    a=f[:-1]-f[:-1].mean(0);b=f[1:]-f[1:].mean(0)
    den=np.sqrt((a*a).sum(0)*(b*b).sum(0))
    corr=np.divide((a*b).sum(0),den,out=np.zeros(4),where=den>1e-12)
    return np.r_[np.quantile(f,.9,axis=0,method='linear'),(f>1.).mean(0),np.clip(corr,-1,1)]


def objective(theta,X,y,w):
    beta,bias=theta[:-1],theta[-1];s=X@beta+bias
    r=w*(M.sigmoid(s)-y)
    return float(w@(np.logaddexp(0,s)-y*s)+.5*L2*(beta@beta)),np.r_[X.T@r+L2*beta,r.sum()]


def train(F,y,ix,deadline):
    selected=np.asarray(F)[ix];target=np.asarray(y)[ix]
    M.require(len(set(target))==2,'training class absent')
    mean=selected.mean(0);sd=np.maximum(selected.std(0),1e-4)
    X=(selected-mean)/sd;w=M.balanced_weights(target)
    theta,info=M.minimise(lambda v:objective(v,X,target,w),13,deadline)
    return dict(mean=mean,sd=sd,beta=theta[:-1],bias=np.asarray(theta[-1])),info


def score(f,model):
    return float(((f-model['mean'])/model['sd'])@model['beta']+model['bias'])


def predict(t,z,classifier,localizer):
    t=np.asarray(t,float);z=np.asarray(z,float)
    M.require(t.ndim==1 and z.ndim==2 and len(t)==len(z) and len(t)>=3,'invalid research input')
    M.require(np.isfinite(t).all() and np.isfinite(z).all() and np.all(np.diff(t)>0) and t[0]>=0,'invalid research input')
    sc=score(summarize(C.change_features({'t':t,'z':z})),classifier)
    f=L.local_features(t,z)
    logits=((f-localizer['mean'])/localizer['sd'])@localizer['beta']+localizer['bias']
    M.require(np.isfinite(sc) and np.isfinite(logits).all(),'nonfinite research prediction')
    return sc,float(t[np.argmax(logits)]),logits


def load_bundle(model_dir):
    return L.load_bundle(model_dir)


def research_predictor(model_dir,samples):
    classifier,localizer=load_bundle(model_dir)
    def provider(sample_id):
        t,z=samples[sample_id];sc,pk,_=predict(t,z,classifier,localizer)
        return {'clip_logit':sc,'candidate_time_sec':pk}
    return provider


def main(_=None):
    start=time.monotonic();out=Path(M.A.lp())/'research_models/stage16_persistence'
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    M.save(out/'attempt.json',M.pack({'started_unix':time.time(),'one_fixed_attempt':True,'budget_seconds':120}))
    U,inputs,_=S.load();y=np.array([u['y'] for u in U]);folds=S.stratified_folds(y)
    M.require(len(U)==40 and y.sum()==21 and sum(len(set(u['endpoints'])) for u in U if u['y'])==22,'coverage differs')
    F=np.array([summarize(C.change_features(u)) for u in U])
    coverage={'unique_clips':40,'positive_clips':21,'negative_clips':19,'vendor_gt_events':22,'classifier_dims':12,'candidate_rows':sum(len(u['t']) for u in U)}
    old=out.parent/'stage15_two_component';om=json.loads((old/'artifact_manifest.json').read_text())
    M.require(all(M.A.fsha(old/n)==h for n,h in om.items()),'stage15 integrity failed')
    for n in om:inputs[str(old/n)]=M.A.fsha(old/n)
    with np.load(old/'weights.npz',allow_pickle=False) as z:oldfinal=dict(z)
    with np.load(old/'cv_weights.npz',allow_pickle=False) as z:oldcv=dict(z)
    for rel in ['tools/stage16_persistence.py','tests/test_stage16_contract.py','docs/PROTOCOL_STAGE16.md',
                'tools/stage15_two_component.py','tools/stage15_api.py','tools/stage14_change.py','tools/stage13_all40.py',
                'tools/stage13_extract.py','tools/stage12_mil.py','tools/stage8_audit.py','tools/stage9_head.py',
                'tools/local_safe.py','tools/evaluate.py','tools/import_labels.py','tools/annotate.py','tools/annotate_intervals.py']:
        inputs[str(M.ROOT/rel)]=M.A.fsha(M.ROOT/rel)
    M.save(out/'input_manifest.json',M.pack(inputs));M.save(out/'coverage.json',M.pack(coverage))
    print('STAGE16_COVERAGE '+json.dumps(coverage),flush=True)
    scores=np.zeros(40);peaks=np.zeros(40);oldscore=np.zeros(40);duration=np.zeros(40,bool);models=[];locals_=[];infos=[]
    for k,held in enumerate(folds):
        ix=[i for i in range(40) if i not in held]
        cls,info=train(F,y,ix,start+120);models.append(cls);infos.append(info)
        loc={n:oldcv[f'fold_{k}_localizer_{n}'] for n in NAMES};locals_.append(loc)
        oldcls={n:oldcv[f'fold_{k}_classifier_{n}'] for n in NAMES}
        for i in held:
            scores[i],peaks[i],_=predict(U[i]['t'],U[i]['z'],cls,loc)
            oldscore[i],pk,_=L.predict(U[i]['t'],U[i]['z'],oldcls,loc)
            M.require(pk==peaks[i],'frozen localizer changed')
        duration[held]=S.duration_prediction(U,ix,held)
    final,finalinfo=train(F,y,list(range(40)),start+120)
    locfinal={n:oldfinal['localizer_'+n] for n in NAMES}
    arrays={**{'classifier_'+n:v for n,v in final.items()},**{'localizer_'+n:v for n,v in locfinal.items()}}
    cvarrays={f'fold_{k}_{kind}_{n}':v for k in range(5) for kind,model in [('classifier',models[k]),('localizer',locals_[k])] for n,v in model.items()}
    for name,arr in [('weights.npz',arrays),('cv_weights.npz',cvarrays)]:
        b=io.BytesIO();np.savez(b,**arr);M.save(out/name,b.getvalue())
    with np.load(out/'weights.npz',allow_pickle=False) as z:restored=dict(z)
    cls2={n:restored['classifier_'+n] for n in NAMES};loc2={n:restored['localizer_'+n] for n in NAMES}
    for u in U:
        a=predict(u['t'],u['z'],final,locfinal);b=predict(u['t'],u['z'],cls2,loc2)
        M.require(a[:2]==b[:2] and np.array_equal(a[2],b[2]),'final reload differs')
    with np.load(out/'cv_weights.npz',allow_pickle=False) as z:cv=dict(z)
    for k,held in enumerate(folds):
        cls={n:cv[f'fold_{k}_classifier_{n}'] for n in NAMES};loc={n:cv[f'fold_{k}_localizer_{n}'] for n in NAMES}
        M.require(all(np.array_equal(loc[n],locals_[k][n]) for n in NAMES),'localizer reload differs')
        for i in held:
            sc,pk,_=predict(U[i]['t'],U[i]['z'],cls,loc)
            M.require(sc==scores[i] and pk==peaks[i],'CV reload differs')
    prior=json.loads((old/'summary.json').read_text())
    oldmetrics=C.metrics(y,oldscore>=0,oldscore);oldevents=L.evaluate_events(U,oldscore,peaks)
    M.require(oldmetrics==prior['clip_oof_unchanged_stage14'] and oldevents==prior['stage15'],'stage15 comparison differs')
    events=L.evaluate_events(U,scores,peaks);clip=C.metrics(y,scores>=0,scores)
    exact=[i for i,u in enumerate(U) if u['ivs']]
    result={'coverage':coverage,'stage16_clip_oof':clip,'stage16_events':events,'stage15_clip_oof':oldmetrics,'stage15_events':oldevents,
            'duration_clip_baseline':C.metrics(y,duration),'always_positive_clip_baseline':C.metrics(y,np.ones(40,bool)),
            'always_open_gate_event_baseline':L.evaluate_events(U,np.ones(40),peaks),
            'exact_interval_hits':sum(any(a<=peaks[i]<=b for a,b in U[i]['ivs']) for i in exact),'exact_interval_clips':len(exact),
            'all_six_fits_converged':all(i['converged'] for i in infos+[finalinfo]),'cv_fit_info':infos,'final_fit_info':finalinfo,
            'localizers_frozen':True,'final_and_cv_reload_exact':True,'runtime_s':round(time.monotonic()-start,3),
            'weights_sha256':M.A.fsha(out/'weights.npz'),'clean_holdout_remaining':False,'production_enabled':False,
            'clip_f1_delta_vs_stage15':clip['f1']-oldmetrics['f1'],'event_f1_delta_vs_stage15':events['end_to_end_events']['f1']-oldevents['end_to_end_events']['f1'],
            'outcome':'EXPLORATORY_NO_GO_NO_AUTOMATIC_PROMOTION'}
    config={'method':'12D distribution/persistence logistic classifier','features':['q90 per channel','fraction >1 per channel','lag1 Pearson per channel'],
            'base_channels':'stage14 label-free change channels','l2':1.,'threshold':0.,'class_balance':'0.5 per class, equal clips',
            'scaler':'training clips only; mean/population std, floor1e-4','folds':'same stage13/14/15 stratified5fold seed13',
            'localizer':'frozen corresponding stage15 fold; frozen final for final model','selection':'none; single fixed experiment',
            'matching':'stage15 one-to-one nearest GT within5s, earlier tie, max1 candidate','production_enabled':False,'deployment_threshold':None}
    M.require(all(M.A.fsha(p)==h for p,h in inputs.items()),'inputs changed')
    M.save(out/'config.json',M.pack(config));M.save(out/'summary.json',M.pack(result))
    M.save(out/'artifact_manifest.json',M.pack({p.name:M.A.fsha(p) for p in out.iterdir() if p.is_file()}))
    print('STAGE16 '+json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(run_guarded(main))
