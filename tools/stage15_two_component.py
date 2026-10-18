"""One fixed localizer fit + frozen stage14 classifier. Local data, aggregate stdout only."""
import io,json,time
from pathlib import Path
import numpy as np
from tools import stage12_mil as M,stage13_all40 as S,stage14_change as C
from tools.local_safe import run_guarded

WINDOW=2.
TOLERANCE=5.
L2=1.


def local_features(t,z):
    base=C.change_features({'t':t,'z':z})
    history=np.array([base[(t<=tt)&(t>tt-3.)].mean(axis=0) for tt in t])
    return np.concatenate([base,history],axis=1)


def training_rows(F,U,indices,mean,sd):
    selected=[U[i] for i in indices]
    counts={y:sum(u['y']==y for u in selected) for y in (0,1)}
    M.require(all(counts.values()),'training class absent')
    rows=[];labels=[];weights=[]
    for i in indices:
        u=U[i]
        if u['y']:
            times=np.asarray(u['endpoints'],float)
            M.require(len(times)>0,'positive vendor timestamps missing')
            mask=np.any(abs(u['t'][:,None]-times[None,:])<=WINDOW,axis=1)
        else:
            mask=np.ones(len(u['t']),bool)
        M.require(mask.any(),'training clip has no supervised rows')
        X=(F[i][mask]-mean)/sd
        rows.append(X);labels.extend([u['y']]*len(X))
        weights.extend([.5/counts[u['y']]/len(X)]*len(X))
    return np.vstack(rows),np.array(labels),np.array(weights)


def objective(theta,X,y,w):
    beta,bias=theta[:-1],theta[-1];logits=X@beta+bias
    loss=float(w@(np.logaddexp(0.,logits)-y*logits)+.5*L2*(beta@beta))
    residual=w*(M.sigmoid(logits)-y)
    return loss,np.r_[X.T@residual+L2*beta,residual.sum()]


def train(F,U,indices,deadline):
    selected=[F[i] for i in indices]
    mean=np.mean([f.mean(0) for f in selected],axis=0)
    sd=np.maximum(np.sqrt(np.mean([((f-mean)**2).mean(0) for f in selected],axis=0)),1e-4)
    X,y,w=training_rows(F,U,indices,mean,sd)
    theta,info=M.minimise(lambda b:objective(b,X,y,w),9,deadline)
    info.update(positive_rows=int(y.sum()),negative_rows=int((y==0).sum()),
                positive_weight=float(w[y==1].sum()),negative_weight=float(w[y==0].sum()))
    return {'mean':mean,'sd':sd,'beta':theta[:-1],'bias':np.asarray(theta[-1])},info


def predict(t,z,classifier,localizer):
    """Inference receives no labels, intervals, IDs, duration or ground-truth times."""
    t=np.asarray(t,float);z=np.asarray(z,float)
    M.require(t.ndim==1 and z.ndim==2 and len(t)==len(z) and len(t)>=3,'invalid research input')
    M.require(np.isfinite(t).all() and np.isfinite(z).all() and np.all(np.diff(t)>0) and t[0]>=0,'invalid research input')
    clip_score,_,_=C.predict(C.change_features({'t':t,'z':z}),t,classifier)
    f=local_features(t,z);logits=((f-localizer['mean'])/localizer['sd'])@localizer['beta']+localizer['bias']
    M.require(np.isfinite(logits).all() and np.isfinite(clip_score),'nonfinite research prediction')
    return float(clip_score),float(t[int(np.argmax(logits))]),logits


def match_one(gt,pred):
    """At most one prediction; nearest GT, ties earlier. -> TP, FP, FN, error or None."""
    gt=sorted(set(float(t) for t in gt));pred=list(pred)
    M.require(len(pred)<=1,'multiple predictions violate fixed contract')
    if not pred:return 0,0,len(gt),None
    valid=[(abs(t-pred[0]),t) for t in gt if abs(t-pred[0])<=TOLERANCE]
    if valid:return 1,0,len(gt)-1,float(min(valid)[0])
    return 0,1,len(gt),None


def event_metrics(tp,fp,fn):
    return {'tp':int(tp),'fp':int(fp),'fn':int(fn),'precision':tp/max(1,tp+fp),
            'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn)}


def evaluate_events(U,scores,peaks):
    totals=np.zeros(3,int);ungated=np.zeros(3,int);nearest=[];allhit=0;negative_fp=0;gate_clips=gate_events=0;open_misses=0
    for i,u in enumerate(U):
        gt=sorted(set(u['endpoints'])) if u['y'] else []
        emitted=[peaks[i]] if scores[i]>=0 else []
        tp,fp,fn,_=match_one(gt,emitted);totals+=np.array([tp,fp,fn])
        if u['y']:
            a,b,c,_=match_one(gt,[peaks[i]]);ungated+=np.array([a,b,c]);allhit+=a
            nearest.append(min(abs(peaks[i]-t) for t in gt))
            if scores[i]<0:gate_clips+=1;gate_events+=len(gt)
            elif not a:open_misses+=1
        elif emitted:negative_fp+=1
    positives=sum(u['y'] for u in U)
    return {'end_to_end_events':event_metrics(*totals),'all_positive_ungated_events':event_metrics(*ungated),
            'localization_all_positive_clips':positives,'localization_hits_within_5s':int(allhit),
            'localization_hit_rate_all_positive':allhit/positives,
            'localization_nearest_error_mean_sec_all_positive':float(np.mean(nearest)),
            'localization_nearest_error_median_sec_all_positive':float(np.median(nearest)),
            'positive_gate_miss_clips':gate_clips,'gate_missed_gt_events':gate_events,
            'localization_miss_clips_among_gate_open_positives':open_misses,
            'false_alarm_negative_clips':negative_fp,'negative_clips':len(U)-positives,
            'false_alarm_rate_negative_clips':negative_fp/(len(U)-positives),
            'emitted_candidates':int((np.asarray(scores)>=0).sum())}


def load_bundle(model_dir):
    base=Path(model_dir);manifest=json.loads((base/'artifact_manifest.json').read_text())
    M.require(M.A.fsha(base/'weights.npz')==manifest['weights.npz'],'model integrity failed')
    with np.load(base/'weights.npz',allow_pickle=False) as z:arrays=dict(z)
    return tuple({n:arrays[prefix+n] for n in ['mean','sd','beta','bias']} for prefix in ['classifier_','localizer_'])


def research_predictor(model_dir,samples):
    """Local in-memory registry maps opaque IDs to (timestamps, centered embeddings); no HTTP paths/data."""
    classifier,localizer=load_bundle(model_dir)
    def provider(sample_id):
        t,z=samples[sample_id]
        score,peak,_=predict(t,z,classifier,localizer)
        return {'clip_logit':score,'candidate_time_sec':peak}
    return provider


def main(_=None):
    start=time.monotonic();deadline=start+180
    out=Path(M.A.lp())/'research_models/stage15_two_component';out.mkdir(mode=0o700,parents=True,exist_ok=True)
    M.save(out/'attempt.json',M.pack({'started_unix':time.time(),'single_localizer_attempt':True,'classifier_refit':False}))
    U,inputs,_=S.load();y=[u['y'] for u in U];folds=S.stratified_folds(y);F=[local_features(u['t'],u['z']) for u in U]
    M.require(len(U)==40 and sum(y)==21 and sum(len(set(u['endpoints'])) for u in U if u['y'])==22,'coverage differs')
    coverage={'unique_clips':40,'positive_clips':21,'negative_clips':19,'vendor_gt_events':22,'candidate_rows':sum(len(f) for f in F),'localizer_dims':8}
    M.save(out/'coverage.json',M.pack(coverage))
    old=Path(M.A.lp())/'research_models/stage14_change'
    oldmanifest=json.loads((old/'artifact_manifest.json').read_text())
    M.require(all(M.A.fsha(old/n)==h for n,h in oldmanifest.items()),'stage14 integrity failed')
    with np.load(old/'weights.npz',allow_pickle=False) as z:classifier=dict(z)
    with np.load(old/'cv_weights.npz',allow_pickle=False) as z:oldcv=dict(z)
    for name in oldmanifest:inputs[str(old/name)]=M.A.fsha(old/name)
    for rel in ['tools/stage15_two_component.py','tools/stage15_api.py','tests/test_stage15_contract.py','tests/test_stage15_api.py',
                'docs/PROTOCOL_STAGE15.md','tools/stage14_change.py','tools/stage13_all40.py','tools/stage13_extract.py',
                'tools/stage12_mil.py','tools/stage8_audit.py','tools/stage9_head.py','tools/local_safe.py',
                'tools/import_labels.py','tools/annotate.py','tools/annotate_intervals.py','tools/evaluate.py']:
        inputs[str(M.ROOT/rel)]=M.A.fsha(M.ROOT/rel)
    M.save(out/'input_manifest.json',M.pack(inputs));print('STAGE15_COVERAGE '+json.dumps(coverage),flush=True)
    scores=np.zeros(40);peaks=np.zeros(40);oldpeaks=np.zeros(40);duration=np.zeros(40,bool);localizers=[];fitinfo=[];classifiers=[]
    for k,held in enumerate(folds):
        ix=[i for i in range(40) if i not in held]
        cls={n:oldcv[f'fold_{k}_{n}'] for n in classifier};classifiers.append(cls)
        loc,info=train(F,U,ix,deadline);localizers.append(loc);fitinfo.append(info)
        for i in held:
            scores[i],peaks[i],_=predict(U[i]['t'],U[i]['z'],cls,loc)
            oldscore,oldpeaks[i],_=C.predict(C.change_features(U[i]),U[i]['t'],cls)
            M.require(scores[i]==oldscore,'frozen classifier score changed')
        duration[held]=S.duration_prediction(U,ix,held)
    final,final_info=train(F,U,list(range(40)),deadline)
    final_arrays={**{'classifier_'+k:v for k,v in classifier.items()},**{'localizer_'+k:v for k,v in final.items()}}
    cv_arrays={}
    for k,(cls,loc) in enumerate(zip(classifiers,localizers)):
        cv_arrays.update({f'fold_{k}_classifier_{n}':v for n,v in cls.items()});cv_arrays.update({f'fold_{k}_localizer_{n}':v for n,v in loc.items()})
    for name,arrays in [('weights.npz',final_arrays),('cv_weights.npz',cv_arrays)]:
        buffer=io.BytesIO();np.savez(buffer,**arrays);M.save(out/name,buffer.getvalue())
    with np.load(out/'weights.npz',allow_pickle=False) as z:restored=dict(z)
    cls2={n:restored['classifier_'+n] for n in classifier};loc2={n:restored['localizer_'+n] for n in final}
    M.require(all(np.array_equal(classifier[n],cls2[n]) for n in classifier),'classifier arrays changed')
    M.require(all(np.array_equal(predict(u['t'],u['z'],classifier,final)[2],predict(u['t'],u['z'],cls2,loc2)[2]) for u in U),'final reload differs')
    with np.load(out/'cv_weights.npz',allow_pickle=False) as z:restoredcv=dict(z)
    for k,held in enumerate(folds):
        cls={n:restoredcv[f'fold_{k}_classifier_{n}'] for n in classifier};loc={n:restoredcv[f'fold_{k}_localizer_{n}'] for n in final}
        for i in held:
            sc,pk,logits=predict(U[i]['t'],U[i]['z'],cls,loc)
            M.require(sc==scores[i] and pk==peaks[i] and np.array_equal(logits,predict(U[i]['t'],U[i]['z'],classifiers[k],localizers[k])[2]),'CV reload differs')
    clip=C.metrics(y,scores>=0,scores)
    M.require(clip==json.loads((old/'summary.json').read_text())['stage14_oof'],'frozen clip metrics differ')
    metrics=evaluate_events(U,scores,peaks);baseline=evaluate_events(U,scores,oldpeaks)
    exact=[i for i,u in enumerate(U) if u['ivs']]
    result={'coverage':coverage,'clip_oof_unchanged_stage14':clip,'stage15':metrics,'stage14_original_localizer_baseline':baseline,
            'duration_clip_baseline_same_folds':C.metrics(y,duration),
            'exact_interval_hits_all_positive':sum(any(a<=peaks[i]<=b for a,b in U[i]['ivs']) for i in exact),'exact_interval_clips':len(exact),
            'folds':5,'all_localizer_fits_converged':all(i['converged'] for i in fitinfo),'fold_training_aggregate':fitinfo,
            'final_localizer_fit':final_info,'classifier_refit':False,'classifier_arrays_unchanged':True,'final_and_cv_reload_exact':True,
            'weights_sha256':M.A.fsha(out/'weights.npz'),'runtime_s':round(time.monotonic()-start,3),
            'clean_holdout_remaining':False,'independent_validation':False,'production_enabled':False,'outcome':'EXPLORATORY_NO_GO'}
    config={'classifier':'saved frozen stage14, same fold and final arrays','classifier_sha256':M.A.fsha(old/'weights.npz'),
            'classifier_gate':0.,'localizer':'linear eight-dimensional change+trailing3s model','localizer_l2':L2,
            'positive_training_window_seconds':[-WINDOW,WINDOW],'weak_negative_policy':'all vendor-negative frames only',
            'unlabeled_positive_frames':'excluded from supervised loss','supervision_balance':'0.5 each class; equal clip then supervised row mass',
            'candidate_policy':'earliest argmax over original1fps cache, independent of labels','emission':'at most1 if frozen clip logit>=0',
            'matching_tolerance_seconds':TOLERANCE,'matching':'nearest one-to-one within clip; tie earlier GT; all unmatched GT count FN',
            'folds':'same stratified5fold seed13 as stage13/14','scaling':'train-fold equal-clip mean/std, floor1e-4',
            'selection':'one pre-fixed method; no threshold/grid search','production_enabled':False,'deployment_threshold':None}
    M.require(all(M.A.fsha(p)==h for p,h in inputs.items()),'inputs changed')
    M.save(out/'config.json',M.pack(config));M.save(out/'summary.json',M.pack(result))
    M.save(out/'artifact_manifest.json',M.pack({p.name:M.A.fsha(p) for p in out.iterdir() if p.is_file()}))
    print('STAGE15 '+json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(run_guarded(main))
