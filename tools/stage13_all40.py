"""Single all40 exploratory fit. Former holdout is explicitly consumed, not a clean test."""
import io,json,time
from pathlib import Path
import numpy as np
from tools import stage12_mil as M,stage13_extract as X
from tools.local_safe import run_guarded


def stratified_folds(y):
    y=np.asarray(y);rng=np.random.default_rng(13);folds=[[] for _ in range(5)]
    for cls in (0,1):
        indices=np.flatnonzero(y==cls);rng.shuffle(indices)
        for k,part in enumerate(np.array_split(indices,5)):folds[k].extend(int(i) for i in part)
    return [sorted(part) for part in folds]


def load():
    units=X.all_units();base=Path(M.A.lp());path=base/'train_probe/stage13_all40/manifest.json'
    manifest=json.loads(path.read_text());inputs=dict(manifest['inputs']);inputs[str(path)]=M.A.fsha(path)
    M.require(all(M.A.fsha(p)==h for p,h in inputs.items()),'extraction inputs changed')
    pos={u[0] for u in units if u[2]=='reviewed_violation'};neg={u[0] for u in units if u[2]=='reviewed_no_violation'}
    sp,d=M.A.snapshot_spec('C02',None,None);labels,_=M.H.supervision(M.A.read_snapshot(sp,d),pos,neg)
    inputs[str(sp)]=d
    for rel in ['splits/C02.json','labels/C02.imported.jsonl','inventory_map.json']:
        inputs[str(base/rel)]=M.A.fsha(base/rel)
    from tools.import_labels import locate_xlsx
    table,n=locate_xlsx(M.A.lp(),'C02');M.require(n==1,'original table ambiguous');inputs[str(table)]=M.A.fsha(table)
    U=[]
    for sha,source,status,endpoints in units:
        M.require(Path(source).resolve().is_relative_to(Path.home()/'Downloads/open'),'source root differs')
        with np.load(manifest['caches'][sha],allow_pickle=False) as z:
            t=z['t'].astype(float);E=z['E_full'].astype(float);duration=float(z['dur_dec'])
        M.require(E.shape==(len(t),768) and np.isfinite(E).all() and np.all(np.diff(t)>0),'invalid full-view cache')
        normalized=E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-9);median=np.median(normalized,axis=0)
        ivs=[(v['start'],v['end']) for v in labels.get(sha,{}).get('ivs',[])]
        az,at,aw=np.empty((0,768)),np.empty(0),np.empty(0)
        if sha in labels:
            with np.load(manifest['auxiliary'][sha],allow_pickle=False) as z:
                raw=np.concatenate([z['E_pre'],z['E_lab_out']]).astype(float)
                times=np.concatenate([z['t_pre'],z['t_lab_out']]).astype(float)
            mask=M.H.in_any(times,ivs);at=times[mask]
            raw=raw[mask];az=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-9)-median
            aw=np.array([max(v['w'] for v in labels[sha]['ivs'] if v['start']<=tt<=v['end']) for tt in at])
            M.require(len(at)>0 and np.isfinite(az).all(),'missing exact full-view supervision')
        U.append({'y':int(sha in pos),'z':normalized-median,'t':t,'dur':duration,
                  'aux_z':az,'aux_t':at,'aux_w':aw,'ivs':ivs,'endpoints':endpoints})
    M.require(len(U)==40 and sum(u['y'] for u in U)==21,'all40 coverage differs')
    return U,inputs,manifest['extraction_runtime_s']


def duration_prediction(U,indices,held):
    from tools.stage7_core import fit_thr_mid
    ys=[U[i]['y'] for i in indices];dur=np.array([U[i]['dur'] for i in indices]);options=[]
    for sign in [1.,-1.]:
        threshold=fit_thr_mid(sign*dur,ys)
        options.append((M.metrics(ys,sign*dur>=threshold)['f1'],sign,threshold))
    _,sign,threshold=max(options,key=lambda v:v[0])
    return [bool(sign*U[i]['dur']>=threshold) for i in held]


def main(_=None):
    start=time.monotonic();deadline=start+180
    out=Path(M.A.lp())/'research_models/stage13_all40';out.mkdir(parents=True,mode=0o700,exist_ok=True)
    M.save(out/'attempt.json',M.pack({'started_unix':time.time(),'all40_explicitly_authorized':True,'clean_holdout':False}))
    before=M.A.immutables('C02');M.require(before[0]==before[1],'original immutables changed')
    U,inputs,extract_runtime=load()
    for rel in ['tools/stage13_all40.py','tools/stage13_extract.py','tools/stage12_mil.py',
                'tools/stage5_encoder.py','tools/stage11_points.py','tools/stage9_head.py','tools/stage8_audit.py',
                'tools/local_safe.py','tools/import_labels.py','tools/annotate_intervals.py',
                'docs/PROTOCOL_STAGE13.md','tests/test_stage13_contract.py']:
        inputs[str(M.ROOT/rel)]=M.A.fsha(M.ROOT/rel)
    M.save(out/'input_manifest.json',M.pack(inputs))
    y=[u['y'] for u in U];folds=stratified_folds(y)
    coverage={'unique_clips':40,'positive':21,'negative':19,'cached_clips':len(U),
              'cached_rows':sum(len(u['t']) for u in U),'auxiliary_clips':sum(bool(len(u['aux_t'])) for u in U),
              'exact_positive_frames':sum(len(u['aux_t']) for u in U),'view':'full','former_holdout_used':23,
              'clean_holdout_remaining':False,'finite_features':all(np.isfinite(u['z']).all() for u in U)}
    M.save(out/'coverage.json',M.pack(coverage));print('FIT_COVERAGE '+json.dumps(coverage),flush=True)
    scores=np.zeros(40);peaks=np.zeros(40);baseline=np.zeros(40,bool);models=[];info=[];fold_sizes=[]
    for held in folds:
        indices=[i for i in range(40) if i not in held]
        model,details=M.train(U,indices,deadline);models.append(model);info.append(details)
        for i in held:scores[i],peaks[i],_=M.inference(U[i],model)
        baseline[held]=duration_prediction(U,indices,held)
        fold_sizes.append({'positive':sum(y[i] for i in held),'negative':sum(1-y[i] for i in held)})
    final,final_info=M.train(U,list(range(40)),deadline)
    for filename,arrays in [('weights.npz',final),('cv_weights.npz',{f'fold_{i}_{k}':v for i,m in enumerate(models) for k,v in m.items()})]:
        b=io.BytesIO();np.savez(b,**arrays);M.save(out/filename,b.getvalue())
    with np.load(out/'weights.npz',allow_pickle=False) as z:restored=dict(z)
    M.require(all(np.array_equal(M.inference(u,final)[2],M.inference(u,restored)[2]) for u in U),'reload differs')
    with np.load(out/'cv_weights.npz',allow_pickle=False) as z: cv=dict(z)
    reload_scores=np.zeros(40)
    for k,held in enumerate(folds):
        restored_fold={n:cv[f'fold_{k}_{n}'] for n in final}
        for i in held:reload_scores[i]=M.inference(U[i],restored_fold)[0]
    M.require(np.array_equal(scores,reload_scores),'CV reload differs')
    pos=[i for i,u in enumerate(U) if u['y']];exact=[i for i,u in enumerate(U) if u['ivs']]
    hit=lambda i,t:any(abs(t-v)<=5 for v in U[i]['endpoints'])
    result={'coverage':coverage,'folds':5,'fold_class_counts':fold_sizes,'all_cv_fits_converged':all(v['converged'] for v in info),
            'clip_oof':M.metrics(y,scores>=0,scores),'always_positive':M.metrics(y,np.ones(40)),
            'always_negative':M.metrics(y,np.zeros(40)),'duration_only_same_folds':M.metrics(y,baseline),
            'endpoint_hits_within_5s':sum(hit(i,peaks[i]) for i in pos),'endpoint_positive_clips':len(pos),
            'random_candidate_endpoint_hit_rate':float(np.mean([np.mean([hit(i,t) for t in U[i]['t']]) for i in pos])),
            'exact_interval_hits':sum(any(a<=peaks[i]<=b for a,b in U[i]['ivs']) for i in exact),'exact_interval_clips':len(exact),
            'exact_candidate_coverage_clips':sum(M.H.in_any(U[i]['t'],U[i]['ivs']).any().item() for i in exact),
            'final_fit':final_info,'final_fit_resubstitution':M.metrics(y,[M.inference(u,final)[0]>=0 for u in U]),
            'weights_sha256':M.A.fsha(out/'weights.npz'),'reload_exact_final_and_cv':True,
            'extraction_runtime_s':round(extract_runtime,3),'fit_evaluation_runtime_s':round(time.monotonic()-start,3),
            'outcome':'EXPLORATORY_ONLY_NO_PRODUCTION_GO','production_enabled':False,
            'former_holdout_consumed':True,'clean_holdout_remaining':False,'independent_validation':False}
    config={'head_parameters':'unchanged stage12','view':'full for all40, including exact auxiliary frames',
            'pca_dims':M.DIM,'l2':M.L2,'auxiliary_weight':M.AUX,'smoothness':M.SMOOTH,'temperature':1.,
            'threshold':0.,'deployment_threshold':None,'projection_scaler':'training-fold clips only, equal clip mass',
            'cv':'5fold stratified by clip label; unique content SHA units; fixed seed13; no tuning',
            'encoder':'same pinned SigLIP2 and stage5 preprocess; frozen','production_enabled':False,
            'disclosure':'old17train+23formerholdout authorized; all evaluation exploratory/adaptive; no clean holdout',
            'unknown_interval_positive_policy':'positive bags, no invented frame negatives'}
    M.require(all(M.A.fsha(p)==h for p,h in inputs.items()),'inputs changed during run')
    M.require(M.A.immutables('C02')==before,'old immutables changed')
    M.save(out/'config.json',M.pack(config));M.save(out/'summary.json',M.pack(result))
    M.save(out/'artifact_manifest.json',M.pack({p.name:M.A.fsha(p) for p in out.iterdir() if p.is_file()}))
    print('STAGE13 '+json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(run_guarded(main))
