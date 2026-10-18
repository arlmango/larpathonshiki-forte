"""Explicit all40 authorisation: local full-view extraction, aggregate output only."""
import json,time,os
from pathlib import Path
from tools import stage12_mil as M, stage5_encoder as E, stage11_points as P
from tools.local_safe import select_split_units,run_guarded


def all_units():
    result=[]
    for side in ['train','holdout']:
        units,counts=select_split_units(M.A.lp(),'C02',side,dry_run=False)
        M.require(not any(counts.values()),'source verification failed')
        result.extend(units)
    result.sort(key=lambda u:u[0])
    M.require(len(result)==len({u[0] for u in result})==40,'unique coverage failed')
    M.require(sum(u[2]=='reviewed_violation' for u in result)==21,'positive coverage failed')
    M.require(sum(u[2]=='reviewed_no_violation' for u in result)==19,'negative coverage failed')
    return result


def main(_=None):
    start=time.time();deadline=1792302960. # 2026-10-18 05:56 UTC, checked below with datetime
    from datetime import datetime,timezone
    deadline=datetime(2026,10,18,5,56,tzinfo=timezone.utc).timestamp()
    out=Path(M.A.lp())/'train_probe/stage13_all40'
    out.mkdir(mode=0o700,parents=True,exist_ok=True)
    M.save(out/'attempt.json',M.pack({'started_unix':start,'former_holdout_authorized':True}))
    units=all_units();inputs={};caches={};missing=[]
    for sha,source,st,ev in units:
        inputs[str(source)]=M.A.fsha(source)
        existing=Path(M.A.lp())/'train_probe/cache_emb_siglip2'/(sha+'.npz')
        if existing.exists():caches[sha]=str(existing);inputs[str(existing)]=M.A.fsha(existing)
        else:missing.append((sha,source))
    print('ALL40_COVERAGE '+json.dumps({'unique':40,'positive':21,'negative':19,'reused':len(caches),'extract_missing':len(missing),'view':'full','clean_holdout_remaining':False}),flush=True)
    spent=[]
    for sha,source in missing:
        remaining=deadline-time.time()
        M.require(remaining>30,'extraction deadline insufficient')
        dest=out/(sha+'.npz');M.require(not dest.exists(),'new cache already exists')
        r=E._spawn(E.MAIN_PY,'embed',{'path':source,'out':str(dest),'zone':None,'dur_expected':None},min(120,remaining-10))
        M.require(r.get('status')=='ok' and 'err' not in r and dest.exists(),'embedding failed')
        os.chmod(dest,0o400);caches[sha]=str(dest);inputs[str(dest)]=M.A.fsha(dest);spent.append(r['wall_s'])
        print('EXTRACT_PROGRESS '+json.dumps({'new_completed':len(spent),'new_total':len(missing),'elapsed_s':round(time.time()-start,2),'mean_clip_s':round(sum(spent)/len(spent),2),'estimated_remaining_s':round(sum(spent)/len(spent)*(len(missing)-len(spent)),2)}),flush=True)
    pos={u[0] for u in units if u[2]=='reviewed_violation'};neg={u[0] for u in units if u[2]=='reviewed_no_violation'}
    sp,d=M.A.snapshot_spec('C02',None,None);labels,_=M.H.supervision(M.A.read_snapshot(sp,d),pos,neg)
    auxiliary={}
    for sha,source,st,ev in units:
        if sha not in labels:continue
        dest=out/(sha+'.aux.npz');M.require(not dest.exists(),'aux cache already exists')
        ivs=[(v['start'],v['end']) for v in labels[sha]['ivs']]
        r=P.run_child({'path':source,'out':str(dest),'zone':None,'ivs':ivs},min(120,max(1,deadline-time.time())))
        M.require('err' not in r and dest.exists(),'aux embedding failed')
        os.chmod(dest,0o400);auxiliary[sha]=str(dest);inputs[str(dest)]=M.A.fsha(dest)
    M.save(out/'manifest.json',M.pack({'caches':caches,'auxiliary':auxiliary,'inputs':inputs,'view':'full','extraction_runtime_s':time.time()-start}))
    print('EXTRACTION_COMPLETE '+json.dumps({'unique_caches':len(caches),'auxiliary_clips':len(auxiliary),'runtime_s':round(time.time()-start,2)}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(run_guarded(main))
