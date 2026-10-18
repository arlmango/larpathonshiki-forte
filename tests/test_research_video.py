"""New synthetic MOV through real pinned encoder and real saved stage15, no stubs."""
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from tools import stage5_encoder as E
from tools.stage15_two_component import load_bundle,predict
from tools.research_video import make_video_provider
from tools.stage15_api import create_research_app

MODEL=Path(__file__).resolve().parents[1]/'local_private/research_models/stage15_two_component'


def test_new_mov_real_encoder_saved_stage15_http_and_reference(tmp_path):
    if not (MODEL/'weights.npz').exists() or not Path(E.ONNX_PATH).exists():
        pytest.skip('optional local pinned SigLIP and saved stage15 artifacts unavailable')
    assert E.onnx_ok()
    import cv2
    path=tmp_path/'new-unseen.mov'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,240))
    assert writer.isOpened()
    for i in range(50):
        frame=np.full((240,320,3),70,np.uint8)
        cv2.rectangle(frame,(15+3*i,70),(45+3*i,140),(20,120,220),-1)
        writer.write(frame)
    writer.release()
    jobs=tmp_path/'jobs';jobs.mkdir(mode=0o700)
    app=create_research_app(video_predictor=make_video_provider(MODEL),api_key='fixture-only',
                           timeout=120,memory_mb=2048,work_root=jobs)
    with TestClient(app) as client:
        response=client.post('/research/stage15/analyze',content=path.read_bytes(),
            headers={'X-API-Key':'fixture-only','Content-Type':'video/quicktime','Content-Disposition':'attachment; filename="../../untrusted.mov"'})
        assert response.status_code==200,response.json()
        output=response.json()
        assert output['research_only'] and output['human_review_required']
        assert output['production_enabled'] is False and output['status']=='not_validated'
        assert not {'verdict','type','norm','violations','embeddings','frames','path'}&output.keys()
        assert not list(jobs.iterdir())
        corrupt=client.post('/research/stage15/analyze',content=b'not a movie',
            headers={'X-API-Key':'fixture-only','Content-Type':'video/quicktime'})
        assert corrupt.status_code==503 and not list(jobs.iterdir())
    # Independent historical extraction path: no fit, labels, registry or known content hashes.
    cache=tmp_path/'reference.npz'
    result=E.child_embed({'path':str(path),'out':str(cache),'zone':None,'dur_expected':None})
    assert 'err' not in result and result['n_frames']==5
    with np.load(cache,allow_pickle=False) as z:
        t=z['t'].astype(float);raw=z['E_full'].astype(float)
    normalized=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-9)
    cls,loc=load_bundle(MODEL)
    score,peak,_=predict(t,normalized-np.median(normalized,axis=0),cls,loc)
    assert output['uncalibrated_clip_logit']==pytest.approx(score,abs=1e-5)
    assert output['candidate_times_sec']==([peak] if score>=0 else [])


def test_upload_bounds_and_auth_before_body():
    import asyncio
    app=create_research_app(api_key='fixture-only')
    async def check(headers,status):
        sent=[]
        async def receive():pytest.fail('rejected upload body consumed')
        async def send(message):sent.append(message)
        await app({'type':'http','method':'POST','path':'/research/stage15/analyze',
                   'headers':headers,'query_string':b'','http_version':'1.1'},receive,send)
        assert sent[0]['status']==status
    asyncio.run(check([],401))
    asyncio.run(check([(b'x-api-key',b'fixture-only'),(b'content-type',b'text/plain')],415))
    asyncio.run(check([(b'x-api-key',b'fixture-only'),(b'content-type',b'video/quicktime'),
                       (b'content-length',str(60*1048576+1).encode())],413))


def test_admission_rejects_second_upload_before_body_and_releases():
    import asyncio
    app=create_research_app(api_key='fixture-only')
    scope={'type':'http','method':'POST','path':'/research/stage15/analyze','headers':[
        (b'x-api-key',b'fixture-only'),(b'content-type',b'video/quicktime')],'query_string':b'','http_version':'1.1'}
    async def run():
        entered=asyncio.Event();release=asyncio.Event();first=[];second=[]
        async def receive1():entered.set();await release.wait();return {'type':'http.disconnect'}
        async def receive2():pytest.fail('second upload consumed body')
        async def send1(m):first.append(m)
        async def send2(m):second.append(m)
        task=asyncio.create_task(app(scope,receive1,send1));await entered.wait()
        await app(scope,receive2,send2);assert second[0]['status']==429
        release.set();await task
        # Disconnected first request released admission: another rejected mime is 415, not429.
        second.clear();bad={**scope,'headers':[(b'x-api-key',b'fixture-only'),(b'content-type',b'text/plain')]}
        await app(bad,receive2,send2);assert second[0]['status']==415
    asyncio.run(run())


def test_ui_contains_no_secrets_and_health_requires_key(monkeypatch):
    monkeypatch.setenv('RESEARCH_INSTANCE_ID','synthetic-instance')
    with TestClient(create_research_app(api_key='fixture-only')) as client:
        ui=client.get('/')
        assert ui.status_code==200 and 'fixture-only' not in ui.text
        assert 'localStorage' not in ui.text and 'sessionStorage' not in ui.text
        assert client.get('/research/health').status_code==401
        health=client.get('/research/health',headers={'X-API-Key':'fixture-only'})
        assert health.json()=={'research_only':True,'production_enabled':False,'status':'not_validated','instance_id':'synthetic-instance'}
