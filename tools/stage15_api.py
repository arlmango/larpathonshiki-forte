"""Authenticated bounded research adapter; separate from the production application."""
import asyncio,json,os,time
from contextlib import asynccontextmanager
from typing import Annotated,Literal
from fastapi import FastAPI,HTTPException,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse,FileResponse
from pathlib import Path
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel,ConfigDict,Field
from app.auth import key_matches
from tools.research_runtime import ResearchRunner,JobTimeout,JobBusy,JobFailure

HEADERS={'Cache-Control':'no-store','Pragma':'no-cache','X-Content-Type-Options':'nosniff'}


class ResearchRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    sample_id:Annotated[str,Field(min_length=1,max_length=64,pattern=r'^[A-Za-z0-9_-]+$')]


class ResearchResponse(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)
    research_only:Literal[True]=True
    production_enabled:Literal[False]=False
    human_review_required:Literal[True]=True
    status:Literal['not_validated']='not_validated'
    uncalibrated_clip_logit:float
    candidate_times_sec:Annotated[list[Annotated[float,Field(ge=0)]],Field(max_length=1)]


class ResearchGuard:
    """Shared auth comparison BEFORE receive; bounded JSON body, no key/path logging."""
    def __init__(self,app,key):self.app=app;self.key=key;self.inflight=0

    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        if scope['method']=='GET' and scope['path']=='/':return await self.app(scope,receive,send)
        keys=[v for k,v in scope['headers'] if k.lower()==b'x-api-key']
        async def reply(code,detail):
            await JSONResponse({'detail':detail},status_code=code,headers=HEADERS)(scope,receive,send)
        if len(keys)!=1 or not key_matches(keys[0],self.key):return await reply(401,'invalid api key')
        if self.inflight:return await reply(429,'research worker busy')
        self.inflight+=1
        try:
            upload=scope['path']=='/research/stage15/analyze' and scope['method']=='POST'
            cap=60*1048576 if upload else 1024
            if upload:
                mime=dict(scope['headers']).get(b'content-type',b'').split(b';')[0].lower()
                if mime not in (b'video/quicktime',b'video/mp4',b'application/octet-stream'):
                    return await reply(415,'unsupported media type')
            lengths=[v for k,v in scope['headers'] if k.lower()==b'content-length']
            try:
                if len(lengths)>1:raise ValueError()
                length=int(lengths[0]) if lengths else 0
                if length<0:raise ValueError()
            except ValueError:return await reply(400,'invalid request')
            if length>cap:return await reply(413,'request too large')
            body=bytearray();deadline=time.monotonic()+(60. if upload else 5.)
            while True:
                try:message=await asyncio.wait_for(receive(),timeout=max(0,deadline-time.monotonic()))
                except asyncio.TimeoutError:return await reply(408,'request timeout')
                if message['type']=='http.disconnect':return
                chunk=message.get('body',b'')
                if len(body)+len(chunk)>cap:return await reply(413,'request too large')
                body.extend(chunk)
                if not message.get('more_body'):break
            async def buffered():return {'type':'http.request','body':bytes(body),'more_body':False}
            async def secure_send(message):
                if message['type']=='http.response.start':
                    message={**message,'headers':list(message.get('headers',[]))+[(k.lower().encode(),v.encode()) for k,v in HEADERS.items()]}
                await send(message)
            await self.app(scope,buffered,secure_send)
        finally:self.inflight-=1


def create_research_app(predictor=None,*,video_predictor=None,api_key=None,timeout=5.,memory_mb=512,work_root=None):
    """Trusted picklable local provider only; no public deployment or request file paths.

    api_key defaults to existing API_KEY environment. No credential is generated or persisted.
    Use make_model_provider for stage15, not a non-picklable closure. CPU/memory/file caps,
    one active child, SIGKILL process-group cleanup and owner-only temporary job storage.
    """
    key=os.environ.get('API_KEY','') if api_key is None else api_key
    if not isinstance(key,str) or not key:raise RuntimeError('API_KEY env var must be set')
    runner=ResearchRunner(predictor,timeout=timeout,memory_mb=memory_mb,work_root=work_root) if predictor is not None else None
    video_runner=ResearchRunner(video_predictor,timeout=timeout,memory_mb=memory_mb,work_root=work_root) if video_predictor is not None else None
    @asynccontextmanager
    async def lifespan(app):
        try:yield
        finally:
            if runner:runner.close()
            if video_runner:video_runner.close()
    app=FastAPI(title='Stage15 local research',docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)
    app.state.runner=runner
    app.state.video_runner=video_runner
    app.add_middleware(ResearchGuard,key=key.encode())
    @app.get('/')
    def upload_ui():
        return FileResponse(Path(__file__).resolve().parents[1]/'static/research.html',headers=HEADERS)
    @app.get('/research/health')
    def health():
        return {'research_only':True,'production_enabled':False,'status':'not_validated',
                'instance_id':os.environ.get('RESEARCH_INSTANCE_ID','local')}
    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request,_error):
        return JSONResponse({'detail':'invalid research request'},status_code=422)
    @app.post('/research/stage15/predict',response_model=ResearchResponse)
    def predict(request:ResearchRequest):
        if runner is None:raise HTTPException(503,'research provider unavailable')
        try:output=runner.run(request.sample_id)
        except JobTimeout:raise HTTPException(504,'research timeout') from None
        except JobBusy:raise HTTPException(429,'research worker busy') from None
        except JobFailure:raise HTTPException(503,'research prediction unavailable') from None
        return ResearchResponse(uncalibrated_clip_logit=output['clip_logit'],
            candidate_times_sec=[output['candidate_time_sec']] if output['clip_logit']>=0 else [])
    @app.post('/research/stage15/analyze',response_model=ResearchResponse)
    async def analyze(request:Request):
        if video_runner is None:raise HTTPException(503,'research provider unavailable')
        data=await request.body()
        if not data:raise HTTPException(400,'empty upload')
        try:output=await run_in_threadpool(video_runner.run,'uploaded',data)
        except JobTimeout:raise HTTPException(504,'research timeout') from None
        except JobBusy:raise HTTPException(429,'research worker busy') from None
        except JobFailure:raise HTTPException(503,'research prediction unavailable') from None
        return ResearchResponse(uncalibrated_clip_logit=output['clip_logit'],
            candidate_times_sec=[output['candidate_time_sec']] if output['clip_logit']>=0 else [])
    return app
