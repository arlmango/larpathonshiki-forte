"""New-video inference only: pinned full-frame SigLIP cache semantics and saved stage15."""
from dataclasses import dataclass
from pathlib import Path
import math

MAX_SECONDS=150.
MAX_PIXELS=1920*1080


def extract(path):
    """No cache lookup or labels. Same index/fps sampling and float16 roundtrip as stage13."""
    import os,struct
    # Reject playlists/non-media before decoder entry; MOV external data references disabled.
    with open(path,'rb') as handle:header=handle.read(8)
    if len(header)!=8 or header[4:] not in (b'ftyp',b'wide',b'free',b'moov',b'mdat',b'skip'):
        raise ValueError('unsupported video container')
    if struct.unpack('>I',header[:4])[0] not in (0,1) and struct.unpack('>I',header[:4])[0]<8:
        raise ValueError('invalid video container')
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']='protocol_whitelist;file|format_whitelist;mov,mp4,m4a,3gp,3g2,mj2|enable_drefs;0|use_absolute_path;0'
    import cv2,numpy as np,onnxruntime as ort
    from tools import stage5_encoder as E
    if not E.onnx_ok():raise ValueError('encoder integrity failed')
    opts=ort.SessionOptions();opts.log_severity_level=4;opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    sess=ort.InferenceSession(E.ONNX_PATH,opts,providers=['CPUExecutionProvider'])
    cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG);emb=[];times=[];batch=[];idx=k=0;broken=False
    def flush():
        if batch:
            raw=sess.run(None,{'pixel_values':E.preprocess(batch,None)})[0]
            if raw.shape!=(len(batch),768) or not np.isfinite(raw).all():raise ValueError('invalid embeddings')
            # Historical cache stores normalized float16, then stage13 loads/re-normalizes float64.
            emb.append((raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-9)).astype(np.float16))
            batch.clear()
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS));n=float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w=float(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not all(math.isfinite(x) for x in [fps,n,w,h]) or not 0<fps<=120 or n<=0 or w*h>MAX_PIXELS or min(w,h)<2:
            raise ValueError('unsupported video')
        duration=n/fps
        if duration>MAX_SECONDS:raise ValueError('video too long')
        while cap.grab():
            if idx/fps>MAX_SECONDS or idx>18000:raise ValueError('video too long')
            if idx/fps>=k+.5:
                ok,fr=cap.retrieve()
                if not ok or fr is None:broken=True;break
                if fr.shape[0]*fr.shape[1]>MAX_PIXELS:raise ValueError('video too large')
                times.append(idx/fps);batch.append(fr);k=int(idx/fps-.5)+1
                if len(batch)==16:flush()
            idx+=1
        flush()
    finally:cap.release()
    expected=max(0,int(np.ceil(duration-.5)))
    if broken or len(times)<max(3,expected-1) or abs(idx/fps-duration)>1.:
        raise ValueError('decode incomplete')
    raw=np.concatenate(emb).astype(float)
    normalized=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-9)
    # Historical timestamps were stored as float32 before float64 loading.
    return np.asarray(times,np.float32).astype(float),normalized-np.median(normalized,axis=0)


@dataclass
class VideoProvider:
    classifier:dict
    localizer:dict
    def __call__(self,sample_id):
        if sample_id!='uploaded':raise ValueError('invalid input')
        t,z=extract(Path('input.mov'))
        from tools.stage15_two_component import predict
        score,peak,_=predict(t,z,self.classifier,self.localizer)
        return {'clip_logit':score,'candidate_time_sec':peak}


def make_video_provider(model_dir):
    from tools.stage15_two_component import load_bundle
    cls,loc=load_bundle(model_dir)
    return VideoProvider(cls,loc)


def create_video_app():
    """ASGI factory; only existing API_KEY is read, no keys generated or persisted."""
    from tools.stage15_api import create_research_app
    root=Path(__file__).resolve().parents[1]
    return create_research_app(video_predictor=make_video_provider(root/'local_private/research_models/stage15_two_component'),
                              timeout=120.,memory_mb=2048)
