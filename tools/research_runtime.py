"""Bounded local research jobs. Trusted Python provider, not an OS filesystem sandbox."""
import io,json,math,os,pickle,signal,stat,subprocess,sys,tempfile,threading,time
from pathlib import Path
from app.jobs import _kill
from tools.research_worker import validate_output

ROOT=Path(__file__).resolve().parents[1]
MAX_INPUT=32*1048576


class JobFailure(Exception):pass
class JobTimeout(JobFailure):pass
class JobBusy(JobFailure):pass


class _BoundedBuffer(io.BytesIO):
    def write(self,b):
        if self.tell()+len(b)>MAX_INPUT:raise ValueError('provider snapshot too large')
        return super().write(b)


def _stop(proc):
    # Platform helper reaps live leaders. Kill group even if leader already exited.
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    _kill(proc)
    proc.wait(timeout=5)


def group_rss(pid):
    # Parent monitors macOS memory (RLIMIT_AS is unsupported there), all group members.
    result=subprocess.run(['/bin/ps','-axo','pid=,pgid=,rss='],capture_output=True,text=True,timeout=.5)
    if result.returncode:raise JobFailure('resource monitor unavailable')
    total=0
    for line in result.stdout.splitlines():
        p,g,r=map(int,line.split())
        if g==pid:total+=r*1024
    return total


class ResearchRunner:
    def __init__(self,provider,*,timeout=5.,memory_mb=512,work_root=None):
        if not math.isfinite(timeout) or not .1<=timeout<=120:raise ValueError('invalid timeout')
        if isinstance(memory_mb,bool) or not isinstance(memory_mb,int) or not 64<=memory_mb<=2048:
            raise ValueError('invalid memory cap')
        self.timeout=timeout;self.memory_mb=memory_mb
        self.root=None
        if work_root is not None:
            p=Path(work_root)
            if p.is_symlink() or not p.is_dir():raise ValueError('invalid private root')
            st=p.stat()
            if st.st_uid!=os.getuid() or stat.S_IMODE(st.st_mode)&0o077:raise ValueError('private root must be owner-only')
            self.root=p.resolve()
        b=_BoundedBuffer()
        try:pickle.dump(provider,b,protocol=5)
        except Exception:raise ValueError('provider must be trusted, picklable and bounded') from None
        self.provider=b.getvalue();self.lock=threading.Lock();self.slot=threading.BoundedSemaphore(1)
        self.active=set();self.closed=False

    def close(self):
        with self.lock:
            self.closed=True;active=list(self.active)
        for proc in active:_stop(proc)

    def run(self,sample_id,upload=None):
        if not self.slot.acquire(blocking=False):raise JobBusy('research worker busy')
        proc=None
        try:
            with tempfile.TemporaryDirectory(prefix='forte-research-',dir=self.root) as directory:
                folder=Path(directory);os.chmod(folder,0o700)
                # Parent's trusted snapshot only. sample_id is separately schema-validated, no paths.
                fd=os.open(folder/'input.pkl',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                with os.fdopen(fd,'wb') as f:f.write(self.provider)
                fd=os.open(folder/'request.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                with os.fdopen(fd,'w') as f:json.dump({'sample_id':sample_id},f)
                if upload is not None:
                    if not isinstance(upload,bytes) or not 0<len(upload)<=60*1048576:raise JobFailure('invalid upload')
                    fd=os.open(folder/'input.mov',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                    with os.fdopen(fd,'wb') as f:f.write(upload)
                deadline=time.monotonic()+self.timeout
                env={'PATH':'/usr/bin:/bin','PYTHONPATH':str(ROOT)+os.pathsep+str(ROOT/'tests'),
                     'HOME':str(folder),'TMPDIR':str(folder),'PYTHONDONTWRITEBYTECODE':'1',
                     'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1',
                     'MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
                with self.lock:
                    if self.closed:raise JobFailure('research worker closed')
                    proc=subprocess.Popen([sys.executable,'-B','-m','tools.research_worker',str(os.getpid()),
                         str(time.time()+self.timeout),str(math.ceil(self.timeout)+1),str(self.memory_mb)],
                         cwd=folder,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,start_new_session=True,close_fds=True)
                    self.active.add(proc)
                try:
                    while proc.poll() is None:
                        if time.monotonic()>=deadline:raise JobTimeout('research timeout')
                        if group_rss(proc.pid)>self.memory_mb*1048576:raise JobFailure('research resource limit')
                        time.sleep(min(.025,max(0,deadline-time.monotonic())))
                    if time.monotonic()>=deadline:raise JobTimeout('research timeout')
                    if proc.returncode:raise JobFailure('research prediction unavailable')
                    fd=os.open(folder/'output.json',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
                    with os.fdopen(fd,'rb') as f:
                        st=os.fstat(f.fileno())
                        if not stat.S_ISREG(st.st_mode) or st.st_size>1024:raise JobFailure('invalid output')
                        output=validate_output(json.loads(f.read(1025)))
                    return output
                finally:
                    _stop(proc)
                    with self.lock:self.active.discard(proc)
        except (JobFailure,JobTimeout):raise
        except Exception:raise JobFailure('research prediction unavailable') from None
        finally:self.slot.release()
