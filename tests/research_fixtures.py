"""Only synthetic trusted providers used by exec-worker security tests."""
from dataclasses import dataclass
import json,os,resource,signal,subprocess,sys,time


@dataclass
class Probe:
    mode:str='ok'
    marker:str=''
    def __call__(self,sample_id):
        if self.marker:
            with open(self.marker,'w') as f:json.dump({'pid':os.getpid(),'cwd':os.getcwd()},f)
        if self.mode=='sleep':
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            time.sleep(20)
        if self.mode=='descendant':
            subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
        if self.mode=='exception':raise RuntimeError('private details '+sample_id)
        if self.mode=='symlink':
            os.symlink(self.marker,'output.json')
        if self.mode=='caps':
            assert resource.getrlimit(resource.RLIMIT_NOFILE)==(64,64)
            assert resource.getrlimit(resource.RLIMIT_FSIZE)==(1048576,1048576)
            assert resource.getrlimit(resource.RLIMIT_CORE)==(0,0)
            assert resource.getrlimit(resource.RLIMIT_CPU)[0]>0
            assert not {'API_KEY','AWS_SECRET_ACCESS_KEY','UNRELATED_TEST_SECRET'}&os.environ.keys()
            assert os.stat('.').st_mode&0o777==0o700
            assert os.stat('input.pkl').st_mode&0o777==0o600
            assert os.environ['HOME']==os.getcwd()==os.environ['TMPDIR']
        if self.mode=='filecap':
            with open('oversized','wb') as f:f.write(b'x'*2097152)
        if self.mode=='memory':
            blocks=[]
            while True:blocks.append(bytearray(8*1048576));time.sleep(.02)
        return {'clip_logit':1.,'candidate_time_sec':2.}
