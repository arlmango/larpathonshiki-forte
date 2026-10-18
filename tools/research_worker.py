"""Private exec child. Only reads a trusted parent-created pickle, never HTTP pickle."""
import json,math,os,pickle,resource,stat,sys,time
from collections.abc import Mapping
from numbers import Real


def limits(cpu_seconds,memory_mb):
    for kind,value in [(resource.RLIMIT_CPU,cpu_seconds),(resource.RLIMIT_FSIZE,1048576),
                       (resource.RLIMIT_NOFILE,64),(resource.RLIMIT_CORE,0)]:
        resource.setrlimit(kind,(value,value))
    if sys.platform.startswith('linux'):
        resource.setrlimit(resource.RLIMIT_AS,(memory_mb*1048576,memory_mb*1048576))


def validate_output(output):
    if not isinstance(output,Mapping) or set(output)!={'clip_logit','candidate_time_sec'}:
        raise ValueError('invalid output')
    values=[output['clip_logit'],output['candidate_time_sec']]
    if any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) for v in values):
        raise ValueError('invalid output')
    logit,timestamp=map(float,values)
    if timestamp<0:raise ValueError('invalid output')
    return {'clip_logit':logit,'candidate_time_sec':timestamp}


def main():
    os.umask(0o077)
    ppid=int(sys.argv[1]);deadline=float(sys.argv[2]);cpu=int(sys.argv[3]);memory=int(sys.argv[4])
    limits(cpu,memory)
    # Parent death and deadline both kill the full process group, not only this leader.
    import threading,signal
    def backstop():
        while True:
            if os.getppid()!=ppid or time.time()>deadline+1:
                os.killpg(os.getpgrp(),signal.SIGKILL)
            time.sleep(.05)
    threading.Thread(target=backstop,daemon=True).start()
    fd=os.open('input.pkl',os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        st=os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size>32*1048576:raise ValueError('invalid input')
        provider=pickle.load(f)
    with open('request.json') as f:sample_id=json.load(f)['sample_id']
    output=validate_output(provider(sample_id))
    fd=os.open('output.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as f:json.dump(output,f,allow_nan=False)


if __name__=='__main__':
    try:main()
    except BaseException:sys.exit(2)  # stderr is also discarded; never surface private errors
