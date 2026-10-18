"""No tunnel/network/real secrets: synthetic supervisor lifecycle checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import urllib.error

import pytest
from tools import research_supervisor as S


def test_cli_requires_explicit_approval_without_reading_environment(monkeypatch):
    def forbidden(*args, **kwargs):pytest.fail('start called without approval')
    monkeypatch.setattr(S, 'run_approved', forbidden)
    with pytest.raises(SystemExit) as error:S.main([])
    assert error.value.code == 2


@pytest.mark.parametrize('seconds', [0, -1, 86401, float('inf'), float('nan')])
def test_maximum_duration_cannot_be_extended(seconds):
    with pytest.raises(ValueError):S.Lifetime(seconds)


def test_sleep_resume_wall_and_monotonic_deadlines():
    clocks = [100., 200.]
    lifetime = S.Lifetime(86400, wall=lambda:clocks[0], monotonic=lambda:clocks[1])
    clocks[0] += 86400
    assert lifetime.expired()
    clocks = [100., 200.]
    lifetime = S.Lifetime(10, wall=lambda:clocks[0], monotonic=lambda:clocks[1])
    clocks[0] -= 1000
    clocks[1] += 10
    assert lifetime.expired()


@pytest.mark.parametrize('reason', ['expired', 'stopped', 'child_exit'])
def test_all_children_cleaned_without_restart(monkeypatch, reason):
    class Child:
        def __init__(self, exited=False):self.exited=exited
        def poll(self):return 0 if self.exited else None
    children = [Child(reason=='child_exit'), Child()]
    stopped=threading.Event()
    if reason=='stopped':stopped.set()
    class Clock:
        def expired(self):return reason=='expired'
    killed=[]
    monkeypatch.setattr(S,'stop_group',killed.append)
    assert S.supervise(children,Clock(),stopped)==reason
    assert killed==list(reversed(children))


def test_real_synthetic_process_group_is_terminated():
    proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],
                          start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        S.stop_group(proc)
        assert proc.poll() is not None
        with pytest.raises(ProcessLookupError):os.kill(proc.pid,0)
    finally:
        if proc.poll() is None:proc.kill();proc.wait()


def test_cloudflare_environment_does_not_inherit_key_or_credentials(monkeypatch,tmp_path):
    monkeypatch.setenv('API_KEY','synthetic-fixture')
    monkeypatch.setenv('UNRELATED_SECRET','synthetic-fixture')
    env=S.child_environment(tmp_path)
    assert 'API_KEY' not in env and 'UNRELATED_SECRET' not in env
    assert env['HOME']==env['TMPDIR']==str(tmp_path)
    assert S.child_environment(tmp_path,'synthetic-fixture')['API_KEY']=='synthetic-fixture'


def test_binary_hash_verification_and_symlink_rejection(tmp_path):
    binary=tmp_path/'synthetic-binary';binary.write_bytes(b'fixture');binary.chmod(0o700)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'sha256':hashlib.sha256(b'fixture').hexdigest()}))
    assert S.verified_binary(binary,manifest)==str(binary.resolve())
    link=tmp_path/'link';link.symlink_to(binary)
    with pytest.raises(ValueError):S.verified_binary(link,manifest)
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError):S.verified_binary(binary,manifest)


@pytest.mark.parametrize('returned,expected', [('instance-fixture',True),('other-instance',False)])
def test_readiness_checks_authenticated_instance_without_network(returned,expected):
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def read(self,limit):
            return json.dumps({'research_only':True,'production_enabled':False,
                'status':'not_validated','instance_id':returned}).encode()
    class Opener:
        def open(self,request,timeout):
            assert request.full_url=='http://127.0.0.1:8000/research/health'
            assert request.data is None
            if not request.has_header('X-api-key'):
                raise urllib.error.HTTPError(request.full_url,401,'fixture',{},None)
            return Response()
    assert S.ready('synthetic-fixture','instance-fixture',opener=Opener()) is expected


def test_existing_listener_fails_closed_without_network(monkeypatch):
    class Socket:
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def bind(self,address):
            assert address==('127.0.0.1',8000)
            raise OSError('synthetic address already in use')
    monkeypatch.setattr(S.socket,'socket',lambda *_:Socket())
    with pytest.raises(OSError):S.require_free_port()


def test_short_real_expiry_kills_synthetic_child():
    proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],
                          start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        assert S.supervise([proc],S.Lifetime(.1),threading.Event())=='expired'
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:proc.kill();proc.wait()


def test_cleanup_failure_does_not_skip_other_children(monkeypatch):
    seen=[]
    def stop(child):
        seen.append(child)
        if child==2:raise OSError('synthetic')
    monkeypatch.setattr(S,'stop_group',stop)
    with pytest.raises(RuntimeError):S.stop_all([1,2])
    assert seen==[2,1]


def test_private_atomic_state_and_active_run_rejected(tmp_path):
    path=tmp_path/'state.json'
    S.write_state(path,{'status':'starting','supervisor_pid':123})
    assert path.stat().st_mode&0o777==0o600
    with pytest.raises(RuntimeError):S.lock_state(path)
    assert json.loads(path.read_text())['status']=='starting'
    S.write_state(path,{'status':'stopped'})
    lock=S.lock_state(path)
    try:
        with pytest.raises(BlockingIOError):S.lock_state(path)
    finally:os.close(lock)
