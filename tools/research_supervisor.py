"""Explicitly approved, maximum-24-hour local research/tunnel supervisor. Never auto-starts."""
import argparse
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
MAX_SECONDS = 86400


class Lifetime:
    def __init__(self, seconds=MAX_SECONDS, *, wall=time.time, monotonic=time.monotonic):
        if not math.isfinite(seconds) or not 0 < seconds <= MAX_SECONDS:
            raise ValueError('duration must be within 24 hours')
        self.seconds = seconds
        self.wall, self.monotonic = wall, monotonic
        self.start_wall, self.start_mono = wall(), monotonic()

    def expired(self):
        # Wall time covers machine sleep even where monotonic excludes suspended time.
        return max(self.wall()-self.start_wall, self.monotonic()-self.start_mono) >= self.seconds


def stop_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)


def stop_all(children):
    failed = False
    for child in reversed(children):
        try:stop_group(child)
        except Exception:failed = True
    if failed:raise RuntimeError('one or more child groups did not stop')


def supervise(children, lifetime, stopped, *, wait=None):
    """One lifetime, no restart. Any child exit, stop request or expiry ends all groups."""
    pause = wait or stopped.wait
    try:
        while True:
            if stopped.is_set():return 'stopped'
            if lifetime.expired():return 'expired'
            if any(child.poll() is not None for child in children):return 'child_exit'
            pause(.25)
    finally:
        stop_all(children)


def verified_binary(binary, manifest):
    binary, manifest = Path(binary), Path(manifest)
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError('cloudflared executable unavailable')
    raw = json.loads(manifest.read_text())
    expected = raw.get('sha256', '')
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('invalid cloudflared manifest')
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if digest != expected:raise ValueError('cloudflared integrity mismatch')
    return str(binary.resolve())


def child_environment(run_dir, key=None):
    env = {'PATH':'/usr/bin:/bin', 'HOME':str(run_dir), 'TMPDIR':str(run_dir),
           'PYTHONPATH':str(ROOT), 'PYTHONDONTWRITEBYTECODE':'1'}
    if key is not None:env['API_KEY'] = key
    return env


def ready(key, instance_id, *, opener=None):
    """Authenticated per-instance loopback readiness; no media/model inference."""
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
    endpoint = 'http://127.0.0.1:8000/research/health'
    for credential, expected in [(None, 401), (key, 200)]:
        headers = {}
        if credential is not None:headers['X-API-Key'] = credential
        request = urllib.request.Request(endpoint, headers=headers)
        body = b''
        try:
            with opener.open(request, timeout=.5) as response:
                code = response.status
                body = response.read(1025)
        except urllib.error.HTTPError as error:
            code = error.code
            error.close()
        except (OSError, ValueError):return False
        if code != expected:return False
        if credential is not None:
            try:payload = json.loads(body) if len(body)<=1024 else None
            except (ValueError, UnicodeError):return False
            if payload != {'research_only':True, 'production_enabled':False,
                           'status':'not_validated', 'instance_id':instance_id}:return False
    return True


def write_state(path, state):
    """Atomically replace private non-secret operational metadata only."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.research-state-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            json.dump(state, output, sort_keys=True)
            output.flush();os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def lock_state(path):
    """One live supervisor; never replace state claiming an active run."""
    path = Path(path)
    fd = os.open(str(path)+'.lock', os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.is_symlink():raise RuntimeError('invalid state path')
        if path.exists() and json.loads(path.read_text()).get('status') in {'starting','running'}:
            raise RuntimeError('existing active research state requires explicit resolution')
        return fd
    except BaseException:
        os.close(fd)
        raise


def require_free_port():
    # Fail before launching if an older listener could answer our readiness probe.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1', 8000))


def run_approved(seconds=MAX_SECONDS):
    # Called only after explicit CLI approval gate. Never creates or persists a key.
    key = os.environ.get('API_KEY', '')
    if not key:raise RuntimeError('existing API_KEY required')
    lifetime = Lifetime(seconds)
    require_free_port()
    binary = verified_binary(ROOT/'local_private/bin/cloudflared',
                             ROOT/'local_private/bin/cloudflared.manifest.json')
    state_path = ROOT/'local_private/research_access_state.json'
    lock_fd = lock_state(state_path)
    stopped = threading.Event()
    previous = {sig:signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGTERM,signal.SIGINT)}
    children = []
    instance_id = uuid.uuid4().hex
    state = {'supervisor_pid':os.getpid(), 'api_pid':None, 'tunnel_pid':None,
             'started_unix':lifetime.start_wall, 'expires_unix':lifetime.start_wall+seconds,
             'run_dir':None, 'status':'starting'}
    try:
        with tempfile.TemporaryDirectory(prefix='approved-research-', dir=ROOT/'local_private') as folder:
            os.chmod(folder, 0o700)
            state['run_dir'] = folder
            write_state(state_path, state)
            try:
                for name in ('api.log', 'tunnel.log'):
                    fd = os.open(Path(folder)/name, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
                    os.close(fd)
                with open(Path(folder)/'api.log', 'ab') as api_log, open(Path(folder)/'tunnel.log', 'ab') as tunnel_log:
                    def start(command, env, log, label):
                        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=log, close_fds=True, start_new_session=True)
                        children.append(proc)
                        state[label+'_pid'] = proc.pid
                        write_state(state_path, state)
                        return proc
                    env = child_environment(folder, key)
                    env['RESEARCH_INSTANCE_ID'] = instance_id
                    api = start([str(ROOT/'.venv/bin/python'), '-B', '-m', 'uvicorn',
                        'tools.research_video:create_video_app', '--factory', '--workers', '1',
                        '--no-proxy-headers', '--host', '127.0.0.1', '--port', '8000', '--no-access-log'],
                        env, api_log, 'api')
                    startup_deadline = time.monotonic()+30
                    while not ready(key, instance_id):
                        if stopped.is_set() or lifetime.expired() or api.poll() is not None or time.monotonic()>=startup_deadline:
                            raise RuntimeError('local research readiness failed')
                        stopped.wait(.25)
                    if stopped.is_set() or lifetime.expired() or api.poll() is not None:
                        raise RuntimeError('local research stopped before tunnel')
                    start([binary, 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:8000',
                           '--metrics', '127.0.0.1:0', '--management-diagnostics=false', '--loglevel', 'info'],
                          child_environment(folder), tunnel_log, 'tunnel')
                    state['status'] = 'running';write_state(state_path, state)
                    reason = supervise(children, lifetime, stopped)
                    children.clear()  # supervise completed all group cleanup successfully.
                    state['status'] = reason
                    return reason
            finally:
                stop_all(children)  # includes startup/health/tunnel failure paths before directory cleanup
                children.clear()
    except BaseException:
        state['status'] = 'failed'
        raise
    finally:
        try:
            write_state(state_path, state)
        finally:
            for sig, handler in previous.items():signal.signal(sig, handler)
            os.close(lock_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-approved', action='store_true', help='start only after explicit user approval')
    parser.add_argument('--seconds', type=float, default=MAX_SECONDS)
    args = parser.parse_args(argv)
    if not args.start_approved:parser.error('explicit --start-approved is required; nothing started')
    try:run_approved(args.seconds)
    except Exception:
        print('Research supervisor stopped; private setup/readiness check failed.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':raise SystemExit(main())
