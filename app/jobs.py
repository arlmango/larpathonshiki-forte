"""Bounded job queue with hard per-job wall-clock deadline.

- Each analysis runs in a child process (`python -m app.worker`) so a hung
  decoder / slow model load can be SIGKILLed. Tests may inject `analyze_fn`,
  which runs in-process (cannot be killed, but the watchdog still fails it on time).
- Deadline = submission time + job_budget_sec (queue wait included).
- Admission control: capacity (workers + queue_limit) and predicted completion
  within budget; rejection raises QueueFull (a queue.Full subclass -> HTTP 429).
- Final records are persisted to data_dir/<job_id>.json; get() falls back to disk.
- Uploads are always deleted; orphans and expired records are swept.
"""
import dataclasses
import heapq
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque

from .worker import InvalidResult, safe_code, safe_error, validate_result

log = logging.getLogger(__name__)

TIMEOUT_ERROR = "time_budget_exceeded"
LOST_ERROR = "job_lost"  # owning process died / service restarted
_JID = re.compile(r"[0-9a-f]{32}")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# files this module (or main.py's uploads) may create in data_dir; nothing else is touched
_TEMP_RE = re.compile(r"[0-9a-f]{32}(\.upload|\.worker\.json|\.worker\.json\.tmp|\.json\.tmp)")
_RESULT_RE = re.compile(r"([0-9a-f]{32})\.json")


class QueueFull(queue.Full):
    """Job rejected by admission control (maps to HTTP 429)."""


class JobStore:
    def __init__(self, settings, analyze_fn=None):
        self.s = settings
        self.analyze = analyze_fn  # None -> isolated subprocess mode
        self.data_dir = os.path.abspath(settings.data_dir)
        self.jobs: dict[str, dict] = {}
        self.pending: deque[str] = deque()
        self.busy: dict[int, tuple[float, float]] = {}  # worker idx -> (start, deadline)
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.stopping = False
        self._stop = threading.Event()
        self._recent: deque[float] = deque(maxlen=20)  # run times of non-timeout jobs
        os.makedirs(self.data_dir, exist_ok=True)
        self._sweep_disk()
        self.threads = [threading.Thread(target=self._work, args=(i,), daemon=True, name=f"job-worker-{i}")
                        for i in range(settings.workers)]
        self.threads.append(threading.Thread(target=self._watchdog, daemon=True, name="job-watchdog"))
        for t in self.threads:
            t.start()

    # ------------------------------------------------------------------ API
    def submit(self, path: str, submitted_at: float | None = None) -> str:
        """Enqueue an uploaded file. Raises QueueFull (queue.Full) when rejected;
        the file is removed in that case."""
        jid = uuid.uuid4().hex
        t0 = submitted_at or time.time()
        try:
            with self.lock:
                if self.stopping:
                    raise QueueFull("shutting down")
                self._admit(t0)
                self.jobs[jid] = {"pub": {"job_id": jid, "status": "processing"}, "state": "queued",
                                  "path": path, "t0": t0, "deadline": t0 + self.s.job_budget_sec,
                                  "proc": None, "finished": None}
                self.pending.append(jid)
                self.cv.notify()
        except BaseException:
            _rm(path)
            raise
        self._write_json(jid, {"job_id": jid, "status": "processing"})
        return jid

    def get(self, jid):
        with self.lock:
            j = self.jobs.get(jid)
            if j is not None:
                return dict(j["pub"])
        return self._read_disk(jid)

    def stats(self) -> dict:
        with self.lock:
            return {"queued": len(self.pending), "running": len(self.busy),
                    "records": len(self.jobs), "estimate_sec": self._estimate()}

    def close(self, timeout: float = 5.0):
        """Stop workers, kill running children, fail unfinished jobs."""
        with self.lock:
            self.stopping = True
            self._stop.set()
            unfinished = [(jid, j["path"]) for jid, j in self.jobs.items() if j["state"] != "final"]
            self.cv.notify_all()
        for jid, path in unfinished:
            self._finish(jid, _failed(jid, "service_shutting_down"))
            _rm(path)
        with self.lock:
            procs = [j["proc"] for j in self.jobs.values() if j["proc"] is not None]
        for p in procs:
            _kill(p)
        for t in self.threads:
            t.join(timeout)

    # ------------------------------------------------------------ admission
    def _estimate(self) -> float:
        """Max of the last 20 observed run times (done or failed, excluding
        time-budget kills); the configured value is only the cold-start guess."""
        return max(self._recent) if self._recent else float(self.s.job_estimate_sec)

    def _admit(self, t0: float):
        """Called with lock held. Rejects if over capacity or if the predicted
        completion of the new job would exceed its budget."""
        cap = self.s.workers + self.s.queue_limit
        if len(self.pending) + len(self.busy) + 1 > cap:
            raise QueueFull("queue full")
        now, est = time.time(), self._estimate()
        # free-at times (relative to now) for every worker slot
        slots = []
        for start, deadline in self.busy.values():
            left = max(deadline - now, 0.0)
            elapsed = now - start
            # within estimate: est - elapsed; overrunning (maybe hung): until its deadline
            slots.append(min(est - elapsed, left) if elapsed < est else left)
        slots += [0.0] * (self.s.workers - len(self.busy))
        heapq.heapify(slots)
        for jid in self.pending:  # FIFO: each queued job occupies the earliest free slot
            j = self.jobs[jid]
            t = heapq.heappop(slots)
            heapq.heappush(slots, min(t + est, max(j["deadline"] - now, t)))
        finish = (now - t0) + slots[0] + est
        if finish > self.s.job_budget_sec + self.s.admission_margin_sec:
            raise QueueFull("predicted completion exceeds budget")

    # --------------------------------------------------------------- worker
    def _work(self, idx: int):
        while True:
            with self.lock:
                while not self.pending and not self.stopping:
                    self.cv.wait()
                if self.stopping:
                    return
                jid = self.pending.popleft()
                j = self.jobs.get(jid)
                if j is None or j["state"] == "final":
                    continue
                j["state"] = "running"
                self.busy[idx] = (time.time(), j["deadline"])
            started = time.time()
            try:
                if time.time() >= j["deadline"]:
                    out = _failed(jid, TIMEOUT_ERROR)
                elif self.analyze is not None:
                    out = self._run_inprocess(jid, j)
                else:
                    out = self._run_subprocess(jid, j)
            except Exception as e:  # never leave a job in processing
                log.error("job %s internal error %s", jid, e.__class__.__name__)
                out = _failed(jid, "internal_error")
            finally:
                _rm(j["path"])
                with self.lock:
                    self.busy.pop(idx, None)
                    j["proc"] = None
                    if out.get("error") != TIMEOUT_ERROR:
                        self._recent.append(time.time() - started)
            self._finish(jid, out)

    def _run_inprocess(self, jid, j) -> dict:
        try:
            res = self.analyze(j["path"], self.s, deadline=j["deadline"])
            return _done(jid, res)
        except Exception as e:  # any error -> failed, never no_violation
            return _failed(jid, safe_error(e))

    def _run_subprocess(self, jid, j) -> dict:
        out_path = os.path.join(self.data_dir, jid + ".worker.json")
        env = dict(os.environ)
        env["PYTHONPATH"] = _ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["JOB_SETTINGS"] = json.dumps(_settings_dict(self.s))
        # soft deadline slightly before the hard kill so the pipeline can fail cleanly
        env["JOB_DEADLINE"] = repr(j["deadline"] - min(1.0, self.s.job_budget_sec * 0.05))
        env["JOB_PARENT_PID"] = str(os.getpid())
        try:
            # stdout/stderr discarded: decoder output may carry real-data details
            proc = subprocess.Popen(
                [sys.executable, "-m", "app.worker", os.path.abspath(j["path"]), out_path],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, start_new_session=True, close_fds=True)
            with self.lock:
                j["proc"] = proc
                abort = self.stopping or j["state"] == "final"
            if abort:
                _kill(proc)
            t_start = time.time()
            try:
                proc.wait(timeout=max(0.0, j["deadline"] - time.time()))
            except subprocess.TimeoutExpired:
                _kill(proc)
                log.warning("job %s killed: timeout elapsed=%.1fs", jid, time.time() - j["t0"])
                return _failed(jid, TIMEOUT_ERROR)
            rc = proc.returncode
            if rc != 0:
                log.warning("job %s worker exit=%s run=%.1fs", jid, rc, time.time() - t_start)
                if rc == -signal.SIGKILL:  # not ours (we return above): likely OOM killer
                    return _failed(jid, "worker_crashed_oom_suspected")
                return _failed(jid, "worker_crashed")
            return _parse_worker_output(jid, out_path)
        finally:
            for p in (out_path, out_path + ".tmp"):
                _rm(p)

    # ------------------------------------------------------------ lifecycle
    def _finish(self, jid: str, out: dict) -> bool:
        """Set the final record exactly once; later attempts are ignored."""
        with self.lock:
            j = self.jobs.get(jid)
            if j is None or j["state"] == "final":
                return False
            if j["state"] == "queued":
                try:
                    self.pending.remove(jid)
                except ValueError:
                    pass
            j["state"], j["pub"], j["finished"] = "final", out, time.time()
        self._write_json(jid, out)
        return True

    def _watchdog(self):
        last_disk = time.time()
        while not self._stop.wait(0.5):
            now = time.time()
            expired, overdue = [], []
            with self.lock:
                for jid, j in self.jobs.items():
                    if j["state"] == "final":
                        if now - j["finished"] > self.s.job_ttl_sec:
                            expired.append(jid)
                    elif now > j["deadline"] + self.s.kill_grace_sec:
                        overdue.append((jid, j["proc"], j["path"]))
                for jid in expired:
                    del self.jobs[jid]
            for jid, proc, path in overdue:
                if proc is not None:
                    _kill(proc)
                if self._finish(jid, _failed(jid, TIMEOUT_ERROR)):
                    log.warning("job %s failed by watchdog", jid)
                _rm(path)
            if now - last_disk > 60:
                last_disk = now
                self._sweep_disk()

    # ----------------------------------------------------------------- disk
    def _write_json(self, jid: str, obj: dict):
        p = os.path.join(self.data_dir, jid + ".json")
        try:
            with self.lock:  # keeps the final write from being overtaken by the pending marker
                if obj.get("status") == "processing" and self.jobs.get(jid, {}).get("state") == "final":
                    return
                with open(p + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False)
                os.replace(p + ".tmp", p)
        except OSError as e:
            log.error("job %s cannot persist status errno=%s", jid, e.errno)

    def _load_disk(self, jid: str):
        if not isinstance(jid, str) or not _JID.fullmatch(jid):
            return None, 0.0
        p = os.path.join(self.data_dir, jid + ".json")
        try:
            age = time.time() - os.path.getmtime(p)
            with open(p, encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, ValueError):
            return None, 0.0
        if not isinstance(obj, dict) or obj.get("job_id") != jid:
            return None, 0.0
        return obj, age

    def _read_disk(self, jid: str):
        obj, age = self._load_disk(jid)
        if obj is None or age > self.s.job_ttl_sec:
            return None
        if obj.get("status") == "done":
            return _done(jid, obj)  # re-validate/project; tampered files never served raw
        if obj.get("status") == "failed":
            return _failed(jid, obj.get("error"))
        if obj.get("status") != "processing":
            return None
        if obj.get("status") == "processing" and age > self.s.job_budget_sec + self.s.kill_grace_sec + 5:
            return _failed(jid, LOST_ERROR)  # owning process is gone
        return obj

    def _sweep_disk(self):
        """Delete orphan temp/upload files and expired result files; convert
        stale 'processing' markers (from a dead process) into failed."""
        now = time.time()
        orphan_age = self.s.orphan_max_age_min * 60
        stale_age = self.s.job_budget_sec + self.s.kill_grace_sec + 5
        try:
            names = os.listdir(self.data_dir)
        except OSError:
            return
        with self.lock:
            live = {os.path.abspath(j["path"]) for j in self.jobs.values() if j["state"] != "final" and j["path"]}
        for name in names:
            p = os.path.join(self.data_dir, name)
            try:
                age = now - os.path.getmtime(p)
            except OSError:
                continue
            if _TEMP_RE.fullmatch(name):
                if age > orphan_age and p not in live:
                    _rm(p)
                continue
            m = _RESULT_RE.fullmatch(name)
            if not m:
                continue  # never touch files we did not create
            if age > self.s.job_ttl_sec:
                _rm(p)
            elif age > stale_age:
                jid = m.group(1)
                with self.lock:
                    if jid in self.jobs:
                        continue
                obj, _ = self._load_disk(jid)
                if obj and obj.get("status") == "processing":
                    self._write_json(jid, _failed(jid, LOST_ERROR))


# ---------------------------------------------------------------- helpers
def _done(jid: str, res) -> dict:
    """Strict contract check + public projection; any violation -> invalid_result."""
    try:
        pub = validate_result(res)
    except InvalidResult:
        return _failed(jid, "invalid_result")
    return {"job_id": jid, "status": "done", **pub}


def _failed(jid: str, error: str) -> dict:
    return {"job_id": jid, "status": "failed", "error": safe_code(error)}


def _parse_worker_output(jid: str, out_path: str) -> dict:
    try:
        with open(out_path, encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, ValueError):
        return _failed(jid, "worker_crashed")
    if isinstance(obj, dict) and obj.get("status") == "done":
        return _done(jid, obj)
    if not isinstance(obj, dict) or obj.get("status") != "failed":
        return _failed(jid, "invalid_result")
    return _failed(jid, safe_code(obj.get("error")))  # parent-side allow-list


def _settings_dict(s) -> dict:
    if dataclasses.is_dataclass(s):
        d = dataclasses.asdict(s)
    else:
        d = dict(vars(s))
    d["data_dir"] = os.path.abspath(d.get("data_dir", "."))
    return {k: v for k, v in d.items() if isinstance(v, (str, int, float, bool)) or v is None}


def _kill(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # whole group: decoder helpers too
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        log.error("pid %s did not exit after SIGKILL", proc.pid)


def _rm(path):
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass
