"""Local batch runner: API-equivalent analysis of one inventory candidate, for offline evaluation.

usage: python -m tools.run_local --candidate C02 [--split train|holdout|all] [--budget-sec 230] [--dry-run]

--split train|holdout selects files BEFORE opening any of them, from <LP>/splits/<ID>.json and the
clip-index -> sha map of <LP>/labels/<ID>.imported.jsonl (tools/local_safe.select_split_units);
only selected files are hashed (and re-checked). --split all hashes every clip of <LP>/inventory_map.json
entry <ID>. Unreadable clips are counted as failed: unreadable, never named.
<LP> = $LOCAL_PRIVATE_DIR or <project>/local_private.
Each clip is analysed in its OWN CHILD PROCESS (own session/process group) with the API's
deadline semantics: soft deadline passed to the pipeline, SIGKILL of the whole group at the
hard budget. Writes
  <LP>/predictions/<sha16>.json   API job body {status, verdict, violations} or {status, error}
                                  (public keys only, via app.worker.validate_result / safe_error)
  <LP>/observations/<sha16>.json  [{type, start_sec, end_sec, confidence}] (pipeline observations)
Rule activation follows the same env as the API (RULESET, ALLOW_UNVALIDATED_RULES, ...).

PRIVACY CONTRACT: stdout carries ONLY counts and elapsed-time statistics. Never clip names,
paths, shas, descriptions or error text. Children's stdout/stderr go to DEVNULL.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:  # script form `python tools/run_local.py` (the -m form already has it)
    sys.path.insert(0, ROOT)
MEDIA_EXT = (".mov", ".mp4", ".avi", ".mkv")
CID_RE = re.compile(r"^C\d{2,4}$")
SHA_RE = re.compile(r"^[0-9a-f]{16}$")


class Refused(Exception):
    """Static, content-free reason (safe to print)."""


def private_dir():
    return os.environ.get("LOCAL_PRIVATE_DIR") or os.path.join(ROOT, "local_private")


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def list_clips(cid, priv):
    if not CID_RE.match(cid or ""):
        raise Refused("candidate id must look like C01")
    try:
        with open(os.path.join(priv, "inventory_map.json"), encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        raise Refused("inventory map missing or unreadable; run tools.inventory first") from None
    d = m.get(cid) if isinstance(m, dict) else None
    if not isinstance(d, str) or not os.path.isdir(d):
        raise Refused("unknown candidate id or directory unavailable")
    out = []
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if os.path.splitext(fn)[1].lower() in MEDIA_EXT and os.path.isfile(p) and not os.path.islink(p):
            out.append(p)
    return out


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def _kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_clip(clip, pred_path, obs_dir, budget):
    """Parent side: spawn, enforce hard deadline, re-validate the child's output."""
    from app import worker
    deadline = time.time() + budget
    env = dict(os.environ, JOB_DEADLINE=repr(deadline - min(1.0, budget * 0.05)),
               JOB_PARENT_PID=str(os.getpid()))
    env.pop("JOB_SETTINGS", None)
    tmp_out = pred_path + ".child"
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    # paths go over stdin (argv is visible to other local users via ps)
    proc = subprocess.Popen([sys.executable, "-m", "tools.run_local", "--child"],
                            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    try:
        proc.stdin.write((json.dumps({"clip": clip, "out": tmp_out, "obs_dir": obs_dir}) + "\n").encode())
        proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=max(0.0, deadline - time.time()))
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        proc.wait()
        out = {"status": "failed", "error": "time_budget_exceeded"}
    else:
        _kill_group(proc)  # reap any helper left in the group
        try:
            with open(tmp_out, encoding="utf-8") as f:
                raw = json.load(f)
            if raw.get("status") == "done":
                out = {"status": "done", **worker.validate_result(raw)}
            elif raw.get("status") == "failed":
                out = {"status": "failed", "error": worker.safe_code(raw.get("error"))}
            else:
                out = {"status": "failed", "error": "invalid_result"}
        except (OSError, ValueError, AttributeError, worker.InvalidResult):
            out = {"status": "failed", "error": "worker_crashed" if proc.returncode else "invalid_result"}
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
    _write_json(pred_path, out)
    return out


def child(clip, out_path, obs_dir):
    """Child side: same analysis + public projection as app.worker.run, plus observations."""
    from app import config, pipeline, worker
    deadline = float(os.environ["JOB_DEADLINE"]) if os.environ.get("JOB_DEADLINE") else None
    ppid = os.environ.get("JOB_PARENT_PID", "")
    if ppid.isdigit() and hasattr(worker, "_watch_parent"):
        worker._watch_parent(int(ppid), deadline)
    try:
        settings = config.get()
        if hasattr(worker, "_limit_memory"):
            worker._limit_memory(getattr(settings, "worker_mem_mb", 0))
        if hasattr(worker, "_limit_cpu"):
            worker._limit_cpu(deadline)
        res = pipeline.analyze(clip, settings, deadline=deadline, observations_dir=obs_dir)
        try:
            out = {"status": "done", **worker.validate_result(res)}
        except worker.InvalidResult:
            out = {"status": "failed", "error": "invalid_result"}
    except Exception as e:  # noqa: BLE001 - any error -> failed, never no_violation
        out = {"status": "failed", "error": worker.safe_error(e)}
    _write_json(out_path, out)
    return 0


def _stats(xs):
    if not xs:
        return "n/a"
    return f"min {min(xs):.1f} / median {statistics.median(xs):.1f} / max {max(xs):.1f}"


def main(argv=None):
    from tools.local_safe import run_guarded
    return run_guarded(_main, argv)


def _hash_all(clips):
    """-> ([(clip, sha16)], n_unreadable); an unreadable clip is counted, never named."""
    ok, bad = [], 0
    for c in clips:
        try:
            ok.append((c, sha16(c)))
        except OSError:
            bad += 1
    return ok, bad


def _main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--child"]:
        job = json.loads(sys.stdin.readline())
        return child(job["clip"], job["out"], job["obs_dir"])
    ap = argparse.ArgumentParser(description="local batch runner (counts-only output)")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--split", choices=("train", "holdout", "all"), default="all")
    ap.add_argument("--budget-sec", type=float, default=230.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    priv = private_dir()
    try:
        if not 5 <= a.budget_sec <= 3600 and not a.dry_run:
            raise Refused("--budget-sec must be within 5..3600")
        unreadable, skipped = 0, 0
        if a.split == "all":
            try:
                clips = list_clips(a.candidate, priv)
            except OSError:
                raise Refused("candidate directory unreadable") from None
            n_candidate = len(clips)
            selected = None if a.dry_run else []
            if not a.dry_run:
                selected, unreadable = _hash_all(clips)
        else:  # select by split side BEFORE opening any file; only selected files are hashed
            from tools.local_safe import Refused as SelRefused, select_split_units
            try:
                units, c = select_split_units(priv, a.candidate, a.split, dry_run=a.dry_run)
            except SelRefused as e:
                raise Refused(str(e)) from None
            n_candidate = None
            selected = [(path, sha) for sha, path, _st, _ev in units]
            unreadable, skipped = c["unreadable"], c["skipped"] + c["index_mismatch"]
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    n = len(selected) if selected is not None else n_candidate
    if n_candidate is not None:
        print(f"clips_in_candidate: {n_candidate}")
    print(f"clips_selected ({a.split}): {n}")
    if skipped:
        print(f"units_skipped (label/index checks failed, not opened or not analysed): {skipped}")
    if unreadable:
        print(f"clips_unreadable (not hashed, not classified, counted as failed: unreadable): {unreadable}")
    if a.dry_run:
        return 0
    pred_dir, obs_dir = os.path.join(priv, "predictions"), os.path.join(priv, "observations")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(obs_dir, exist_ok=True)
    status, errors, verdicts, elapsed, obs_written = Counter(), Counter(), Counter(), [], 0
    status["failed"] += unreadable
    if unreadable:
        errors["unreadable"] += unreadable
    for clip, sh in selected:
        obs_path = os.path.join(obs_dir, sh + ".json")
        t0 = time.time()
        try:
            if os.path.exists(obs_path):
                os.remove(obs_path)  # never leave a stale observation file from an earlier run
            out = run_clip(clip, os.path.join(pred_dir, sh + ".json"), obs_dir, a.budget_sec)
        except OSError:
            out = {"status": "failed", "error": "unreadable"}
        except Exception:  # noqa: BLE001 - counted, never printed
            out = {"status": "failed", "error": "internal"}
        elapsed.append(time.time() - t0)
        status[out["status"]] += 1
        if out["status"] == "done":
            verdicts[out["verdict"]] += 1
        else:
            errors[out["error"]] += 1
        obs_written += os.path.exists(obs_path)
    print(f"done: {status['done']} (violation: {verdicts['violation']}, no_violation: {verdicts['no_violation']})")
    print(f"failed: {status['failed']}" + (" " + json.dumps(dict(sorted(errors.items()))) if errors else ""))
    print(f"observation_files_written: {obs_written}")
    print(f"elapsed_sec: {_stats(elapsed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
