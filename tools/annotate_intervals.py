"""STAGE 8 local interval annotation: a SEPARATE minimal loopback app (does not mount tools/annotate routers).
Storage: <PRIVATE>/labels/<cid>.intervals.v1.jsonl, append-only revision log
{task_id, sha16, role, rev, ts, payload}; task_id = sha256(sha16|role|idx)[:16]. Train-only tasks: 5 positives at
start, control tasks on demand (<=2 per positive, lowest-sha16 unassigned train negative, each negative <=1 positive).
Nothing about clips (names, paths, frames, labels) is ever printed or returned; ids are opaque hashes.
  python -m tools.annotate_intervals --cid C02 --port 8791
  python -m tools.annotate_intervals --cid C02 --preflight     (aggregates only)"""
import argparse
import hashlib, json, math, os, re, threading, time  # noqa: E401
from urllib.parse import urlparse

from .annotate import MIME, _atomic_write
from .inventory import PRIVATE, _probe_one

CID_RE = re.compile(r"^C\d{2,4}$")
TID_RE = re.compile(r"^[0-9a-f]{16}$")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
VIS = ("good", "partial", "poor")
CONF = ("sure", "unsure")
ANSWERS = (None, "not_comparable", "not_visible", "unknown")
MAX_PER_POS, MAX_IV = 2, 50
IMMUTABLE = (  # name, path relative to PRIVATE ({cid} substituted), expected sha16
    ("attrs", "labels/{cid}.attrs.jsonl", "ab7a86b165346141"), ("R1", "labels/{cid}.audit_r1.jsonl", "09247337a21dc401"),
    ("map", "inventory_map.json", "e413108f8b7f0080"), ("stage4", "train_probe/stage4_summary.json", "284e70c08e70c02b"),
    ("stage5", "train_probe/stage5_summary.json", "5d399b48ca612b9e"), ("stage7", "train_probe/stage7_summary.json", "61280551b417cc3e"),
    ("tar", "release/forte_code_20261018.tar.gz", "1bc28585d2d36e7d"), ("demo", "demo/forte_demo.mp4", "d8f6fc948bf2ca7f"))
_lock = threading.Lock()

class Bad(Exception):
    """Validation failure; message is a static code (safe to return)."""

def task_id(sha, role, idx):
    return hashlib.sha256(f"{sha}|{role}|{idx}".encode()).hexdigest()[:16]

# ------------------------------------------------------------------ validation
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def _bool(d, k):
    v = d.get(k, False)
    if not isinstance(v, bool):
        raise Bad("bad_flag")
    return v

def _only(d, keys):
    if not isinstance(d, dict) or set(d) - set(keys):
        raise Bad("bad_fields")

def _t1(v, dur, flag):
    """One boundary: number in [0, dur]; None = pending draft, or unknown when flagged (flag with a number is bad)."""
    if v is None:
        return None
    if flag or not _num(v) or not 0 <= v <= dur + 1e-6:
        raise Bad("bad_times")
    return round(min(float(v), dur), 3)

def _times(d, dur, unk_ok):
    su, eu = (_bool(d, "start_unknown"), _bool(d, "end_unknown")) if unk_ok else (False, False)
    if su and eu:
        raise Bad("bad_times")
    s, e = _t1(d.get("start"), dur, su), _t1(d.get("end"), dur, eu)
    if s is not None and e is not None and not s < e:
        raise Bad("bad_times")
    return s, e, su, eu

def _complete(i, unk_ok):
    return all(i[k] is not None or (unk_ok and i.get(k + "_unknown")) for k in ("start", "end"))

def _enum(d, k, allowed):
    if d.get(k) not in allowed:
        raise Bad("bad_enum")
    return d[k]

def clean_pos_iv(d, dur):
    _only(d, ("start", "end", "start_unknown", "end_unknown", "start_uncertain", "end_uncertain", "visibility", "confidence"))
    s, e, su, eu = _times(d, dur, True)
    return {"start": s, "end": e, "start_unknown": su, "end_unknown": eu, "start_uncertain": _bool(d, "start_uncertain"),
            "end_uncertain": _bool(d, "end_uncertain"), "visibility": _enum(d, "visibility", VIS),
            "confidence": _enum(d, "confidence", CONF)}

def clean_ctl_iv(d, dur):
    _only(d, ("start", "end", "visibility", "confidence", "comparable_visible", "no_violation"))
    s, e, _a, _b = _times(d, dur, False)
    return {"start": s, "end": e, "visibility": _enum(d, "visibility", VIS), "confidence": _enum(d, "confidence", CONF),
            "comparable_visible": _bool(d, "comparable_visible"), "no_violation": _bool(d, "no_violation")}

def _list(v, fn, dur):
    if not isinstance(v, list) or len(v) > MAX_IV or not all(isinstance(x, dict) for x in v):
        raise Bad("bad_list")
    return [fn(x, dur) for x in v]

def clean_payload(role, p, dur, prev):
    """Validate a client payload; server-owned fields (viewed, parent) come from prev only."""
    if role == "positive":
        _only(p, ("viewed", "media_error", "intervals", "clip_flags", "same_clip_controls", "control_not_found", "done"))
        cf = p.get("clip_flags", {})
        _only(cf, ("not_visible", "unknown", "event_unconfirmed"))
        out = {"intervals": _list(p.get("intervals", []), clean_pos_iv, dur),
               "clip_flags": {k: _bool(cf, k) for k in ("not_visible", "unknown", "event_unconfirmed")},
               "same_clip_controls": _list(p.get("same_clip_controls", []), clean_ctl_iv, dur),
               "control_not_found": _bool(p, "control_not_found"), "done": _bool(p, "done")}
        if out["intervals"] and any(out["clip_flags"].values()):
            raise Bad("unknown_with_intervals")
        if out["done"] and not (out["intervals"] or any(out["clip_flags"].values())):
            raise Bad("need_interval_or_flag")
        if out["done"] and not (all(_complete(i, True) for i in out["intervals"])
                                and all(_complete(i, False) for i in out["same_clip_controls"])):
            raise Bad("unfinished_interval")
    else:
        _only(p, ("viewed", "media_error", "parent", "intervals", "answer", "done"))
        out = {"intervals": _list(p.get("intervals", []), clean_ctl_iv, dur), "answer": p.get("answer"),
               "done": _bool(p, "done")}
        if out["answer"] not in ANSWERS:
            raise Bad("bad_enum")
        if out["answer"] and out["intervals"]:
            raise Bad("answer_and_intervals")
        if out["done"] and not out["answer"] and not (out["intervals"] and all(
                i["comparable_visible"] and i["no_violation"] and _complete(i, False) for i in out["intervals"])):
            raise Bad("need_confirmations")
        out["parent"] = (prev or {}).get("parent")
    out["viewed"] = bool((prev or {}).get("viewed", False))
    out["media_error"] = bool((prev or {}).get("media_error", False))
    return out

def classify(role, p):
    """-> None (not done) | positive: 'intervals'|'unknown' | control: 'normal'|'unclear'."""
    if not p or not p.get("done"):
        return None
    if role == "positive":
        return "unknown" if any(p["clip_flags"].values()) else "intervals"
    if p.get("answer"):
        return "unclear"
    return "normal_confident" if all(i["visibility"] == "good" and i["confidence"] == "sure" for i in p["intervals"]) else "normal_low"

# ------------------------------------------------------------------ storage
def log_path(private, cid):
    return os.path.join(private, "labels", f"{cid}.intervals.v1.jsonl")

def read_log(private, cid):
    """-> {task_id: latest record}. Malformed lines are skipped."""
    latest = {}
    try:
        with open(log_path(private, cid), encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and isinstance(r.get("task_id"), str) and isinstance(r.get("rev"), int):
                    if r["rev"] >= latest.get(r["task_id"], {"rev": -1})["rev"]:
                        latest[r["task_id"]] = r
    except OSError:
        pass
    return latest

def append_log(private, cid, rec):
    p = log_path(private, cid)
    try:
        with open(p, encoding="utf-8") as f:
            old = f.read()
    except OSError:
        old = ""
    if old and not old.endswith("\n"):
        old += "\n"
    _atomic_write(p, old + json.dumps(rec, sort_keys=True) + "\n")

# ------------------------------------------------------------------ tasks
def build_units(private, cid):
    """-> (positives [(sha, path)], negatives [(sha, path)], holdout shas). Train side only is ever opened."""
    from .local_safe import select_split_units
    from .evaluate import load_split
    units, _ = select_split_units(private, cid, "train")
    pos = [(s, p) for s, p, st, _e in units if st == "reviewed_violation"]
    neg = [(s, p) for s, p, st, _e in units if st == "reviewed_no_violation"]
    side = load_split(os.path.join(private, "splits", f"{cid}.json"))
    return sorted(pos), sorted(neg), {s for s, v in side.items() if v == "holdout"}

def make_assignment(pos, neg):
    """Stable one-time rule: positives by sha16; each takes the two lowest-sha16 still unassigned train negatives."""
    free = [s for s, _ in sorted(neg)]
    out = {}
    for i, (s, _p) in enumerate(sorted(pos)):
        out[task_id(s, "positive", i)] = [free.pop(0) for _ in range(min(MAX_PER_POS, len(free)))]
    return out

def counters(private, cid, pos, neg):
    lg = read_log(private, cid)
    amax = sum(len(v) for v in (lg.get("_assign", {}).get("payload") or {}).values())
    c = {"positives": {"total": len(pos), "shown": 0, "completed_with_intervals": 0, "completed_unknown": 0},
         "controls": {"generated": 0, "max": amax, "shown": 0, "confirmed_normal_confident": 0,
                      "confirmed_normal_low_confidence": 0, "not_comparable_or_unknown": 0, "total": amax}}
    for r in lg.values():
        p = r.get("payload") or {}
        if r.get("role") == "positive":
            c["positives"]["shown"] += bool(p.get("viewed"))
            k = classify("positive", p)
            c["positives"]["completed_with_intervals"] += k == "intervals"
            c["positives"]["completed_unknown"] += k == "unknown"
        elif r.get("role") == "control":
            c["controls"]["generated"] += 1
            c["controls"]["shown"] += bool(p.get("viewed"))
            k = classify("control", p)
            c["controls"]["confirmed_normal_confident"] += k == "normal_confident"
            c["controls"]["confirmed_normal_low_confidence"] += k == "normal_low"
            c["controls"]["not_comparable_or_unknown"] += k == "unclear"
    return c

def range_response(path, rng):
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    try:
        fh = open(path, "rb")   # opened BEFORE the response starts, so failures map to a static code
        size = os.fstat(fh.fileno()).st_size
    except OSError:
        return JSONResponse({"detail": "unavailable"}, status_code=404)
    start, end, status = 0, size - 1, 200
    m = re.match(r"^bytes=(\d*)-(\d*)$", rng or "")
    if m and (m.group(1) or m.group(2)):
        if m.group(1):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
        else:
            start = max(size - int(m.group(2)), 0)
        end = min(end, size - 1)
        if start > end or start >= size:
            fh.close()
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        status = 206
    length = end - start + 1

    def gen():
        with fh as f:
            f.seek(start)
            left = length
            while left > 0 and (b := f.read(min(1 << 20, left))):
                left -= len(b)
                yield b
    h = {"Accept-Ranges": "bytes", "Content-Length": str(length), "Cache-Control": "no-store"}
    if status == 206:
        h["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(gen(), status_code=status, headers=h,
                             media_type=MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream"))

# ------------------------------------------------------------------ app
def create_app(private=PRIVATE, cid="C02"):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from .annotate_intervals_ui import PAGE
    if not CID_RE.match(cid):
        raise ValueError("cid")
    pos, neg, holdout = build_units(private, cid)
    paths = {}                       # task_id -> (sha, role, idx, path)  (server-side only; allowlist)
    pos_ids = []
    for i, (s, p) in enumerate(pos):
        t = task_id(s, "positive", i)
        paths[t] = (s, "positive", i, p)
        pos_ids.append(t)
    neg_by_sha = dict(neg)
    with _lock:   # fixed once; resume/visit order never changes pairs
        if "_assign" not in read_log(private, cid):
            append_log(private, cid, {"task_id": "_assign", "sha16": "", "role": "assignment", "rev": 1, "ts": 0,
                                      "payload": make_assignment(pos, neg)})
    durs = {}
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def err(code, status=422, **kw):
        return JSONResponse({"detail": code, **kw}, status_code=status)

    def nf():
        return err("not found", 404)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in LOOPBACK:
            return JSONResponse({"detail": "forbidden"}, status_code=403)
        if request.method not in ("GET", "HEAD"):
            sfs, org = request.headers.get("sec-fetch-site"), request.headers.get("origin")
            ok = sfs == "same-origin" or (sfs is None and org is not None
                                          and urlparse(org).netloc == request.headers.get("host"))
            if not ok:
                return JSONResponse({"detail": "forbidden"}, status_code=403)
        try:
            return await call_next(request)
        except Exception:  # noqa: BLE001 - static code only; exception text may contain real paths
            return JSONResponse({"detail": "error"}, status_code=500)

    def known(c, t):
        """task path entry for an allowlisted id, else None (also rebuilds generated controls from the log)."""
        if c != cid or not TID_RE.match(t or ""):
            return None
        if t in paths:
            return paths[t]
        r = read_log(private, cid).get(t)
        if r and r.get("role") == "control" and r.get("sha16") in neg_by_sha:
            e = (r["sha16"], "control", int(r["payload"].get("k", 0)), neg_by_sha[r["sha16"]])
            if task_id(e[0], "control", e[2]) == t:
                paths[t] = e
                return e
        return None

    def duration(path):
        if path not in durs:
            v = _probe_one(path)
            durs[path] = v[1] / v[0] if v and v[0] > 0 and v[1] > 0 else None
        return durs[path]

    def view(t, e, lg):
        r = lg.get(t)
        pl = r["payload"] if r else None
        v = {"task_id": t, "role": e[1], "rev": r["rev"] if r else 0, "payload": pl, "parent": (pl or {}).get("parent"),
             "ord": pos_ids.index(t if e[1] == "positive" else (pl or {}).get("parent", t)) + 1 if e[1] == "positive" or pl else 0}
        if e[1] == "control" and pl:
            v.update(cand=pl.get("k", 0) + 1,
                     of=len((lg.get("_assign", {}).get("payload") or {}).get(pl.get("parent"), [])))
        return v

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE.replace("__CID__", cid)

    @app.get("/api/i/{c}/tasks")
    def tasks(c: str):
        if c != cid:
            return nf()
        for t in list(read_log(private, cid)):
            known(cid, t)
        lg = read_log(private, cid)
        out = []
        for t, e in paths.items():
            v = view(t, e, lg)
            v["state"] = classify(e[1], v["payload"])
            v["shown"] = bool((v["payload"] or {}).get("viewed"))
            v.pop("payload")
            out.append(v)
        out.sort(key=lambda v: (v["ord"], v["role"] != "positive", v.get("cand", 0)))
        return {"cid": cid, "tasks": out, "counters": counters(private, cid, pos, neg)}

    @app.get("/api/i/{c}/task/{t}")
    def get_task(c: str, t: str):
        e = known(c, t)
        if not e:
            return nf()
        d = duration(e[3])
        v = view(t, e, read_log(private, cid))
        v["duration"] = d
        return v

    @app.put("/api/i/{c}/task/{t}")
    async def put_task(c: str, t: str, request: Request):
        e = known(c, t)
        if not e:
            return nf()
        try:
            body = await request.json()
        except ValueError:
            return err("bad_json")
        if not isinstance(body, dict) or set(body) - {"rev", "payload"} or not isinstance(body.get("rev"), int) \
                or isinstance(body.get("rev"), bool) or not isinstance(body.get("payload"), dict):
            return err("bad_body")
        dur = duration(e[3])
        if not dur:
            return err("duration_unavailable")
        with _lock:
            cur = read_log(private, cid).get(t)
            cur_rev = cur["rev"] if cur else 0
            if body["rev"] != cur_rev:
                return JSONResponse({"detail": "stale", "current_rev": cur_rev}, status_code=409)
            try:
                pl = clean_payload(e[1], body["payload"], dur, cur and cur["payload"])
            except Bad as b:
                return err(str(b))
            if e[1] == "control":
                pl["k"] = (cur or {}).get("payload", {}).get("k", e[2])
            append_log(private, cid, {"task_id": t, "sha16": e[0], "role": e[1], "rev": cur_rev + 1,
                                      "ts": round(time.time(), 3), "payload": pl})
        return {"rev": cur_rev + 1, "counters": counters(private, cid, pos, neg)}

    async def flag_event(c, t, key, request):
        e = known(c, t)
        if not e:
            return nf()
        try:
            body = await request.json()
        except ValueError:
            return err("bad_json")
        if not isinstance(body, dict) or set(body) - {"rev"} or not isinstance(body.get("rev"), int) or isinstance(body.get("rev"), bool):
            return err("bad_body")
        with _lock:
            cur = read_log(private, cid).get(t)
            pl = dict((cur or {}).get("payload") or {})
            if body["rev"] != (cur["rev"] if cur else 0):   # never adopt a foreign revision
                return JSONResponse({"detail": "stale", "current_rev": cur["rev"] if cur else 0}, status_code=409)
            if pl.get(key):
                return {"rev": cur["rev"]}
            if not pl:
                pl = {"intervals": [], "done": False, **(
                    {"clip_flags": {"not_visible": False, "unknown": False, "event_unconfirmed": False},
                     "same_clip_controls": [], "control_not_found": False} if e[1] == "positive" else
                    {"answer": None, "parent": None, "k": e[2]})}
            pl[key] = True
            rev = (cur["rev"] if cur else 0) + 1
            append_log(private, cid, {"task_id": t, "sha16": e[0], "role": e[1], "rev": rev,
                                      "ts": round(time.time(), 3), "payload": pl})
        return {"rev": rev}

    @app.post("/api/i/{c}/task/{t}/viewed")   # client: video 'loadeddata' only
    async def viewed(c: str, t: str, request: Request):
        return await flag_event(c, t, "viewed", request)

    @app.post("/api/i/{c}/task/{t}/media_error")  # recorded, does NOT count as shown
    async def media_error(c: str, t: str, request: Request):
        return await flag_event(c, t, "media_error", request)

    @app.post("/api/i/{c}/task/{t}/next_control")
    def next_control(c: str, t: str):
        e = known(c, t)
        if not e or e[1] != "positive":
            return nf()
        with _lock:
            lg = read_log(private, cid)
            r = lg.get(t)
            if not r or not r["payload"].get("control_not_found"):
                return err("not_found_not_set")
            mine = sorted((x for x in lg.values() if x["role"] == "control" and x["payload"].get("parent") == t),
                          key=lambda x: x["payload"].get("k", 0))
            table = (lg.get("_assign", {}).get("payload") or {}).get(t, [])
            if len(mine) >= len(table):
                return err("no_candidate")
            if mine and not mine[-1]["payload"].get("done"):
                return err("previous_control_unfinished")
            if mine and classify("control", mine[-1]["payload"]) == "normal_confident":
                return err("not_needed")
            cand = table[len(mine)]
            if cand not in neg_by_sha:
                return err("no_candidate")
            k = len(mine)
            ct = task_id(cand, "control", k)
            append_log(private, cid, {"task_id": ct, "sha16": cand, "role": "control", "rev": 1,
                                      "ts": round(time.time(), 3),
                                      "payload": {"intervals": [], "answer": None, "done": False, "viewed": False,
                                                  "parent": t, "k": k}})
        known(cid, ct)
        return {"task_id": ct, "counters": counters(private, cid, pos, neg)}

    @app.get("/api/i/{c}/video/{t}")
    def video(c: str, t: str, request: Request):
        e = known(c, t)
        if not e:
            return nf()
        return range_response(e[3], request.headers.get("range"))

    return app

# ------------------------------------------------------------------ CLI
def preflight(private, cid):
    from .annotate import sha_prefix
    pos, neg, holdout = build_units(private, cid)
    ok = 0
    for _s, p in pos:
        v = _probe_one(p)
        ok += bool(v and v[0] > 0 and v[1] > 0 and v[2] > 0 and v[3] > 0)
    print("positives_generated", len(pos))
    print("unique", len({s for s, _ in pos}))
    print("readable_by_probe", ok)
    print("holdout_in_tasks", len({s for s, _ in pos + neg} & holdout))
    print("control_candidates_max", sum(len(v) for v in make_assignment(pos, neg).values()))
    for name, rel, exp in IMMUTABLE:
        fp = os.path.join(private, rel.format(cid=cid))
        try:
            st = "unchanged" if sha_prefix(fp) == exp else "CHANGED"
        except OSError:
            st = "CHANGED(missing)"
        print("hash", name, st)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", required=True)
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args(argv)
    from .local_safe import run_guarded

    def go(_):
        if not CID_RE.match(a.cid):
            raise ValueError("cid")
        if a.preflight:
            preflight(PRIVATE, a.cid)
            return 0
        import uvicorn
        app = create_app(PRIVATE, a.cid)
        fd = os.open(os.path.join(PRIVATE, "stage8_server.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.dup2(fd, 2)   # tracebacks (may hold real paths) go to a local 0600 log, never to the terminal
        print(f"http://127.0.0.1:{a.port}/  (loopback only)")
        uvicorn.run(app, host="127.0.0.1", port=a.port, access_log=False, log_level="warning")
        return 0
    return run_guarded(go)

if __name__ == "__main__":
    raise SystemExit(main())
