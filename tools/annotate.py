"""Local human annotation app. usage: python -m tools.annotate [--port 8790]

Loopback only (bind 127.0.0.1 + per-request client/Host check). Clips are shown as anonymous
indices; file names never reach the browser, stdout or logs. Output (all under local_private/):
  labels/<cid>.jsonl   events + observation reviews (clip index + sha256 prefix of the file)
  splits/<cid>.json    whole-clip train/holdout assignment (deterministic by hash)
  observations/<sha>.json  (input, written by the pipeline) list of {type,start_sec,end_sec,confidence}
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import threading

from .inventory import MEDIA_EXT, PRIVATE

try:
    from app.norms import NORMS
except Exception:  # noqa: BLE001
    NORMS = {}

CID_RE = re.compile(r"^C\d{2,}$")
SHA_RE = re.compile(r"^[0-9a-f]{16}$")
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
MIME = {".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo", ".mkv": "video/x-matroska"}
_lock = threading.Lock()


# ------------------------------------------------------------ pure logic (unit-tested)
def sha_prefix(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


def list_clips(directory):
    """Stable ordered clip paths of one candidate dir (index = position). Not recursive."""
    return sorted(os.path.join(directory, f) for f in os.listdir(directory)
                  if f.lower().endswith(MEDIA_EXT) and os.path.isfile(os.path.join(directory, f)))


def split_side(sha, ratio, seed="forte"):
    """Deterministic by hash: the same clip always lands on the same side for the same ratio/seed."""
    u = int(hashlib.sha256((seed + sha).encode()).hexdigest()[:8], 16) / 2 ** 32
    return "holdout" if u < ratio else "train"


def make_split(shas, ratio, seed="forte"):
    ratio = min(max(float(ratio), 0.0), 1.0)
    a = {s: split_side(s, ratio, seed) for s in shas}
    return {"ratio": ratio, "seed": seed, "clips": a,
            "counts": {"train": sum(v == "train" for v in a.values()),
                       "holdout": sum(v == "holdout" for v in a.values())}}


def _atomic_write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def read_labels(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    return out


def replace_clip_records(path, sha, kind, new):
    """Replace all records of `kind` for clip `sha`, keep everything else."""
    with _lock:
        keep = [r for r in read_labels(path) if not (r.get("sha") == sha and r.get("kind") == kind)]
        _atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep + new))


def clean_event(e, idx, sha):
    t = float(e["t"])
    if not (0 <= t < 86400):
        raise ValueError("t")
    norm = e.get("norm") or None
    if norm is not None and norm not in NORMS:
        raise ValueError("norm")
    return {"kind": "event", "clip": idx, "sha": sha, "t": round(t, 3),
            "note": str(e.get("note") or "")[:2000], "norm": norm}


def clean_review(r, idx, sha):
    if r.get("decision") not in ("accept", "reject"):
        raise ValueError("decision")
    return {"kind": "obs_review", "clip": idx, "sha": sha, "obs_index": int(r["obs_index"]),
            "type": str(r.get("type", ""))[:80], "start_sec": float(r["start_sec"]),
            "end_sec": float(r["end_sec"]), "decision": r["decision"]}


STATUSES = ("unreviewed", "reviewed_no_violation", "reviewed_violation")


def clean_status(status, idx, sha):
    """Explicit human decision only. 'unreviewed' = no record (cleared)."""
    if status not in STATUSES:
        raise ValueError("status")
    import time
    return {"kind": "clip_status", "type": "clip_status", "clip": idx, "sha": sha, "sha16": sha,
            "status": status, "ts": round(time.time(), 3)}


def status_map(labels):
    """sha -> status; clips without a clip_status record are unreviewed."""
    return {r["sha"]: r["status"] for r in labels if r.get("kind") == "clip_status"}


def status_counts(shas, labels):
    m = status_map(labels)
    c = {k: 0 for k in STATUSES}
    for s in shas:
        c[m.get(s, "unreviewed")] += 1
    return {"unreviewed": c["unreviewed"], "reviewed_nv": c["reviewed_no_violation"],
            "reviewed_v": c["reviewed_violation"]}


def read_observations(private, sha):
    p = os.path.join(private, "observations", sha + ".json")
    if not SHA_RE.match(sha) or not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        o = json.load(f)
    return [{"type": str(x.get("type", "")), "start_sec": float(x.get("start_sec", 0)),
             "end_sec": float(x.get("end_sec", 0)), "confidence": float(x.get("confidence", 0))}
            for x in o if isinstance(x, dict)] if isinstance(o, list) else []


# ------------------------------------------------------------ web app
def create_app(private=PRIVATE, extra_allowed_clients=()):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    allowed = LOOPBACK | set(extra_allowed_clients)
    hash_cache = {}

    @app.middleware("http")
    async def loopback_only(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        if (request.client is None or request.client.host not in allowed
                or (host not in LOOPBACK and host not in allowed)):
            return JSONResponse({"detail": "forbidden"}, status_code=403)
        return await call_next(request)

    def cmap():
        p = os.path.join(private, "inventory_map.json")
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

    def clips(cid):
        m = cmap()
        if not CID_RE.match(cid) or cid not in m or not os.path.isdir(m[cid]):
            return None
        return list_clips(m[cid])

    def sha_of(path):
        st = os.stat(path)
        k = (path, st.st_size, st.st_mtime_ns)
        if k not in hash_cache:
            hash_cache[k] = sha_prefix(path)
        return hash_cache[k]

    def lp(cid):
        return os.path.join(private, "labels", cid + ".jsonl")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE.replace("</main></body>", AT.HTML + AU.HTML + "</main></body>")

    @app.get("/api/candidates")
    def candidates():
        return {"ids": sorted(cmap()), "norms": {k: v[0] for k, v in NORMS.items()}}

    @app.get("/api/c/{cid}/clips")
    def clip_list(cid: str):
        cl = clips(cid)
        if cl is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        shas = [sha_of(p) for p in cl]
        sp = os.path.join(private, "splits", cid + ".json")
        split = json.load(open(sp)) if os.path.exists(sp) else None
        labs = read_labels(lp(cid))
        sm = status_map(labs)
        return {"clips": [{"index": i, "sha": s, "events": sum(r.get("sha") == s and r["kind"] == "event" for r in labs),
                           "status": sm.get(s, "unreviewed"),
                           "side": (split or {}).get("clips", {}).get(s)}
                          for i, s in enumerate(shas)],
                "status_counts": status_counts(shas, labs),
                "split": split and {"ratio": split["ratio"], "counts": split["counts"]}}

    @app.get("/api/c/{cid}/video/{idx}")
    def video(cid: str, idx: int, request: Request):
        cl = clips(cid)
        if cl is None or not 0 <= idx < len(cl):
            return JSONResponse({"detail": "not found"}, status_code=404)
        path = cl[idx]
        size = os.path.getsize(path)
        mime = MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
        start, end, status = 0, size - 1, 200
        m = re.match(r"^bytes=(\d*)-(\d*)$", request.headers.get("range", ""))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(size - int(m.group(2)), 0)
            end = min(end, size - 1)
            if start > end or start >= size:
                return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
            status = 206
        length = end - start + 1

        def gen():
            with open(path, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    b = f.read(min(1 << 20, left))
                    if not b:
                        break
                    left -= len(b)
                    yield b
        h = {"Accept-Ranges": "bytes", "Content-Length": str(length)}
        if status == 206:
            h["Content-Range"] = f"bytes {start}-{end}/{size}"
        return StreamingResponse(gen(), status_code=status, media_type=mime, headers=h)

    @app.get("/api/c/{cid}/clip/{idx}/state")
    def state(cid: str, idx: int):
        cl = clips(cid)
        if cl is None or not 0 <= idx < len(cl):
            return JSONResponse({"detail": "not found"}, status_code=404)
        sha = sha_of(cl[idx])
        labs = [r for r in read_labels(lp(cid)) if r.get("sha") == sha]
        return {"sha": sha, "events": [r for r in labs if r["kind"] == "event"],
                "reviews": [r for r in labs if r["kind"] == "obs_review"],
                "status": status_map(labs).get(sha, "unreviewed"),
                "status_counts": status_counts([sha_of(p) for p in cl], read_labels(lp(cid))),
                "observations": read_observations(private, sha),
                "imported": imported_for(cid, sha, labs)}

    def imported_for(cid, sha, labs):
        ip = os.path.join(private, "labels", cid + ".imported.jsonl")
        dec = {round(r["timestamp_sec"], 3): r["decision"] for r in labs if r["kind"] == "imported_review"}
        return [{"t": r["timestamp_sec"], "note": r.get("note", ""), "decision": dec.get(round(r["timestamp_sec"], 3))}
                for r in sorted(read_labels(ip), key=lambda r: r.get("timestamp_sec", 0))
                if r.get("kind") == "imported_event" and r.get("sha") == sha]

    @app.post("/api/c/{cid}/clip/{idx}/import_review")
    async def import_review(cid: str, idx: int, request: Request):
        cl = clips(cid)
        if cl is None or not 0 <= idx < len(cl):
            return JSONResponse({"detail": "not found"}, status_code=404)
        sha = sha_of(cl[idx])
        try:
            b = await request.json()
            t, d = round(float(b["timestamp_sec"]), 3), b["decision"]
            if d not in ("accept", "reject"):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"detail": "invalid"}, status_code=422)
        labs = [r for r in read_labels(lp(cid)) if r.get("sha") == sha]
        keep = [r for r in labs if r["kind"] == "imported_review" and round(r["timestamp_sec"], 3) != t]
        replace_clip_records(lp(cid), sha, "imported_review", keep + [
            {"kind": "imported_review", "clip": idx, "sha": sha, "timestamp_sec": t, "decision": d}])
        if d == "accept" and not any(abs(r["t"] - t) < 0.002 for r in labs if r["kind"] == "event"):
            ev = [r for r in labs if r["kind"] == "event"] + [
                {"kind": "event", "clip": idx, "sha": sha, "t": t, "note": "imported (confirmed by human)", "norm": None}]
            replace_clip_records(lp(cid), sha, "event", ev)
        return {"decision": d}

    from . import import_labels as IL
    from . import annotate_attrs as AT
    app.include_router(AT.make_router(private, clips))
    from . import annotate_audit as AU
    app.include_router(AU.make_router(private, clips))

    @app.get("/api/c/{cid}/import/preview")
    def import_preview(cid: str):
        # the user's OWN workbook, shown in the local browser only; never printed or logged
        path, n = IL.locate_xlsx(private, cid)
        if path is None:
            return JSONResponse({"detail": "xlsx", "count": n}, status_code=404)
        try:
            rows = IL.load_rows(path)
        except Exception:  # noqa: BLE001
            return JSONResponse({"detail": "unreadable"}, status_code=422)
        cut = lambda r: [IL._s(v)[:120] for v in r]  # noqa: E731
        return {"header": cut(rows[0]) if rows else [], "rows": [cut(r) for r in rows[1:4]],
                "ncols": max((len(r) for r in rows), default=0)}

    @app.get("/api/c/{cid}/import/needs_user")
    def import_needs_user(cid: str):
        # rows the importer could not resolve; the user's own cells, shown in the local browser only
        path, _n = IL.locate_xlsx(private, cid)
        if path is None:
            return []
        try:
            rows = IL.load_rows(path)
            recs = [r for r in read_labels(os.path.join(private, "labels", cid + ".imported.jsonl"))
                    if r.get("kind") == "imported_needs_user"]
            return [{"clip": r["clip"], "reason": r["reason"], "cells": [IL._s(v)[:200] for v in rows[r["row"]]]}
                    for r in recs]
        except Exception:  # noqa: BLE001
            return []

    @app.post("/api/c/{cid}/import/run")
    async def import_run(cid: str, request: Request):
        if clips(cid) is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        try:
            m = await request.json()
            if not isinstance(m, dict):
                raise ValueError
            _atomic_write(os.path.join(private, "import_mapping", cid + ".json"), json.dumps(m))
            cnt = IL.run_import(private, cid, m, bool(m.get("auto_ru")))
        except Exception:  # noqa: BLE001 - never echo details
            return JSONResponse({"detail": "import failed"}, status_code=422)
        print("import " + cid + ": " + " ".join(f"{k}={v}" for k, v in cnt.items()), flush=True)
        return cnt

    async def _put(cid, idx, request, kind, cleaner):
        cl = clips(cid)
        if cl is None or not 0 <= idx < len(cl):
            return JSONResponse({"detail": "not found"}, status_code=404)
        sha = sha_of(cl[idx])
        try:
            body = await request.json()
            new = [cleaner(e, idx, sha) for e in body]
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"detail": "invalid"}, status_code=422)
        replace_clip_records(lp(cid), sha, kind, new)
        return {"saved": len(new)}

    @app.put("/api/c/{cid}/clip/{idx}/events")
    async def put_events(cid: str, idx: int, request: Request):
        return await _put(cid, idx, request, "event", clean_event)

    @app.put("/api/c/{cid}/clip/{idx}/reviews")
    async def put_reviews(cid: str, idx: int, request: Request):
        return await _put(cid, idx, request, "obs_review", clean_review)

    @app.put("/api/c/{cid}/clip/{idx}/status")
    async def put_status(cid: str, idx: int, request: Request):
        cl = clips(cid)
        if cl is None or not 0 <= idx < len(cl):
            return JSONResponse({"detail": "not found"}, status_code=404)
        sha = sha_of(cl[idx])
        try:
            st = (await request.json())["status"]
            new = [] if st == "unreviewed" else [clean_status(st, idx, sha)]
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"detail": "invalid"}, status_code=422)
        replace_clip_records(lp(cid), sha, "clip_status", new)
        c = status_counts([sha_of(p) for p in cl], read_labels(lp(cid)))
        print(f"status {cid}: unreviewed={c['unreviewed']} reviewed_nv={c['reviewed_nv']} reviewed_v={c['reviewed_v']}", flush=True)
        return {"status": st, "status_counts": c}

    @app.post("/api/c/{cid}/split")
    async def split(cid: str, request: Request):
        cl = clips(cid)
        if cl is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        try:
            body = await request.json()
            sp = make_split([sha_of(p) for p in cl], body.get("ratio", 0.3), str(body.get("seed", "forte")))
        except (ValueError, TypeError):
            return JSONResponse({"detail": "invalid"}, status_code=422)
        _atomic_write(os.path.join(private, "splits", cid + ".json"), json.dumps(sp))
        print(f"split {cid}: train={sp['counts']['train']} holdout={sp['counts']['holdout']}", flush=True)
        return {"ratio": sp["ratio"], "counts": sp["counts"]}

    return app


PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Разметка (локально)</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--bg:#fff;--fg:#1a1a1a;--card:#f4f5f7;--ac:#0b5cff;--mut:#666}
@media(prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--card:#1c1e22;--ac:#6ea2ff;--mut:#9aa}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}main{max-width:960px;margin:0 auto;padding:16px}
.card{background:var(--card);border-radius:10px;padding:12px;margin:10px 0}video{width:100%;background:#000;border-radius:8px}
button,input,select{font:inherit;padding:6px 10px;border-radius:8px;border:1px solid var(--mut);background:var(--bg);color:var(--fg)}
button{background:var(--ac);color:#fff;border:0;cursor:pointer}.mut{color:var(--mut);font-size:13px}
#clips button{margin:2px;background:var(--card);color:var(--fg);border:1px solid var(--mut)}#clips .on{outline:2px solid var(--ac)}
</style></head><body><main><h1>Локальная разметка</h1>
<p class="mut">Только 127.0.0.1. Клипы показаны по индексам; имена файлов не отображаются. Метки: local_private/labels/.</p>
<div class="card">Набор: <select id="cid"></select>
 доля hold-out <input id="ratio" type="number" min="0" max="1" step="0.05" value="0.3" style="width:80px">
 <button id="split">Разбить по клипам целиком</button> <span id="splitinfo" class="mut"></span></div>
<div class="card"><span id="cnt" class="mut"></span><div id="clips"></div></div>
<div class="card"><b>Импорт разметки</b> <button id="imp_load">Показать мою таблицу</button>
 <div id="imp_prev" class="mut"></div><div id="imp_form" hidden>
 колонка файла <input id="m_fn" type="number" min="1" value="1" style="width:60px">
 тайм-код <input id="m_tc" type="number" min="1" style="width:60px" placeholder="нет">
 статус <input id="m_sc" type="number" min="1" style="width:60px" placeholder="нет">
 заметка <input id="m_nc" type="number" min="1" style="width:60px" placeholder="нет"><br>
 единица <select id="m_unit"><option>sec</option><option>mm:ss</option><option>hh:mm:ss</option><option value="frames">frames@fps</option><option value="excel_time">Excel time</option><option value="text_list">список времён (текст)</option></select>
 fps <input id="m_fps" type="number" value="30" style="width:60px">
 «нарушений нет» = значения (через ;, пусто = нет) <input id="m_nv" style="width:160px">
 «есть нарушение» = значения <input id="m_vv" style="width:160px">
 <button id="imp_run">Импортировать</button> <button id="imp_ru">Авто-импорт (русские метки)</button>
 <button id="imp_nu">Строки, требующие решения</button><div id="imp_nuw"></div> <span id="imp_res" class="mut"></span></div>
 <div class="mut">Пустая ячейка никогда не считается «нарушений нет». Импорт — неподтверждённый, правила не валидирует.</div></div>
<div id="work" hidden><video id="v" controls></video>
<div class="card"><b>Событие</b>: <button id="mark">Отметить текущее время</button>
 <input id="note" placeholder="заметка (необязательно)" style="width:40%"> <select id="norm"><option value="">норма-кандидат: нет</option></select>
 <div id="events"></div></div>
<div class="card"><b>Импортировано (не подтверждено)</b><div id="impev"></div></div>
<div class="card"><b>Статус клипа</b> <span id="stat" class="mut"></span><br>
 <button id="s_nv">Просмотр завершён: нарушений нет</button> <button id="s_v">Просмотр завершён: есть события</button>
 <button id="s_u">Сбросить</button> <span id="sugg" class="mut"></span>
 <div class="mut">Пустой список событий у непросмотренного клипа — не отрицательная метка. Статус ставит только человек.</div></div>
<div class="card"><b>Наблюдения пайплайна</b> <span class="mut">(принять/отклонить; на вердикт не влияют)</span><div id="obs"></div></div></div>
<script>
const $=id=>document.getElementById(id);let cid,idx,st;
const j=async(u,o)=>{const r=await fetch(u,o);if(!r.ok)throw new Error(r.status);return r.json()};
const put=(u,b)=>j(u,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)});
const tc=s=>Math.floor(s/60)+":"+String(Math.floor(s%60)).padStart(2,"0");
(async()=>{const c=await j("/api/candidates");$("cid").innerHTML=c.ids.map(i=>`<option>${i}</option>`).join("");
 $("norm").innerHTML+=Object.entries(c.norms).map(([k,v])=>`<option value="${k}">${k} (${v})</option>`).join("");
 $("cid").onchange=loadClips;if(c.ids.length)loadClips()})();
async function loadClips(){cid=$("cid").value;const d=await j(`/api/c/${cid}/clips`);
 $("cnt").textContent=`не просмотрено ${d.status_counts.unreviewed} / без нарушений ${d.status_counts.reviewed_nv} / с событиями ${d.status_counts.reviewed_v}`;
 $("splitinfo").textContent=d.split?`train ${d.split.counts.train} / holdout ${d.split.counts.holdout}`:"";
 $("clips").innerHTML=d.clips.map(c=>`<button data-i="${c.index}">#${c.index} [${c.status}]${c.side?" ("+c.side+")":""} · меток ${c.events}</button>`).join("");
 $("clips").onclick=e=>{const i=e.target.dataset.i;if(i!==undefined)openClip(+i)}}
async function openClip(i){idx=i;st=await j(`/api/c/${cid}/clip/${i}/state`);$("work").hidden=false;
 $("v").src=`/api/c/${cid}/video/${i}`;draw()}
function draw(){$("impev").innerHTML=(st.imported||[]).map((e,k)=>`<div style="border-left:4px solid #c80;padding-left:6px">imported, unconfirmed · ${tc(e.t)} ${esc(e.note||"")}
 <button data-seek="${e.t}">▶</button> <button data-ik="${k}" data-d="accept">Подтвердить</button> <button data-ik="${k}" data-d="reject">Отклонить</button> <b>${e.decision||""}</b></div>`).join("")||'<span class="mut">нет</span>';
$("stat").textContent=st.status;
 $("sugg").textContent=(st.events.length&&st.status==="unreviewed")?"есть события — предлагается «есть события» (подтвердите явно)":"";
$("events").innerHTML=st.events.map((e,k)=>`<div>${tc(e.t)} ${e.norm||""} ${esc(e.note)} <button data-k="${k}" class="del">×</button></div>`).join("");
 const rv=new Map(st.reviews.map(r=>[r.obs_index,r.decision]));
 $("obs").innerHTML=st.observations.length?st.observations.map((o,k)=>`<div>${esc(o.type)} ${tc(o.start_sec)}–${tc(o.end_sec)} (${o.confidence.toFixed(2)})
 <button data-k="${k}" data-d="accept">принять</button> <button data-k="${k}" data-d="reject">отклонить</button> <b>${rv.get(k)||""}</b>
 <button data-seek="${o.start_sec}">▶</button></div>`).join(""):'<span class="mut">нет файла наблюдений для этого клипа</span>'}
const esc=t=>String(t).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
$("mark").onclick=async()=>{st.events.push({t:$("v").currentTime,note:$("note").value,norm:$("norm").value||null});
 await put(`/api/c/${cid}/clip/${idx}/events`,st.events);$("note").value="";draw();loadClips()};
$("events").onclick=async e=>{if(e.target.className!=="del")return;st.events.splice(+e.target.dataset.k,1);
 await put(`/api/c/${cid}/clip/${idx}/events`,st.events);draw()};
$("obs").onclick=async e=>{const t=e.target;if(t.dataset.seek){$("v").currentTime=+t.dataset.seek;return}
 if(!t.dataset.d)return;const k=+t.dataset.k,o=st.observations[k];
 st.reviews=st.reviews.filter(r=>r.obs_index!==k).concat([{obs_index:k,type:o.type,start_sec:o.start_sec,end_sec:o.end_sec,decision:t.dataset.d}]);
 await put(`/api/c/${cid}/clip/${idx}/reviews`,st.reviews);draw()};
$("impev").onclick=async e=>{const t=e.target;if(t.dataset.seek){$("v").currentTime=+t.dataset.seek;return}
 if(!t.dataset.d)return;const ev=st.imported[+t.dataset.ik];
 await j(`/api/c/${cid}/clip/${idx}/import_review`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({timestamp_sec:ev.t,decision:t.dataset.d})});
 await openClip(idx)};
$("imp_load").onclick=async()=>{try{const d=await j(`/api/c/${$("cid").value}/import/preview`);
 $("imp_prev").innerHTML="<table border=1 cellpadding=3>"+[d.header,...d.rows].map(r=>"<tr>"+r.map(c=>"<td>"+esc(c)+"</td>").join("")+"</tr>").join("")+"</table>";
 $("imp_form").hidden=false}catch(e){$("imp_prev").textContent="таблица не найдена или нечитаема"}};
$("imp_run").onclick=async()=>{const n=id=>$(id).value?+$(id).value:null,l=id=>$(id).value.split(";").map(x=>x.trim()).filter(Boolean);
 const m={header_rows:1,filename_col:n("m_fn"),timecode_col:n("m_tc"),status_col:n("m_sc"),note_col:n("m_nc"),timecode_unit:$("m_unit").value,fps:+$("m_fps").value,no_violation_values:l("m_nv"),violation_values:l("m_vv")};
 try{const r=await j(`/api/c/${$("cid").value}/import/run`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(m)});
 $("imp_res").textContent=Object.entries(r).map(([k,v])=>k+"="+v).join(" ");loadClips()}catch(e){$("imp_res").textContent="ошибка импорта"}};
$("imp_ru").onclick=async()=>{try{const r=await j(`/api/c/${$("cid").value}/import/run`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({auto_ru:true})});
 $("imp_res").textContent=Object.entries(r).map(([k,v])=>k+"="+v).join(" ");loadClips()}catch(e){$("imp_res").textContent="авто-импорт невозможен"}};
$("imp_nu").onclick=async()=>{const d=await j(`/api/c/${$("cid").value}/import/needs_user`);
 $("imp_nuw").innerHTML=d.map(x=>`<div>клип #${x.clip} (${x.reason}): ${x.cells.map(esc).join(" | ")}</div>`).join("")||"нет"};
const setS=v=>async()=>{const r=await put(`/api/c/${cid}/clip/${idx}/status`,{status:v});st.status=v;draw();loadClips()};
$("s_nv").onclick=setS("reviewed_no_violation");$("s_v").onclick=setS("reviewed_violation");$("s_u").onclick=setS("unreviewed");
$("split").onclick=async()=>{await j(`/api/c/${cid}/split`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({ratio:+$("ratio").value})});loadClips()};
</script></main></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args(argv)
    import uvicorn
    n = len(json.load(open(os.path.join(PRIVATE, "inventory_map.json")))) if os.path.exists(
        os.path.join(PRIVATE, "inventory_map.json")) else 0
    print(f"annotate: {n} candidate dirs; http://127.0.0.1:{a.port}", flush=True)
    uvicorn.run(create_app(), host="127.0.0.1", port=a.port, access_log=False, log_level="warning")


if __name__ == "__main__":
    sys.exit(main())
