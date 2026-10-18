"""Stage-2 blind attribute annotation (mounted by tools/annotate.py). Local, loopback only.

Prompts are generated for TRAIN clips only (local_private/splits/<cid>.json); holdout clip files are never
opened (shas/idx come from the imported-label file, durations are probed for train clips only). The tag
label|control lives only in local_private/attrs_prompts/<cid>.json and is never sent to the answering UI.
Answers: local_private/labels/<cid>.attrs.jsonl  {"kind":"attr",clip,sha,t,probe,answers,ts}, one per (sha,t).
stdout: counts only. /attrs/summary: counts per question/option split by probe tag (no times, no shas).
"""
import json
import os
import random
import time

from .annotate import _atomic_write, _lock, read_labels

QUESTIONS = [
    ("q1", "Людей в рабочей зоне кассира", [("0", "0"), ("1", "1"), ("2plus", "2+")]),
    ("q2", "Руки кассира видны", [("both", "обе"), ("one", "одна"), ("none", "нет"), ("unseen", "не видно")]),
    ("q3", "Банкноты на стойке видны", [("yes", "да"), ("no", "нет"), ("unseen", "не видно")]),
    ("q4", "Идёт пересчёт наличных", [("yes", "да"), ("no", "нет"), ("unseen", "не видно")]),
    ("q5", "Ящик/бокс/сейф открыт", [("yes", "да"), ("no", "нет"), ("unseen", "не видно")]),
    ("q6", "Передача предмета через стойку в ±5 с", [("yes", "да"), ("no", "нет"), ("unseen", "не видно")]),
    ("q7", "Клиент у стойки", [("yes", "да"), ("no", "нет"), ("unseen", "не видно")]),
    ("q8", "Кассир", [("facing", "лицом к стойке"), ("turned", "отвернулся"), ("unseen", "не виден")]),
]
VALID = {q: {k for k, _ in opts} for q, _t, opts in QUESTIONS}
MIN_AWAY = 20.0


def _paths(private, cid):
    return (os.path.join(private, "attrs_prompts", cid + ".json"),
            os.path.join(private, "labels", cid + ".attrs.jsonl"))


def generate_prompts(private, cid, durations, seed="forte-attrs"):
    """Pure-ish generator. durations: {clip_idx: seconds|None} for TRAIN clips only (caller probes only those).
    Returns (prompt list in shuffled order WITH tags, counts). Raises LookupError if no split/imports."""
    sp = os.path.join(private, "splits", cid + ".json")
    ip = os.path.join(private, "labels", cid + ".imported.jsonl")
    if not (os.path.exists(sp) and os.path.exists(ip)):
        raise LookupError("split or imported labels missing")
    side = json.load(open(sp, encoding="utf-8"))["clips"]
    train = {s for s, v in side.items() if v == "train"}
    recs = read_labels(ip)
    idx, st, times = {}, {}, {}
    for r in recs:
        sha = r.get("sha")
        if sha not in train:
            continue
        idx.setdefault(sha, r["clip"])
        if r["kind"] == "imported_clip_status":
            st.setdefault(sha, set()).add(r["status"])
        elif r["kind"] == "imported_event":
            times.setdefault(sha, set()).add(r["timestamp_sec"])
    pos = sorted(s for s in train if st.get(s) == {"reviewed_violation"})
    neg = sorted(s for s in train if st.get(s) == {"reviewed_no_violation"})
    cnt = dict(train_clips=len(train), violation=len(pos), negative=len(neg), conflicting_or_unknown=len(train) - len(pos) - len(neg),
               violation_without_time=sum(1 for s in pos if not times.get(s)), skipped_controls=0)
    rnd = random.Random(f"{seed}|{cid}")
    seen, out = set(), []

    def add(sha, t, probe):
        k = (sha, round(t, 1))
        if k not in seen:
            seen.add(k)
            out.append({"sha": sha, "clip": idx[sha], "t": round(t, 1), "probe": probe})
            return True
        return False

    for s in pos:
        for t in sorted(times.get(s, ())):
            add(s, t, "label")

    def pick(sha, avoid):
        d = durations.get(idx.get(sha))
        if not d or d < 1.0:
            return None
        for _ in range(300):
            t = round(rnd.uniform(0.5, max(0.5, d - 0.5)), 1)
            if all(abs(t - a) >= MIN_AWAY for a in avoid) and (sha, t) not in seen:
                return t
        return None

    for s in sorted(train & set(idx)):
        t = pick(s, times.get(s, ()))                      # (a) same clip, >=20 s from any label
        if t is None or not add(s, t, "control"):
            cnt["skipped_controls"] += 1
        pool = [x for x in neg if x != s] or neg           # (b) random time in a random train negative clip
        t2, o = None, None
        if pool:
            o = rnd.choice(pool)
            t2 = pick(o, times.get(o, ()))
        if t2 is None or not add(o, t2, "control"):
            cnt["skipped_controls"] += 1
    rnd.shuffle(out)                                       # presentation order; tags stay in the file only
    cnt.update(prompts=len(out), label=sum(p["probe"] == "label" for p in out),
               control=sum(p["probe"] == "control" for p in out))
    return out, cnt


def load_prompts(private, cid):
    p, _a = _paths(private, cid)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def answers_by_key(private, cid):
    _p, a = _paths(private, cid)
    return {(r["sha"], round(r["t"], 1)): r for r in read_labels(a)
            if r.get("kind") == "attr" and r.get("mode") != "audit"}


def save_answer(private, cid, prompt, answers):
    if set(answers) != set(VALID) or any(answers[q] not in VALID[q] for q in VALID):
        raise ValueError("answers")
    _p, a = _paths(private, cid)
    rec = {"kind": "attr", "clip": prompt["clip"], "sha": prompt["sha"], "t": prompt["t"], "probe": prompt["probe"],
           "answers": answers, "ts": round(time.time(), 3)}
    with _lock:
        keep = [r for r in read_labels(a) if not (r.get("kind") == "attr" and r.get("mode") != "audit"
                                                  and r["sha"] == rec["sha"] and round(r["t"], 1) == rec["t"])]
        _atomic_write(a, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep + [rec]))
    return rec


def summary(private, cid):
    pr = load_prompts(private, cid) or []
    ans = answers_by_key(private, cid)
    out = {"n_prompts": len(pr), "n_answered": 0, "by_probe": {}}
    for probe in ("label", "control"):
        out["by_probe"][probe] = {"n_prompts": 0, "n_answered": 0,
                                  **{q: {k: 0 for k, _ in o} for q, _t, o in QUESTIONS}}
    for p in pr:
        b = out["by_probe"][p["probe"]]
        b["n_prompts"] += 1
        r = ans.get((p["sha"], round(p["t"], 1)))
        if r:
            b["n_answered"] += 1
            out["n_answered"] += 1
            for q in VALID:
                b[q][r["answers"][q]] += 1
    return out


def make_router(private, clips_fn):
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse
    r = APIRouter()

    @r.get("/api/attrs/questions")
    def questions():
        return [{"id": q, "text": t, "options": [{"key": k, "label": lb} for k, lb in o]} for q, t, o in QUESTIONS]

    @r.post("/api/c/{cid}/attrs/generate")
    async def generate(cid: str, request: Request):
        cl = clips_fn(cid)
        if cl is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        try:
            body = await request.json()
        except ValueError:
            body = {}
        pf, _a = _paths(private, cid)
        if os.path.exists(pf) and not body.get("force"):
            return JSONResponse({"detail": "exists"}, status_code=409)
        try:
            sp = json.load(open(os.path.join(private, "splits", cid + ".json"), encoding="utf-8"))["clips"]
            recs = read_labels(os.path.join(private, "labels", cid + ".imported.jsonl"))
            train = {s for s, v in sp.items() if v == "train"}
            need = sorted({x["clip"] for x in recs if x.get("sha") in train})
            from .inventory import _probe_one          # isolated child; TRAIN clips only
            dur = {}
            for i in need:
                v = _probe_one(cl[i]) if 0 <= i < len(cl) else None
                dur[i] = v[1] / v[0] if v and v[0] > 0 and v[1] > 0 else None
            prompts, cnt = generate_prompts(private, cid, dur, str(body.get("seed", "forte-attrs")))
        except (LookupError, OSError, ValueError, KeyError):
            return JSONResponse({"detail": "split/imported labels required"}, status_code=409)
        _atomic_write(pf, json.dumps(prompts))
        print("attrs " + cid + ": " + " ".join(f"{k}={v}" for k, v in cnt.items()), flush=True)
        return cnt

    @r.get("/api/c/{cid}/attrs/prompts")
    def prompts(cid: str):
        pr = load_prompts(private, cid)
        if pr is None:
            return JSONResponse({"detail": "not generated"}, status_code=404)
        ans = answers_by_key(private, cid)
        items = []
        for pid, p in enumerate(pr):                       # NO probe tag, no sha
            a = ans.get((p["sha"], round(p["t"], 1)))
            items.append({"pid": pid, "clip": p["clip"], "t": p["t"], "answers": a["answers"] if a else None})
        return {"items": items, "total": len(items), "answered": sum(i["answers"] is not None for i in items)}

    @r.put("/api/c/{cid}/attrs/answer/{pid}")
    async def answer(cid: str, pid: int, request: Request):
        pr = load_prompts(private, cid)
        if pr is None or not 0 <= pid < len(pr):
            return JSONResponse({"detail": "not found"}, status_code=404)
        try:
            save_answer(private, cid, pr[pid], (await request.json())["answers"])
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"detail": "invalid"}, status_code=422)
        s = summary(private, cid)
        print(f"attrs {cid}: answered={s['n_answered']}/{s['n_prompts']}", flush=True)
        return {"answered": s["n_answered"], "total": s["n_prompts"]}

    @r.get("/api/c/{cid}/attrs/summary")
    def summ(cid: str):
        if load_prompts(private, cid) is None:
            return JSONResponse({"detail": "not generated"}, status_code=404)
        return summary(private, cid)

    return r


HTML = """
<div class="card" id="attrs"><b>Атрибуты момента</b> <span id="a_prog" class="mut"></span>
<span class="mut"> · кнопки 1–4 выбирают вариант текущего вопроса</span>
<div><button id="a_load">Загрузить задания</button> <button id="a_prev">Назад</button> <button id="a_next">Далее</button></div>
<video id="av" controls style="margin-top:8px"></video>
<div id="a_q"></div></div>
<script>
(()=>{const g=id=>document.getElementById(id);
const J=async(u,o)=>{const r=await fetch(u,o);if(!r.ok)throw new Error(r.status);return r.json()};
let Q=[],items=[],cur=0,ans={},cid="";
const cc=()=>g("cid").value;
async function load(){cid=cc();try{Q=await J("/api/attrs/questions");const d=await J(`/api/c/${cid}/attrs/prompts`);items=d.items;
 const f=items.findIndex(x=>!x.answers);cur=f<0?0:f;show()}catch(e){g("a_prog").textContent="задания не созданы"}}
function prog(){const a=items.filter(x=>x.answers).length;g("a_prog").textContent=`отвечено ${a}/${items.length}`}
function show(){if(!items.length)return;const it=items[cur];ans=Object.assign({},it.answers||{});prog();
 const v=g("av"),src=`/api/c/${cid}/video/${it.clip}`;
 const seek=()=>{v.currentTime=it.t};
 if(v.dataset.src!==src){v.dataset.src=src;v.src=src;v.onloadedmetadata=seek}else seek();
 g("a_q").innerHTML=Q.map((q,qi)=>`<div style="margin:6px 0"><b>${qi+1}) ${q.text}</b><br>`+q.options.map((o,oi)=>
  `<button data-q="${q.id}" data-k="${o.key}" style="margin:2px;${ans[q.id]===o.key?'':'background:var(--card);color:var(--fg);border:1px solid var(--mut)'}">${oi+1}. ${o.label}</button>`).join("")+`</div>`).join("")}
async function pick(q,k){ans[q]=k;items[cur].answers=Object.keys(ans).length===Q.length?Object.assign({},ans):null;
 if(Object.keys(ans).length===Q.length){await J(`/api/c/${cid}/attrs/answer/${items[cur].pid}`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({answers:ans})});
  const n=items.findIndex((x,i)=>i>cur&&!x.answers);if(n>=0)cur=n;else if(cur<items.length-1)cur++}
 show()}
g("a_q").onclick=e=>{const t=e.target;if(t.dataset.q)pick(t.dataset.q,t.dataset.k)};
g("a_load").onclick=load;
g("a_next").onclick=()=>{if(cur<items.length-1){cur++;show()}};g("a_prev").onclick=()=>{if(cur>0){cur--;show()}};
document.addEventListener("keydown",e=>{if(!items.length||e.target.tagName==="INPUT"||e.target.tagName==="SELECT")return;
 const n=+e.key;if(!(n>=1&&n<=9))return;const q=Q.find(q=>!ans[q.id]);if(q&&q.options[n-1])pick(q.id,q.options[n-1].key)});
})();
</script>
"""
