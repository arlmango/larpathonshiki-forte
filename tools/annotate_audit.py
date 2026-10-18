"""Blind audit (mounted by tools/annotate.py): label moments + controls x (q1,q3,q7) + T1/V1, and R1 per train clip.
Tags (label|control) and shas never leave the server; stdout/responses = counts only.

Moments: EVERY valid train label prompt is kept. Control strategy per label moment (labels in canonical (sha,t)
order, fixed seed only shuffles presentation order; never invents time points beyond attrs_prompts/<cid>.json):
  (a) first unused same-clip control >= 20 s from every label of that clip (an already-answered same-clip control
      is reused first, so existing answers are never orphaned),
  (b) else first unused control from ANOTHER train clip with the same imported status class,
  (c) else first unused control from a train-negative clip.
No total is forced; a label with no control gets none (counted as strategy "none"). Previously answered controls
that no label selected are kept ("kept_answered"). Moment id = hash of (prompt pid if any, sha, t, probe), made
unique deterministically if inputs collide -- never positional, so reordering/duplicating pids changes nothing.
Pairing: only branch (a) gives a same-clip PAIR (pair_kind same_clip, paired true). Branches (b)/(c) are
cross-clip (pair_kind cross_clip_status / cross_clip_negative, paired false): they cannot support the originally
promised paired sign test and are usable only as unpaired background. Items expose a bool `paired` for labels and
controls alike (a label is paired iff its control is same-clip), stored records carry pair_kind, and
audit_summary reports paired vs unpaired counts. Prompt sha<->clip index is validated against the current
imported-label file before use; mismatching entries are skipped (skipped_stale).
Merge: audit records keyed by (sha, t) are never rewritten or dropped; new answers are appended; re-answer
replaces only that record. Train-only: every prompt/R1 entry (also from caches) is checked against the split;
non-train entries are skipped and counted (skipped_holdout).
Records: attr {"mode":"audit", pid, answers:{q1,q3,q7}, t1, v1, probe} in labels/<cid>.attrs.jsonl;
R1 in labels/<cid>.audit_r1.jsonl (cache attrs_prompts/<cid>.audit_r1.json; neutral time 25 % of duration).
CLI (counts only): python -m tools.annotate_audit CID [--private DIR]  (prepares moments + R1 cache)."""
import hashlib
import json
import os
import random
import time

from . import annotate_attrs as AT
from .annotate import _atomic_write, _lock, read_labels
from .annotate_audit_ui import HTML  # noqa: F401  (mounted by annotate.py)

AUDIT_Q = ("q1", "q3", "q7")
T1 = [("start", "начало события"), ("middle", "середина"), ("end", "конец"), ("not_event", "не событие"),
      ("unseen", "не видно")]
R1 = [("bottom", "нижней части кадра"), ("top", "верхней части кадра"), ("left", "левой части кадра"),
      ("right", "правой части кадра"), ("unseen", "не видно")]
V1 = [("visible", "видно"), ("partial", "частично"), ("unseen", "не видно"), ("cannot_say", "не могу сказать")]
V1_KEYS = {k for k, _ in V1}
T1_KEYS, R1_KEYS = {k for k, _ in T1}, {k for k, _ in R1}
MIN_AWAY = AT.MIN_AWAY


def _h(*a):
    return hashlib.sha1("|".join(map(str, a)).encode()).hexdigest()[:10]


def _side(private, cid):
    """{sha: side}; LookupError if the split is missing/unreadable."""
    try:
        return json.load(open(os.path.join(private, "splits", cid + ".json"), encoding="utf-8"))["clips"]
    except (OSError, ValueError, KeyError, TypeError):
        raise LookupError("split") from None


def _status_class(private, cid):
    st = {}
    for r in read_labels(os.path.join(private, "labels", cid + ".imported.jsonl")) \
            if os.path.exists(os.path.join(private, "labels", cid + ".imported.jsonl")) else []:
        if r.get("kind") == "imported_clip_status":
            st.setdefault(r.get("sha"), set()).add(r.get("status"))
    return {s: ("violation" if v == {"reviewed_violation"} else "negative" if v == {"reviewed_no_violation"} else None)
            for s, v in st.items()}


def _train_units(private, cid, side):
    """{sha: clip index} for TRAIN shas, from the current imported-label file (authoritative)."""
    ip = os.path.join(private, "labels", cid + ".imported.jsonl")
    if not os.path.exists(ip):
        raise LookupError("imported")
    first = {}
    for r in read_labels(ip):
        if side.get(r.get("sha")) == "train" and isinstance(r.get("clip"), int) and not isinstance(r.get("clip"), bool):
            first.setdefault(r["sha"], r["clip"])
    return first


def _audit_path(private, cid):
    return AT._paths(private, cid)[1]


def audit_records(private, cid):
    return {(r["sha"], round(r["t"], 1)): r for r in read_labels(_audit_path(private, cid))
            if r.get("kind") == "attr" and r.get("mode") == "audit"}


def _prompt_ok(p):
    return (isinstance(p, dict) and isinstance(p.get("sha"), str) and isinstance(p.get("t"), (int, float))
            and not isinstance(p.get("t"), bool) and p.get("probe") in ("label", "control")
            and isinstance(p.get("clip"), int))


def moments(private, cid, seed="forte-audit", info=None):
    """List of moments {id,sha,clip,t,probe,strategy}; None if no prompts file. LookupError if no split.
    `info` (dict) receives counts: strategy{a,b,c,none,kept_answered}, skipped_holdout, skipped_invalid."""
    pr = AT.load_prompts(private, cid)
    if pr is None:
        return None
    side = _side(private, cid)
    cls = _status_class(private, cid)
    ans = audit_records(private, cid)
    units = _train_units(private, cid, side)
    inf = {"strategy": {"a": 0, "b": 0, "c": 0, "none": 0, "kept_answered": 0}, "skipped_holdout": 0,
           "skipped_invalid": 0, "skipped_stale": 0}
    seen, labels, controls, ids = set(), [], [], set()
    for p in sorted((p for p in pr if _prompt_ok(p)), key=lambda p: (p["sha"], p["t"], p["probe"] != "label")):
        if side.get(p["sha"]) != "train":
            inf["skipped_holdout"] += 1
            continue
        if units.get(p["sha"]) != p["clip"]:             # stale sha<->index: never serve / open
            inf["skipped_stale"] += 1
            continue
        k = (p["sha"], round(p["t"], 1))
        if k in seen:
            continue
        seen.add(k)
        mid, n = "m" + _h(p.get("pid", ""), *k, p["probe"]), 0
        while mid in ids:
            n += 1
            mid = "m" + _h(p.get("pid", ""), *k, p["probe"], n)
        ids.add(mid)
        m = {"id": mid, "sha": p["sha"], "clip": p["clip"], "t": round(p["t"], 1), "probe": p["probe"],
             "strategy": None, "pair_kind": "none"}
        (labels if p["probe"] == "label" else controls).append(m)
    inf["skipped_invalid"] = sum(1 for p in pr if not _prompt_ok(p))
    key = lambda m: (m["sha"], m["t"])  # noqa: E731
    used, picked = set(), []

    KIND = {"a": "same_clip", "b": "cross_clip_status", "c": "cross_clip_negative", "kept_answered": "unmatched"}

    def take(c, tag, L=None):
        used.add(key(c))
        c["strategy"] = tag
        c["pair_kind"] = KIND[tag]
        if L is not None:
            L["pair_kind"] = KIND[tag]
        picked.append(c)
        inf["strategy"][tag] += 1

    pending = []
    for L in labels:                          # pass 1: reuse an already-answered same-clip control
        c = next((c for c in controls if c["sha"] == L["sha"] and key(c) not in used and key(c) in ans), None)
        take(c, "a", L) if c else pending.append(L)
    for L in pending:
        lts = [x["t"] for x in labels if x["sha"] == L["sha"]]
        free = [c for c in controls if key(c) not in used]
        c = next((c for c in free if c["sha"] == L["sha"] and all(abs(c["t"] - t) >= MIN_AWAY for t in lts)), None)
        tag = "a"
        if c is None and cls.get(L["sha"]):
            c = next((c for c in free if c["sha"] != L["sha"] and cls.get(c["sha"]) == cls[L["sha"]]), None)
            tag = "b"
        if c is None:
            c = next((c for c in free if c["sha"] != L["sha"] and cls.get(c["sha"]) == "negative"), None)
            tag = "c"
        if c is None:
            inf["strategy"]["none"] += 1
        else:
            take(c, tag, L)
    for c in controls:                        # answered controls no label selected stay visible
        if key(c) not in used and key(c) in ans:
            take(c, "kept_answered")
    out = sorted(labels + picked, key=lambda m: m["id"])
    for m in out:
        m["paired"] = m["pair_kind"] == "same_clip"
    random.Random(f"{seed}|{cid}").shuffle(out)
    if info is not None:
        info.update(inf)
    return out


def save_audit(private, cid, m, answers, t1, v1):
    if (not isinstance(answers, dict) or set(answers) != set(AUDIT_Q)
            or any(answers[q] not in AT.VALID[q] for q in AUDIT_Q) or t1 not in T1_KEYS or v1 not in V1_KEYS):
        raise ValueError("answers")
    rec = {"kind": "attr", "mode": "audit", "pid": m["id"], "clip": m["clip"], "sha": m["sha"], "t": m["t"],
           "probe": m["probe"], "pair_kind": m.get("pair_kind", "none"), "answers": answers, "t1": t1, "v1": v1, "ts": round(time.time(), 3)}
    path = _audit_path(private, cid)
    with _lock:
        keep = [r for r in read_labels(path) if not (r.get("kind") == "attr" and r.get("mode") == "audit"
                                                     and r["sha"] == rec["sha"] and round(r["t"], 1) == rec["t"])]
        _atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep + [rec]))


def _progress(total, saved):
    return {"total": total, "saved": saved, "remaining": total - saved}


def r1_list(private, cid, clips_fn, info=None):
    """Expected set = TRAIN shas (split) -> clip index (current imported labels). Every call: cached entries are
    kept only if sha is still train AND the stored index equals the expected one (else dropped, counted); missing
    entries are probed (train files only) and merged into the cache. Totals come from the expected set.
    LookupError if no split/imports."""
    side = _side(private, cid)
    expected = _train_units(private, cid, side)
    f = os.path.join(private, "attrs_prompts", cid + ".audit_r1.json")
    inf = {"expected": len(expected), "skipped_holdout": 0, "stale_index": 0, "unprobed": 0, "regenerated": 0}
    try:
        raw = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else []
    except (OSError, ValueError):
        raw = []
    have, dirty = {}, not os.path.exists(f)
    for x in raw if isinstance(raw, list) else []:
        if not (isinstance(x, dict) and isinstance(x.get("sha"), str) and isinstance(x.get("t"), (int, float))
                and isinstance(x.get("clip"), int)):
            dirty = True
            continue
        if side.get(x["sha"]) != "train":
            inf["skipped_holdout"] += 1
            dirty = True
        elif expected.get(x["sha"]) != x["clip"] or x["sha"] in have:
            inf["stale_index"] += 1
            dirty = True
        else:
            have[x["sha"]] = x
    missing = sorted(s for s in expected if s not in have)
    if missing:
        cl = clips_fn(cid)
        if cl is None:
            raise LookupError("clips")
        from .inventory import _probe_one
        for sha in missing:
            i = expected[sha]
            v = _probe_one(cl[i]) if 0 <= i < len(cl) else None
            if v and v[0] > 0 and v[1] > 0:
                have[sha] = {"sha": sha, "clip": i, "t": round(0.25 * v[1] / v[0], 1)}
                inf["regenerated"] += 1
                dirty = True
            else:
                inf["unprobed"] += 1
    items = [have[s] for s in sorted(have)]
    if dirty:
        _atomic_write(f, json.dumps(items))
    for x in items:
        x["id"] = "r" + _h("r1", x["sha"])
    if info is not None:
        info.update(inf)
    return items


def r1_answers(private, cid):
    p = os.path.join(private, "labels", cid + ".audit_r1.jsonl")
    return {r["sha"]: r for r in read_labels(p) if r.get("kind") == "audit_r1"}


def r1_progress(private, cid, items, total=None):
    ra = r1_answers(private, cid)
    return _progress(total if total is not None else len(items), sum(i["sha"] in ra for i in items)), ra


def summary(private, cid, r1items, ms=None, info=None, r1total=None):
    ms = ms if ms is not None else (moments(private, cid) or [])
    ans = audit_records(private, cid)
    out = {"moments": len(ms), "answered": 0, "by_probe": {}, "t1_stored": 0, "t1_label": {k: 0 for k, _ in T1},
           "v1_stored": 0, "v1_label": {k: 0 for k, _ in V1}}
    for pb in ("label", "control"):
        out["by_probe"][pb] = {"n": 0, "answered": 0, **{q: {k: 0 for k, _ in dict((a, b) for a, _t, b in AT.QUESTIONS)[q]}
                                                          for q in AUDIT_Q}}
    for m in ms:
        b = out["by_probe"][m["probe"]]
        b["n"] += 1
        r = ans.get((m["sha"], m["t"]))
        if r:
            b["answered"] += 1
            out["answered"] += 1
            out["t1_stored"] += 1
            out["v1_stored"] += 1
            for q in AUDIT_Q:
                b[q][r["answers"][q]] += 1
            if m["probe"] == "label":
                out["t1_label"][r["t1"]] += 1
                out["v1_label"][r["v1"]] += 1
    out["progress"] = _progress(len(ms), out["answered"])
    out["saved_not_in_list"] = len(ans) - out["answered"]
    out.update({k: v for k, v in (info or {}).items()})
    pr, ra = r1_progress(private, cid, r1items, r1total)
    pg = out["pairing"] = {"paired_labels": 0, "unpaired_labels": 0, "paired_controls": 0, "unpaired_controls": 0,
                           "answered_paired": 0, "answered_unpaired": 0,
                           "pair_kinds": {k: 0 for k in ("same_clip", "cross_clip_status", "cross_clip_negative",
                                                         "unmatched", "none")}}
    for m in ms:
        pg[("paired_" if m["paired"] else "unpaired_") + ("labels" if m["probe"] == "label" else "controls")] += 1
        pg["pair_kinds"][m["pair_kind"]] += 1 if m["probe"] == "control" or m["pair_kind"] == "none" else 0
        if (m["sha"], m["t"]) in ans:
            pg["answered_paired" if m["paired"] else "answered_unpaired"] += 1
    out["r1"] = {"n_clips": len(r1items), "answered": pr["saved"], **pr,
                 "counts": {k: sum(1 for i in r1items if ra.get(i["sha"], {}).get("r1") == k) for k, _ in R1}}
    return out


def make_router(private, clips_fn):
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse
    r = APIRouter()
    NOSPLIT = JSONResponse({"detail": "split/imported labels required"}, status_code=409)

    def moments_info(cid):
        inf = {}
        try:
            return moments(private, cid, info=inf), inf
        except LookupError:
            return "nosplit", inf

    @r.get("/api/audit/questions")
    def questions():
        qs = {q: (t, o) for q, t, o in AT.QUESTIONS}
        return {"q": [{"id": q, "text": qs[q][0], "options": [{"key": k, "label": lb} for k, lb in qs[q][1]]}
                      for q in AUDIT_Q],
                "v1": {"text": "Нарушение видно в кадре", "options": [{"key": k, "label": lb} for k, lb in V1]},
                "t1": {"text": "Оцените момент", "options": [{"key": k, "label": lb} for k, lb in T1]},
                "r1": {"text": "Кассир находится в", "options": [{"key": k, "label": lb} for k, lb in R1]}}

    @r.get("/api/c/{cid}/audit/moments")
    def get_moments(cid: str):
        ms, inf = moments_info(cid)
        if ms == "nosplit":
            return NOSPLIT
        if ms is None:
            return JSONResponse({"detail": "not generated"}, status_code=404)
        ans = audit_records(private, cid)
        items = []
        for m in ms:                                     # no probe tag, no sha
            a = ans.get((m["sha"], m["t"]))
            items.append({"pid": m["id"], "clip": m["clip"], "t": m["t"], "answered": a is not None, "paired": m["paired"],
                          "answers": a["answers"] if a else None, "t1": a["t1"] if a else None,
                          "v1": a["v1"] if a else None})
        n = sum(i["answered"] for i in items)
        return {"items": items, **_progress(len(items), n), "answered": n, "strategy": inf["strategy"],
                "skipped_holdout": inf["skipped_holdout"], "skipped_stale": inf["skipped_stale"]}

    @r.put("/api/c/{cid}/audit/answer/{pid}")
    async def put_answer(cid: str, pid: str, request: Request):
        ms, _inf = moments_info(cid)
        m = next((x for x in ms if x["id"] == pid), None) if isinstance(ms, list) else None
        if m is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        try:
            b = await request.json()
            save_audit(private, cid, m, b["answers"], b["t1"], b["v1"])
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"detail": "invalid"}, status_code=422)
        ans = audit_records(private, cid)
        n = sum((x["sha"], x["t"]) in ans for x in ms)
        print(f"audit {cid}: answered={n}/{len(ms)}", flush=True)
        return {**_progress(len(ms), n), "answered": n}

    def r1_get(cid):
        inf = {}
        try:
            return r1_list(private, cid, clips_fn, inf), inf
        except (LookupError, OSError, ValueError, KeyError, IndexError, TypeError):
            return None, inf

    @r.get("/api/c/{cid}/audit/r1")
    def get_r1(cid: str):
        items, inf = r1_get(cid)
        if items is None:
            return NOSPLIT
        pr, ra = r1_progress(private, cid, items, inf["expected"])
        return {"items": [{"pid": x["id"], "clip": x["clip"], "t": x["t"], "r1": ra.get(x["sha"], {}).get("r1")}
                          for x in items], **pr, "skipped_holdout": inf["skipped_holdout"],
                "unprobed": inf["unprobed"], "stale_index": inf["stale_index"], "regenerated": inf["regenerated"]}

    @r.put("/api/c/{cid}/audit/r1/{pid}")
    async def put_r1(cid: str, pid: str, request: Request):
        items, _inf = r1_get(cid)
        x = next((i for i in items or [] if i["id"] == pid), None)
        if x is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        try:
            v = (await request.json())["r1"]
            if v not in R1_KEYS:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"detail": "invalid"}, status_code=422)
        path = os.path.join(private, "labels", cid + ".audit_r1.jsonl")
        with _lock:
            keep = [q for q in read_labels(path) if q.get("sha") != x["sha"]]
            keep.append({"kind": "audit_r1", "mode": "audit", "clip": x["clip"], "sha": x["sha"], "t": x["t"],
                         "r1": v, "ts": round(time.time(), 3)})
            _atomic_write(path, "".join(json.dumps(q) + "\n" for q in keep))
        pr, _ = r1_progress(private, cid, items, _inf["expected"])
        print(f"audit {cid}: r1_answered={pr['saved']}/{pr['total']}", flush=True)
        return pr

    @r.get("/api/c/{cid}/attrs/audit_summary")
    def audit_summary(cid: str):
        ms, inf = moments_info(cid)
        if ms == "nosplit":
            return NOSPLIT
        if ms is None:
            return JSONResponse({"detail": "not generated"}, status_code=404)
        items, rinf = r1_get(cid)
        return summary(private, cid, items or [], ms, {"strategy": inf["strategy"],
                                                      "skipped_holdout": inf["skipped_holdout"] + rinf["skipped_holdout"]},
                       rinf.get("expected"))

    return r


def main(argv=None):
    """Prepare moments + R1 cache locally; prints COUNTS only (no names, shas, times)."""
    import argparse
    from . import annotate
    ap = argparse.ArgumentParser()
    ap.add_argument("cid")
    ap.add_argument("--private", default=annotate.PRIVATE)
    a = ap.parse_args(argv)

    def clips(cid):
        p = os.path.join(a.private, "inventory_map.json")
        m = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        return annotate.list_clips(m[cid]) if cid in m and os.path.isdir(m[cid]) else None

    inf, rinf = {}, {}
    try:
        ms = moments(a.private, a.cid, info=inf)
        items = r1_list(a.private, a.cid, clips, rinf)
    except LookupError:
        print("split/imported labels required")
        return 1
    if ms is None:
        print("moments: prompts file missing (run the attrs generation first)")
        pr, _ = r1_progress(a.private, a.cid, items, rinf["expected"])
    else:
        s = summary(a.private, a.cid, items, ms, inf, rinf["expected"])
        pr = s["r1"]
        print("moments " + " ".join(f"{k}={s['progress'][k]}" for k in ("total", "saved", "remaining")),
              "strategy " + " ".join(f"{k}={v}" for k, v in inf["strategy"].items()),
              f"skipped_holdout={inf['skipped_holdout']}")
    print("r1 " + " ".join(f"{k}={pr[k]}" for k in ("total", "saved", "remaining")),
          " ".join(f"{k}={v}" for k, v in rinf.items()))
    return 0 if ms is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
