"""Synthetic self-test for tools/annotate_intervals.py (TestClient; fixture only; prints counts only).
  python -m tools._selftest_intervals [SCRATCH_DIR]"""
import io
import json
import math
import os
import shutil
import sys
import contextlib

from . import annotate_intervals as AI
from .annotate import sha_prefix
from .make_synthetic import make

SENT = "SENTINELCLIP"
SAME = {"Sec-Fetch-Site": "same-origin"}
ST = [0, 0]


def check(name, cond):
    ST[0 if cond else 1] += 1
    if not cond:
        print("FAIL", name, file=sys.stderr)


def build(root, cid="C02"):
    """4 train clips (2 positive, 2 negative) + 1 holdout (chmod 000). Returns (private, holdout_sha, status dict)."""
    shutil.rmtree(root, ignore_errors=True)
    d, priv = os.path.join(root, "clips"), os.path.join(root, "private")
    for x in ("labels", "splits"):
        os.makedirs(os.path.join(priv, x))
    os.makedirs(d)
    names = [f"{SENT}_{i}.mp4" for i in range(5)]
    for i, n in enumerate(names):
        make(os.path.join(d, n), 3 + i * 0.5, size=(160, 90))
    shas = [sha_prefix(os.path.join(d, n)) for n in names]
    # clip index = sorted file order = names order
    status = {0: "reviewed_violation", 1: "reviewed_violation", 2: "reviewed_no_violation",
              3: "reviewed_no_violation", 4: "reviewed_violation"}
    with open(os.path.join(priv, "labels", f"{cid}.imported.jsonl"), "w") as f:
        for i, s in enumerate(shas):
            f.write(json.dumps({"kind": "imported_clip_status", "sha16": s, "clip": i, "status": status[i]}) + "\n")
    json.dump({"clips": {s: ("holdout" if i == 4 else "train") for i, s in enumerate(shas)}},
              open(os.path.join(priv, "splits", f"{cid}.json"), "w"))
    json.dump({cid: d}, open(os.path.join(priv, "inventory_map.json"), "w"))
    os.chmod(os.path.join(d, names[4]), 0)
    return priv, shas[4]


def client(priv, cid="C02", hdr=SAME):
    from fastapi.testclient import TestClient
    return TestClient(AI.create_app(priv, cid), base_url="http://127.0.0.1:8791", headers=hdr)


def PL(**kw):
    p = {"intervals": [], "clip_flags": {"not_visible": False, "unknown": False, "event_unconfirmed": False},
         "same_clip_controls": [], "control_not_found": False, "done": False}
    p.update(kw)
    return p


def piv(s, e, **kw):
    d = {"start": s, "end": e, "visibility": "good", "confidence": "sure"}
    d.update(kw)
    return d


def civ(s, e, a=True, b=True):
    return {"start": s, "end": e, "visibility": "partial", "confidence": "unsure", "comparable_visible": a, "no_violation": b}


def put(c, t, rev, pl, **kw):
    return c.put(f"/api/i/C02/task/{t}", json={"rev": rev, "payload": pl}, **kw)


def tasks(c):
    j = c.get("/api/i/C02/tasks").json()
    return j["tasks"], j["counters"]


def main(argv=None):
    root = (argv or sys.argv[1:] or ["/private/tmp/claude-501/scratch_stage8"])[0]
    priv, hosha = build(os.path.join(root, "a"))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        c = client(priv)
        tl, cn = tasks(c)
        P = [t["task_id"] for t in tl]
        check("2 positives only at start", len(tl) == 2 and all(t["role"] == "positive" for t in tl))
        check("holdout not a task", AI.task_id(hosha, "positive", 0) not in P and hosha not in json.dumps(tl))
        check("initial counters", cn["positives"] == {"total": 2, "shown": 0, "completed_with_intervals": 0, "completed_unknown": 0}
              and cn["controls"]["generated"] == 0 and cn["controls"]["max"] == 2)
        t0, t1 = P
        g = c.get(f"/api/i/C02/task/{t0}").json()
        dur = g["duration"]
        check("duration probed", 2.5 < dur < 4 and g["rev"] == 0 and g["payload"] is None)
        # save + reload/resume + multiple + draft
        r = put(c, t0, 0, PL(intervals=[piv(0.5, 1.0, start_uncertain=True)]))
        check("save rev1", r.status_code == 200 and r.json()["rev"] == 1)
        r = put(c, t0, 1, PL(intervals=[piv(0.5, 1.0, start_uncertain=True), piv(1.5, None)]))
        check("draft pending start saved", r.status_code == 200 and r.json()["rev"] == 2)
        c2 = client(priv)
        g = c2.get(f"/api/i/C02/task/{t0}").json()
        check("reload resumes pending start", g["rev"] == 2 and g["payload"]["intervals"][1] == {**piv(1.5, None), "start_unknown": False,
              "end_unknown": False, "start_uncertain": False, "end_uncertain": False} and g["payload"]["intervals"][0]["start_uncertain"])
        r = put(c2, t0, 2, PL(intervals=[piv(0.5, 1.0, start_uncertain=True), piv(1.5, 2.0)]))
        g = c2.get(f"/api/i/C02/task/{t0}").json()
        check("completing draft bumps rev, keeps earlier", r.json()["rev"] == 3 and len(g["payload"]["intervals"]) == 2
              and g["payload"]["intervals"][0]["start"] == 0.5)
        # null boundary + flag; invalid times
        r = put(c2, t0, 3, PL(intervals=[piv(None, 1.0, start_unknown=True)]))
        check("null+flag accepted", r.status_code == 200)
        rev = 4
        for name, iv in (("num with flag", piv(0.5, 1.0, start_unknown=True)), ("end<=start", piv(2.0, 1.0)),
                         ("negative", piv(-1, 1)), ("beyond duration", piv(0.1, dur + 5)), ("string", piv("a", 1)),
                         ("bool", piv(True, 2)), ("both unknown", piv(None, None, start_unknown=True, end_unknown=True)),
                         ("bad enum", piv(0, 1, visibility="x")), ("extra field", {**piv(0, 1), "note": "t"})):
            r = put(c2, t0, rev, PL(intervals=[iv]))
            check("422 " + name, r.status_code == 422)
        r = c2.put(f"/api/i/C02/task/{t0}", content='{"rev":4,"payload":{"intervals":[{"start":NaN,"end":1,"visibility":"good","confidence":"sure"}],'
                   '"clip_flags":{},"same_clip_controls":[],"control_not_found":false,"done":false}}', headers={"Content-Type": "application/json"})
        check("422 NaN", r.status_code == 422)
        r = put(c2, t0, rev, PL(intervals=[piv(0.5, None)], done=True))
        check("422 done with unfinished interval", r.status_code == 422)
        r = put(c2, t0, rev, PL(done=True))
        check("422 done with nothing", r.status_code == 422)
        check("invalid never stored", c2.get(f"/api/i/C02/task/{t0}").json()["rev"] == 4)
        # stale rev
        r = put(c2, t0, 1, PL(intervals=[piv(0.1, 0.2)]))
        g = c2.get(f"/api/i/C02/task/{t0}").json()
        check("409 stale + no overwrite", r.status_code == 409 and r.json()["current_rev"] == 4 and g["rev"] == 4
              and g["payload"]["intervals"][0]["start"] is None)
        # completion: intervals vs unknown; counters
        r = put(c2, t0, 4, PL(intervals=[piv(0.5, 1.0)], done=True))
        check("complete with intervals", r.status_code == 200)
        r = put(c2, t1, 0, PL(clip_flags={"not_visible": False, "unknown": True, "event_unconfirmed": False}, done=True))
        _, cn = tasks(c2)
        check("unknown counted separately", cn["positives"]["completed_with_intervals"] == 1 and cn["positives"]["completed_unknown"] == 1)
        # viewed / media_error semantics
        check("shown 0 before player", cn["positives"]["shown"] == 0)
        r = put(c2, t1, 1, {**PL(clip_flags={"not_visible": False, "unknown": True, "event_unconfirmed": False}, done=True), "viewed": True})
        check("client cannot set viewed", c2.get(f"/api/i/C02/task/{t1}").json()["payload"].get("viewed") is False)
        c2.post(f"/api/i/C02/task/{t1}/media_error", json={"rev": c2.get(f"/api/i/C02/task/{t1}").json()["rev"]})
        _, cn = tasks(c2)
        check("media_error not shown", cn["positives"]["shown"] == 0)
        cur0 = c2.get(f"/api/i/C02/task/{t0}").json()["rev"]
        check("viewed stale rev -> 409, nothing adopted", c2.post(f"/api/i/C02/task/{t0}/viewed", json={"rev": cur0 - 1}).status_code == 409
              and c2.post(f"/api/i/C02/task/{t0}/viewed", json={}).status_code == 422)
        v = c2.post(f"/api/i/C02/task/{t0}/viewed", json={"rev": cur0})
        _, cn = tasks(c2)
        check("loadeddata -> shown", v.status_code == 200 and cn["positives"]["shown"] == 1)
        c2.post(f"/api/i/C02/task/{t0}/viewed", json={"rev": cur0 + 1})
        g = c2.get(f"/api/i/C02/task/{t0}").json()
        check("viewed idempotent, answers kept", g["payload"]["done"] and g["payload"]["intervals"][0]["end"] == 1.0)
        # controls
        r = c2.post(f"/api/i/C02/task/{t0}/next_control")
        check("422 control before not_found", r.status_code == 422)
        g = c2.get(f"/api/i/C02/task/{t0}").json()
        pl = dict(g["payload"]); pl.update(control_not_found=True, same_clip_controls=[civ(2.0, 2.5)])
        for k in ("viewed", "media_error"):
            pl.pop(k, None)
        r = put(c2, t0, g["rev"], pl)
        check("same-clip control saved", r.status_code == 200)
        r = put(c2, t0, g["rev"] + 1, {**pl, "same_clip_controls": [civ(2.0, 2.5, True, False)]})
        r = c2.post(f"/api/i/C02/task/{t0}/next_control")
        c1 = r.json()["task_id"]
        check("control generated", r.status_code == 200 and len(c1) == 16)
        r2 = c2.post(f"/api/i/C02/task/{t0}/next_control")
        check("second blocked until first done", r2.status_code == 422)
        gc = c2.get(f"/api/i/C02/task/{c1}").json()
        check("control readable + parent", gc["role"] == "control" and gc["payload"]["parent"] == t0 and gc["duration"])
        check("control PUT needs both confirmations", put(c2, c1, gc["rev"], {"intervals": [civ(0.2, 0.8, True, False)], "answer": None, "done": True}).status_code == 422)
        r = put(c2, c1, gc["rev"], {"intervals": [], "answer": "not_comparable", "done": True})
        check("not_comparable completes", r.status_code == 200)
        _, cn = tasks(c2)
        check("control counters unclear", cn["controls"]["generated"] == 1 and cn["controls"]["not_comparable_or_unknown"] == 1
              and cn["controls"]["confirmed_normal_confident"] == 0 and cn["controls"]["total"] == 2)
        r = c2.post(f"/api/i/C02/task/{t0}/next_control")
        c2id = r.json()["task_id"]
        check("second candidate after unclear", r.status_code == 200 and c2id != c1)
        gc2 = c2.get(f"/api/i/C02/task/{c2id}").json()
        r = put(c2, c2id, gc2["rev"], {"intervals": [civ(0.2, 0.8)], "answer": None, "done": True})
        _, cn = tasks(c2)
        check("confirmed normal counted", r.status_code == 200 and cn["controls"]["confirmed_normal_low_confidence"] == 1
              and cn["controls"]["confirmed_normal_confident"] == 0 and cn["controls"]["generated"] == 2)
        good = {"intervals": [{**civ(0.2, 0.8), "visibility": "good", "confidence": "sure"}], "answer": None, "done": True}
        check("classify confident vs low", AI.classify("control", good) == "normal_confident"
              and AI.classify("control", {**good, "intervals": [civ(0.2, 0.8)]}) == "normal_low")
        tl2 = c2.get("/api/i/C02/tasks").json()["tasks"]
        ctl = [t for t in tl2 if t["role"] == "control"]
        check("control listing has parent ordinal + candidate", sorted((t["ord"], t["cand"], t["of"]) for t in ctl) == [(1, 1, 2), (1, 2, 2)]
              or sorted((t["ord"], t["cand"], t["of"]) for t in ctl) == [(2, 1, 2), (2, 2, 2)])
        check("limit 2 per positive", c2.post(f"/api/i/C02/task/{t0}/next_control").status_code == 422)
        check("no candidates for 2nd positive", c2.post(f"/api/i/C02/task/{t1}/next_control").status_code == 422)
        # Range / allowlist / host / origin
        full = c2.get(f"/api/i/C02/video/{t0}")
        size = len(full.content)
        check("full 200", full.status_code == 200 and size > 100)
        rg = c2.get(f"/api/i/C02/video/{t0}", headers={"Range": "bytes=0-99"})
        check("range 206", rg.status_code == 206 and len(rg.content) == 100 and rg.headers["content-range"] == f"bytes 0-99/{size}")
        check("range 416", c2.get(f"/api/i/C02/video/{t0}", headers={"Range": f"bytes={size + 5}-"}).status_code == 416)
        check("control video streams", c2.get(f"/api/i/C02/video/{c1}", headers={"Range": "bytes=0-9"}).status_code == 206)
        bad = [AI.task_id(hosha, "positive", 0), AI.task_id(hosha, "control", 0), "0" * 16, "../etc/passwd", "C02", "_assign"]
        for b in bad:
            for u in (f"/api/i/C02/video/{b}", f"/api/i/C02/task/{b}"):
                check("404 " + u[-8:], c2.get(u).status_code in (404,))
        check("404 other cid", c2.get(f"/api/i/C99/task/{t0}").status_code == 404)
        check("404 listing/dirs", all(c2.get(u).status_code == 404 for u in ("/api/i/C02/video/", "/api/i/C02/", "/clips", "/api/c/C02/clips", "/api/candidates")))
        check("403 foreign host", c2.get("/api/i/C02/tasks", headers={"Host": "evil.example"}).status_code == 403)
        nh = client(priv, hdr={})
        check("403 write without origin", put(nh, t0, 99, PL()).status_code == 403)
        check("403 cross-site write", put(nh, t0, 99, PL(), headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403)
        check("403 foreign origin", put(nh, t0, 99, PL(), headers={"Origin": "http://evil.example"}).status_code == 403)
        check("origin same ok (409 not 403)", put(nh, t0, 99, PL(), headers={"Origin": "http://127.0.0.1:8791"}).status_code == 409)
        check("no model hints in page", all(w not in c2.get("/").text.lower() for w in ("score", "peak", "predict", "http://", "https://")))
        # deterministic order in an independent fixture
        priv_b, _ = build(os.path.join(root, "b"))
        cb = client(priv_b)
        tb = [t["task_id"] for t in tasks(cb)[0]]
        check("task ids reproducible", tb == P)
        pb = cb.get(f"/api/i/C02/task/{tb[0]}").json()
        put(cb, tb[0], 0, PL(control_not_found=True))
        rb = cb.post(f"/api/i/C02/task/{tb[0]}/next_control").json()["task_id"]
        check("control order reproducible", rb == c1)
        lg = AI.read_log(priv, "C02")
        asg = lg["_assign"]["payload"]
        flat = [s for v in asg.values() for s in v]
        check("assignment fixed + unique, <=2 each", len(flat) == len(set(flat)) == 2 and all(len(v) <= 2 for v in asg.values()))
        c3 = client(priv)
        check("assignment survives restart", AI.read_log(priv, "C02")["_assign"]["payload"] == asg and len(tasks(c3)[0]) == 4)
        # append-only log: revisions strictly increase per task
        revs = {}
        ok = True
        for line in open(AI.log_path(priv, "C02")):
            r_ = json.loads(line)
            ok &= r_["rev"] == revs.get(r_["task_id"], 0) + 1
            revs[r_["task_id"]] = r_["rev"]
        check("log append-only rev chain", ok)
        # (E) unreadable/unexpected failures map to static codes, no path leaks
        os.chmod(os.path.join(root, "a", "clips", f"{SENT}_1.mp4"), 0)
        os.chmod(os.path.join(root, "a", "clips", f"{SENT}_0.mp4"), 0)
        rr = [c3.get(f"/api/i/C02/video/{t}", headers={"Range": "bytes=0-9"}) for t in (t0, t1)]
        check("unreadable clip -> static 404", all(x.status_code == 404 and x.json() == {"detail": "unavailable"} for x in rr))
        orig = AI.read_log
        AI.read_log = lambda *a: (_ for _ in ()).throw(OSError(os.path.join(root, SENT)))
        try:
            x = c3.get("/api/i/C02/tasks")
        finally:
            AI.read_log = orig
        check("handler exception -> static 500", x.status_code == 500 and x.json() == {"detail": "error"} and SENT not in x.text)
        # sentinel
        blob = out.getvalue() + c3.get("/api/i/C02/tasks").text + c3.get("/").text + c3.get(f"/api/i/C02/task/{t0}").text
        check("sentinel absent from output/responses", SENT not in blob and root not in blob)
    shutil.rmtree(root, ignore_errors=True)
    try:
        browser(os.path.join(root, "w"))
    except ImportError:
        print("browser tests skipped: playwright missing")
    shutil.rmtree(root, ignore_errors=True)
    print(f"selftest_intervals: passed {ST[0]} failed {ST[1]}; stdout_sentinel={'LEAK' if SENT in out.getvalue() else 'clean'}")
    return 1 if ST[1] else 0


def _vp9(path, dur=6, fps=10):
    import cv2
    import numpy as np
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"VP90"), fps, (160, 90))
    for i in range(int(dur * fps)):
        f = np.full((90, 160, 3), 80, np.uint8)
        cv2.rectangle(f, (5 + i % 100, 30), (35 + i % 100, 60), (200, 200, 200), -1)
        w.write(f)
    w.release()


def browser(root):
    """Headless Chromium against a real uvicorn on loopback (synthetic VP9-in-mp4 clips; nothing saved)."""
    import socket
    import threading
    import time
    import uvicorn
    from playwright.sync_api import sync_playwright
    priv, _ = build(root)
    d = os.path.join(root, "clips")
    for i in range(5):  # replace with browser-decodable clips of the same names; rebuild shas/labels
        os.chmod(os.path.join(d, f"{SENT}_{i}.mp4"), 0o600)
        _vp9(os.path.join(d, f"{SENT}_{i}.mp4"), 5 + i)
    shas = [sha_prefix(os.path.join(d, f"{SENT}_{i}.mp4")) for i in range(5)]
    stt = ["reviewed_violation"] * 2 + ["reviewed_no_violation"] * 2 + ["reviewed_violation"]
    with open(os.path.join(priv, "labels", "C02.imported.jsonl"), "w") as f:
        for i, s_ in enumerate(shas):
            f.write(json.dumps({"kind": "imported_clip_status", "sha16": s_, "clip": i, "status": stt[i]}) + "\n")
    json.dump({"clips": {s_: ("holdout" if i == 4 else "train") for i, s_ in enumerate(shas)}},
              open(os.path.join(priv, "splits", "C02.json"), "w"))
    os.chmod(os.path.join(d, f"{SENT}_4.mp4"), 0)
    sk = socket.socket(); sk.bind(("127.0.0.1", 0)); port = sk.getsockname()[1]; sk.close()
    srv = uvicorn.Server(uvicorn.Config(AI.create_app(priv, "C02"), host="127.0.0.1", port=port, log_level="critical"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    url = f"http://127.0.0.1:{port}/"
    n0 = ST[0]
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        ctx = br.new_context()
        pg = ctx.new_page()
        posts = []
        pg.on("request", lambda r: posts.append(r.url.rsplit("/", 1)[-1]) if r.method == "POST" else None)
        def wst(sub, page=None):
            try:
                (page or pg).wait_for_function("t=>document.getElementById('st').textContent.includes(t)", arg=sub, timeout=8000)
            except Exception:  # noqa: BLE001
                check("status wait: " + sub + " | " + (page or pg).inner_text("#st"), False)
        seek = lambda x: pg.evaluate("x=>new Promise(r=>{const v=document.getElementById('v');v.onseeked=()=>r(v.currentTime);v.currentTime=x})", x)
        ct = lambda: pg.evaluate("document.getElementById('v').currentTime")
        pg.goto(url)
        pg.wait_for_selector("#list button")
        pg.locator("#list button").nth(1).click()   # positive #2: no control candidates
        pg.wait_for_function("window.__S&&window.__S.pl.viewed===true", timeout=8000)
        check("viewed after loadeddata", posts.count("viewed") == 1)
        pg.get_by_text("Добавить интервал", exact=True).click()
        pg.evaluate("document.getElementById('v').__mark=1")
        seek(2.0); seek(3.5)
        pg.get_by_text("Начало = текущее").click()
        wst("Сохранено rev")
        check("start taken from currentTime, player not reset", abs(ct() - 3.5) < 0.2 and pg.evaluate("document.getElementById('v').__mark===1")
              and abs(float(pg.input_value("input[data-f=start]")) - 3.5) < 0.2)
        seek(4.5)
        pg.get_by_text("Конец = текущее").click()
        wst("Сохранено rev")
        check("end from currentTime, player persistent", abs(ct() - 4.5) < 0.2 and pg.evaluate("document.getElementById('v').__mark===1")
              and abs(float(pg.input_value("input[data-f=end]")) - 4.5) < 0.2)
        pg.get_by_text("Добавить интервал", exact=True).click(); pg.get_by_text("Добавить интервал", exact=True).click()
        pg.locator("input[data-f=start_unknown]").nth(1).check()
        check("viewed posted once despite re-renders", posts.count("viewed") == 1)
        # pending start survives reload
        pg.locator("input[data-f=start]").nth(2).fill("1.5")
        pg.locator("[data-a=del]").nth(1).click()   # drop the all-empty second draft
        wst("Сохранено rev")
        rv = pg.evaluate("window.__S.rev")
        pg.reload(); pg.wait_for_selector("#list button"); pg.locator("#list button").nth(1).click()
        pg.wait_for_selector("input[data-f=start]")
        vals = pg.eval_on_selector_all("input[data-f=start]", "e=>e.map(x=>x.value)")
        ends = pg.eval_on_selector_all("input[data-f=end]", "e=>e.map(x=>x.value)")
        check("reload shows pending start", "1.5" in vals and "" in ends and len(vals) == 2 and pg.evaluate("window.__S.rev") == rv)
        pg.wait_for_timeout(500)
        check("no extra viewed after reload", posts.count("viewed") == 1)
        seek(3.0)
        pg.locator("[data-a=ce]").nth(1).click()
        wst("Сохранено rev")
        check("completing pending start bumps rev, keeps first", pg.evaluate("window.__S.rev") == rv + 1
              and pg.evaluate("window.__S.pl.intervals.length") == 2)
        # (6) unknown while intervals remain
        pg.locator("input[data-c=unknown]").check()
        wst("Удалите интервалы или снимите unknown")
        check("unknown+intervals message, draft kept", pg.evaluate("window.__S.pl.clip_flags.unknown") is True
              and ctx.request.get(url + f"api/i/C02/task/{pg.evaluate('window.__S.id')}").json()["rev"] == rv + 1)
        pg.locator("#list button").nth(0).click(); pg.wait_for_timeout(300)
        check("task switch blocked while unsaved error", pg.evaluate("window.__S.pl.clip_flags.unknown") is True)
        pg.locator("input[data-c=unknown]").uncheck()
        wst("Сохранено rev")
        # (2) stale rev from another tab
        tid = pg.evaluate("window.__S.id")
        g = ctx.request.get(url + f"api/i/C02/task/{tid}").json()
        r = ctx.request.put(url + f"api/i/C02/task/{tid}", data=json.dumps({"rev": g["rev"], "payload": g["payload"]}),
                            headers={"Content-Type": "application/json", "Sec-Fetch-Site": "same-origin"})
        check("other tab write ok", r.status == 200)
        pg.get_by_text("Добавить интервал", exact=True).click()
        wst("Конфликт")
        check("409 shown, draft+backup kept locally, no overwrite", ctx.request.get(url + f"api/i/C02/task/{tid}").json()["rev"] == g["rev"] + 1
              and pg.evaluate("localStorage.getItem('ivd:C02:'+window.__S.id)") is not None
              and pg.evaluate("localStorage.getItem('ivb:C02:'+window.__S.id)") is not None and pg.is_visible("#cf_s") and pg.is_visible("#cf_m"))
        nint = pg.evaluate("window.__S.pl.intervals.length")
        pg.click("#cf_m"); pg.wait_for_timeout(200)
        check("restore needs confirmation (no write yet)", ctx.request.get(url + f"api/i/C02/task/{tid}").json()["rev"] == g["rev"] + 1)
        pg.click("#cf_m"); wst("Сохранено rev")
        g2 = ctx.request.get(url + f"api/i/C02/task/{tid}").json()
        check("restore-over bumps rev with my draft", g2["rev"] == g["rev"] + 2 and len(g2["payload"]["intervals"]) == nint)
        # second conflict -> load server version
        ctx.request.put(url + f"api/i/C02/task/{tid}", data=json.dumps({"rev": g2["rev"], "payload": {**g2["payload"], "intervals": g2["payload"]["intervals"][:1]}}),
                        headers={"Content-Type": "application/json", "Sec-Fetch-Site": "same-origin"})
        pg.get_by_text("Добавить интервал", exact=True).click(); wst("Конфликт")
        pg.click("#cf_s"); wst("Загружена серверная версия")
        check("load server version, backup retained", pg.evaluate("window.__S.pl.intervals.length") == 1
              and pg.evaluate("localStorage.getItem('ivb:C02:'+window.__S.id)") is not None and pg.evaluate("window.__S.rev") == g2["rev"] + 1)
        # (D) typed value survives reload before blur / debounce
        pg.locator("[data-a=del]").first.click(); wst("Сохранено rev")
        pg.get_by_text("Добавить интервал", exact=True).click(); wst("Сохранено rev")
        pg.locator("input[data-f=start]").first.fill("2.25")
        pg.reload(); pg.wait_for_selector("#list button"); pg.locator("#list button").nth(1).click()
        pg.wait_for_selector("input[data-f=start]"); pg.wait_for_timeout(600)
        check("input event saved before blur", pg.input_value("input[data-f=start]") == "2.25")
        # (3) control not found on positive #1 awaited save then next_control
        pg2 = ctx.new_page(); pg2.goto(url); pg2.wait_for_selector("#list button")
        pg2.locator("#list button").nth(0).click(); pg2.wait_for_function("window.__S&&window.__S.pl.viewed===true", timeout=8000)
        pg2.get_by_text("Контроль в этом клипе не найден").click()
        pg2.wait_for_function("document.getElementById('nfs').textContent.includes('добавлен')", timeout=8000)
        lst = ctx.request.get(url + "api/i/C02/tasks").json()
        check("not_found saved then control generated", sum(t["role"] == "control" for t in lst["tasks"]) == 1
              and lst["counters"]["controls"]["generated"] == 1)
        # control header + back button
        pg2.locator("#list button", has_text="Контроль к ролику").first.click()
        pg2.wait_for_function("window.__S.role==='control'", timeout=8000)
        hd = pg2.inner_text("#hd")
        check("control header: parent ordinal + candidate", "Контроль к ролику 1, кандидат 1/2" in hd)
        pg2.get_by_text("Вернуться к ролику 1").click()
        pg2.wait_for_function("window.__S.role==='positive'", timeout=8000)
        check("back returns to parent, intervals intact", pg2.evaluate("window.__S.rev") >= 2 and pg2.evaluate("window.__S.pl.control_not_found") is True)
        # offline PUT -> local draft recovered on reopen
        pg2.route("**/api/i/C02/task/*", lambda route: route.abort() if route.request.method == "PUT" else route.continue_())
        pg2.get_by_text("Добавить интервал", exact=True).click()
        wst("Не сохранено — черновик сохранён локально", pg2)
        tid2 = pg2.evaluate("window.__S.id")
        pg2.unroute("**/api/i/C02/task/*")
        pg2.reload(); pg2.wait_for_selector("#list button"); pg2.locator("#list button").nth(0).click()
        pg2.wait_for_function("window.__S&&window.__S.restored===true", timeout=8000)
        wst("Сохранено rev", pg2)
        check("local draft restored and saved", ctx.request.get(url + f"api/i/C02/task/{tid2}").json()["payload"]["intervals"] != [])
        br.close()
    srv.should_exit = True
    print(f"browser checks: {ST[0] - n0} passed", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
