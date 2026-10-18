"""API-level audit selftest on an isolated synthetic fixture (called from tools/_selftest_ui.py).
Fixture: 5 label moments of which only 1 has a same-clip control; 2 pre-existing answers; holdout injections."""
import contextlib
import io
import json
import os

from . import annotate, annotate_audit as AU
from .make_synthetic import make

S = "ZZSENTINEL"


def run(t):
    from fastapi.testclient import TestClient
    cdir = os.path.join(t, "aud_" + S)
    os.makedirs(cdir)
    for i in range(8):
        make(os.path.join(cdir, f"{S}_v{i}.mp4"), dur=50 + i)
    cl = annotate.list_clips(cdir)
    sh = [annotate.sha_prefix(p) for p in cl]
    priv = os.path.join(t, "priv_aud")
    for d in ("labels", "splits", "attrs_prompts"):
        os.makedirs(os.path.join(priv, d))
    json.dump({"C01": cdir}, open(os.path.join(priv, "inventory_map.json"), "w"))
    json.dump({"clips": {s: ("holdout" if i >= 6 else "train") for i, s in enumerate(sh)}},
              open(os.path.join(priv, "splits", "C01.json"), "w"))
    recs = [{"kind": "imported_clip_status", "clip": i, "sha": s,
             "status": "reviewed_violation" if i < 5 else "reviewed_no_violation"} for i, s in enumerate(sh)]
    open(os.path.join(priv, "labels", "C01.imported.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in recs))

    def P(i, tt, probe):
        return {"sha": sh[i], "clip": i, "t": tt, "probe": probe}
    prompts = [P(i, 8.0, "label") for i in range(5)] + [P(0, 40.0, "control"), P(1, 12.0, "control"),
              P(2, 11.0, "control"), P(5, 20.0, "control"), P(5, 30.0, "control"), P(5, 40.0, "control"),
              P(6, 15.0, "label"), P(7, 15.0, "control")]            # last two: holdout, must never be served
    pf = os.path.join(priv, "attrs_prompts", "C01.json")
    json.dump(prompts, open(pf, "w"))
    ans = {"q1": "1", "q3": "yes", "q7": "no"}
    pre = [{"kind": "attr", "mode": "audit", "clip": 0, "sha": sh[0], "t": tt, "probe": pb, "answers": ans,
            "t1": "start", "v1": "visible", "ts": 1.0} for tt, pb in ((8.0, "label"), (40.0, "control"))]
    ap = os.path.join(priv, "labels", "C01.attrs.jsonl")
    open(ap, "w").write("".join(json.dumps(r) + "\n" for r in pre))
    pre_lines = open(ap).read().splitlines()
    for i in (6, 7):
        os.chmod(cl[i], 0)
    texts, out, err = [], io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            def app():
                return TestClient(annotate.create_app(priv, extra_allowed_clients=("testclient",)),
                                  base_url="http://127.0.0.1")
            cli = app()

            def get(u):
                r = cli.get(u)
                texts.append(r.text)
                return r
            d = get("/api/c/C01/audit/moments").json()
            st = d["strategy"]
            assert d["total"] == 10 and d["saved"] == 2 and d["remaining"] == 8 and d["answered"] == 2, d
            assert st["a"] == 1 and st["b"] + st["c"] == 4 and st["none"] == 0 and st["kept_answered"] == 0, st
            assert d["skipped_holdout"] == 2
            assert all(set(i) == {"pid", "clip", "t", "answered", "paired", "answers", "t1", "v1"} for i in d["items"])
            assert all(i["clip"] < 6 for i in d["items"]) and sum(i["answered"] for i in d["items"]) == 2
            ids = {i["pid"]: i["answered"] for i in d["items"]}
            order = [i["pid"] for i in d["items"]]
            # reorder + reload: same ids, same order, answers preserved
            dup = [dict(p, pid=7) if i < 4 else dict(p) for i, p in enumerate(prompts)]       # repeated pid
            json.dump(list(reversed(prompts)), open(pf, "w"))
            d2 = app().get("/api/c/C01/audit/moments").json()
            texts.append(json.dumps(d2))
            assert [i["pid"] for i in d2["items"]] == order and {i["pid"]: i["answered"] for i in d2["items"]} == ids
            assert d2["strategy"] == st and len(ids) == 10 and d2["total"] == 10
            json.dump(list(reversed(dup)), open(pf, "w"))
            d4 = app().get("/api/c/C01/audit/moments").json()
            assert d4["total"] == 10 and len({i["pid"] for i in d4["items"]}) == 10 and d4["saved"] == 2
            stale = dup + [dict(P(1, 30.0, "control"), clip=3)]                # sha of clip1 with a wrong index
            json.dump(stale, open(pf, "w"))
            d3 = app().get("/api/c/C01/audit/moments").json()
            assert d3["skipped_stale"] == 1 and d3["total"] == 10
            assert all(isinstance(i["paired"], bool) for i in d3["items"]) and sum(i["paired"] for i in d3["items"]) == 2
            json.dump(prompts, open(pf, "w"))
            # new answer appended, old records byte-identical, re-answer replaces only that record
            new = next(p for p, a in ids.items() if not a)
            r = cli.put(f"/api/c/C01/audit/answer/{new}", json={"answers": ans, "t1": "end", "v1": "partial"})
            assert r.status_code == 200 and r.json() == {"total": 10, "saved": 3, "remaining": 7, "answered": 3}
            lines = open(ap).read().splitlines()
            assert lines[:2] == pre_lines and len(lines) == 3
            assert cli.put(f"/api/c/C01/audit/answer/{new}", json={"answers": dict(ans, q1="0"), "t1": "end", "v1": "visible"}).status_code == 200
            lines = open(ap).read().splitlines()
            assert lines[:2] == pre_lines and len(lines) == 3 and json.loads(lines[2])["answers"]["q1"] == "0"
            assert cli.put("/api/c/C01/audit/answer/nope", json={"answers": ans, "t1": "end", "v1": "visible"}).status_code == 404
            hid = "m" + AU._h(sh[6], 15.0)                                 # holdout id is never served/accepted
            assert cli.put(f"/api/c/C01/audit/answer/{hid}", json={"answers": ans, "t1": "end", "v1": "visible"}).status_code == 404
            assert len(open(ap).read().splitlines()) == 3
            # R1: no cache yet -> generate (train only), cache, save, reload
            rf = os.path.join(priv, "attrs_prompts", "C01.audit_r1.json")
            assert not os.path.exists(rf)
            r1 = get("/api/c/C01/audit/r1").json()
            assert r1["total"] == 6 and r1["saved"] == 0 and r1["remaining"] == 6 and os.path.exists(rf), r1
            assert all(x["clip"] < 6 and x["t"] > 10 for x in r1["items"])
            cache = json.load(open(rf))
            json.dump(cache + [{"sha": sh[6], "clip": 6, "t": 10.0}], open(rf, "w"))   # holdout injected into cache
            r1b = get("/api/c/C01/audit/r1").json()
            assert r1b["total"] == 6 and r1b["skipped_holdout"] == 1 and [x["pid"] for x in r1b["items"]] == [x["pid"] for x in r1["items"]]
            assert cli.put(f"/api/c/C01/audit/r1/{'r' + AU._h('r1', sh[6])}", json={"r1": "top"}).status_code == 404
            pid0 = r1["items"][0]["pid"]
            assert cli.put(f"/api/c/C01/audit/r1/{pid0}", json={"r1": "zzz"}).status_code == 422
            assert cli.put(f"/api/c/C01/audit/r1/{pid0}", json={"r1": "top"}).json() == {"total": 6, "saved": 1, "remaining": 5}
            r1c = app().get("/api/c/C01/audit/r1").json()
            assert r1c["saved"] == 1 and r1c["items"][0]["r1"] == "top"
            assert os.path.exists(os.path.join(priv, "labels", "C01.audit_r1.jsonl"))
            # holdout injected into the prompt cache + summary progress
            json.dump(prompts + [P(6, 20.0, "control")], open(pf, "w"))
            sm = get("/api/c/C01/attrs/audit_summary").json()
            assert sm["progress"] == {"total": 10, "saved": 3, "remaining": 7} and sm["skipped_holdout"] >= 3
            assert sm["r1"]["total"] == 6 and sm["r1"]["saved"] == 1 and sm["saved_not_in_list"] == 0
            pg = sm["pairing"]
            assert pg["paired_labels"] == 1 and pg["paired_controls"] == 1 and pg["unpaired_labels"] == 4
            assert pg["unpaired_controls"] == 4 and pg["answered_paired"] == 2 and pg["answered_unpaired"] == 1
            assert sum(1 for r_ in map(json.loads, open(ap).read().splitlines()) if r_.get("pair_kind")) == 1
            # R1 incomplete cache (3 of 6) -> missing regenerated, total from expected set
            json.dump(json.load(open(rf))[:3], open(rf, "w"))
            r1d = get("/api/c/C01/audit/r1").json()
            assert r1d["total"] == 6 and len(r1d["items"]) == 6 and r1d["regenerated"] == 3 and len(json.load(open(rf))) == 6
            # stale index (train sha, index of a holdout clip) -> dropped + counted, regenerated with the right index
            c2 = json.load(open(rf))
            c2[0]["clip"] = 6
            json.dump(c2, open(rf, "w"))
            r1e = get("/api/c/C01/audit/r1").json()
            assert r1e["stale_index"] == 1 and r1e["regenerated"] == 1 and r1e["total"] == 6
            assert all(x["clip"] < 6 for x in r1e["items"]) and r1e["unprobed"] == 0
            # CLI prepares from scratch and prints counts only
            os.remove(rf)
            assert AU.main(["C01", "--private", priv]) == 0
            # changed split: clip 5 -> holdout, clip 6 -> train
            os.chmod(cl[6], 0o644)
            os.chmod(cl[5], 0)
            json.dump({"clips": {s_: ("holdout" if i in (5, 7) else "train") for i, s_ in enumerate(sh)}},
                      open(os.path.join(priv, "splits", "C01.json"), "w"))
            r1f = get("/api/c/C01/audit/r1").json()
            assert r1f["total"] == 6 and r1f["skipped_holdout"] == 1 and r1f["regenerated"] == 1 and r1f["unprobed"] == 0
            assert sorted(x["clip"] for x in r1f["items"]) == [0, 1, 2, 3, 4, 6] and r1f["saved"] <= 1
            # missing prompts file: no traceback, exit 2, R1 still generated
            os.remove(pf)
            os.remove(rf)
            o2 = io.StringIO()
            with contextlib.redirect_stdout(o2):
                assert AU.main(["C01", "--private", priv]) == 2
            assert "prompts file missing" in o2.getvalue() and "r1 total=6" in o2.getvalue() and os.path.exists(rf)
            assert S not in o2.getvalue() and t not in o2.getvalue()
        o = out.getvalue()
        assert "moments total=10 saved=3 remaining=7" in o and "strategy a=1" in o and "r1 total=6" in o
        blob = o + err.getvalue() + "".join(texts)
        assert S not in blob and t not in blob and '"sha"' not in "".join(texts) and '"probe"' not in "".join(texts)
        assert not any(s in blob for s in sh)
    finally:
        for i in range(8):
            os.chmod(cl[i], 0o644)
    return 1


if __name__ == "__main__":
    import shutil
    import tempfile
    _t = tempfile.mkdtemp(prefix="forte_audit_")
    try:
        print(f"audit selftest ok: {run(_t)}/1")
    finally:
        shutil.rmtree(_t, ignore_errors=True)
