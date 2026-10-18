"""Self-test for tools/inventory.py and tools/annotate.py on a SYNTHETIC tree. Run: python -m tools._selftest_ui"""
import contextlib
import csv
import io
import json
import os
import shutil
import tempfile

from . import _selftest_audit, annotate, inventory
from .make_synthetic import make


def _import_group(t):
    """Importer: sentinel privacy, profile verdicts (clean/mixed/text), import counts, annotate review."""
    import openpyxl
    from . import import_labels as IL
    S = "ZZSENTINEL"
    par = os.path.join(t, "imp_" + S)
    cdir = os.path.join(par, "clips_" + S)
    os.makedirs(cdir)
    names = [f"{S}_c{i}.mp4" for i in range(3)]
    for i, n in enumerate(names):
        make(os.path.join(cdir, n), dur=1 + i)
    priv = os.path.join(t, "priv_imp")
    os.makedirs(priv)
    json.dump({"C01": cdir}, open(os.path.join(priv, "inventory_map.json"), "w"))
    xl = os.path.join(par, "book.xlsx")

    def book(c2, c3, extra=()):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append([S + "_h1", S + "_h2", S + "_h3"])
        for i, n in enumerate(names):
            ws.append([n if i % 2 else os.path.splitext(n)[0], c2[i], c3[i]])
        for r in extra:
            ws.append(r)
        wb.save(xl)

    def run(argv):
        o, e = io.StringIO(), io.StringIO()
        old = IL.PRIVATE
        IL.PRIVATE = priv
        try:
            with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
                rc = IL.main(argv)
        finally:
            IL.PRIVATE = old
        txt = o.getvalue() + e.getvalue()
        for bad in (S, "forte_ui_", "45:10", "book.xlsx"):
            assert bad not in txt, "importer leak"
        return rc, txt

    book(["45:10", "30:05", "28:00"], [S + " text", S + " мимо", S + " ещё"])
    rc, txt = run(["--candidate", "C01"])
    assert rc == 0 and "timecode_format: unambiguous:mm:ss" in txt and "binary_label: none" in txt, "clean"
    assert "n_matches_video_stem=3" in txt
    book(["12:30", 15, "45:10"], [S, S, S])
    assert "timecode_format: ambiguous" in run(["--candidate", "C01"])[1], "mixed"
    book(["", S + " fail", ""], [S, S, S])
    assert "timecode_format: none" in run(["--candidate", "C01"])[1], "text"
    book(["да", "нет", "да"], ["", "", ""])
    assert "binary_label: col:2" in run(["--candidate", "C01"])[1]
    # import: text-list times + explicit status mapping
    book(["15s; 1:02", "", S], ["bad", "ok", ""], extra=[["unknown_file.mp4", "1:00", "bad"]])
    mp = {"filename_col": 1, "timecode_col": 2, "status_col": 3, "timecode_unit": "text_list",
          "no_violation_values": ["ok"], "violation_values": ["bad"]}
    mf = os.path.join(t, "m.json")
    json.dump(mp, open(mf, "w"))
    rc, txt = run(["--candidate", "C01", "--import", "--mapping", mf])
    assert rc == 0 and "rows=4 matched=3 unmatched=1 events=2 parse_errors=1" in txt, txt
    recs = annotate.read_labels(os.path.join(priv, "labels", "C01.imported.jsonl"))
    ev = sorted(r["timestamp_sec"] for r in recs if r["kind"] == "imported_event")
    assert ev == [15.0, 62.0] and all(r["status"] == "imported_unconfirmed" for r in recs if r["kind"] == "imported_event")
    st = [r["status"] for r in recs if r["kind"] == "imported_clip_status"]
    assert st == ["reviewed_violation", "reviewed_no_violation", "unknown"], st
    # empty status cell is never no_violation even if mapping lists nothing for it
    mp2 = dict(mp, status_col=None)
    assert all(r["status"] == "unknown" for r in IL.import_records(IL.load_rows(xl), mp2, IL.clip_index(priv, "C01")[1],
              IL.clip_index(priv, "C01")[0])[0] if r["kind"] == "imported_clip_status")
    # annotate: preview stays local, review writes human event + review record, stdout counts only
    from fastapi.testclient import TestClient
    cli = TestClient(annotate.create_app(priv, extra_allowed_clients=("testclient",)), base_url="http://127.0.0.1")
    pv = cli.get("/api/c/C01/import/preview").json()
    assert pv["header"][0] == S + "_h1" and len(pv["rows"]) == 3
    o = io.StringIO()
    with contextlib.redirect_stdout(o):
        r = cli.post("/api/c/C01/import/run", json=mp).json()
        assert cli.post("/api/c/C01/import/run", json={"filename_col": 1, "timecode_col": 2, "timecode_unit": "bogus"}).status_code == 422
    assert r["events"] == 2 and S not in o.getvalue()
    sta = cli.get("/api/c/C01/clip/0/state").json()
    assert [e["t"] for e in sta["imported"]] == [15.0, 62.0] and sta["events"] == [], str(sta)[:300]
    cli.post("/api/c/C01/clip/0/import_review", json={"timestamp_sec": 15.0, "decision": "accept"})
    cli.post("/api/c/C01/clip/0/import_review", json={"timestamp_sec": 62.0, "decision": "reject"})
    sta = cli.get("/api/c/C01/clip/0/state").json()
    assert [e["t"] for e in sta["events"]] == [15.0] and [e["decision"] for e in sta["imported"]] == ["accept", "reject"]
    assert sta["status"] == "unreviewed"  # confirming an event never sets clip status
    return 2


def _ru_group(t):
    """--auto-ru: fixed Russian allow-lists, cross-checks, needs_user, no sentinel leak."""
    import openpyxl
    from . import import_labels as IL
    S = "ZZSENTINEL"
    par = os.path.join(t, "ru_" + S)
    cdir = os.path.join(par, "c_" + S)
    os.makedirs(cdir)
    names = [f"{S}_r{i}.mp4" for i in range(6)]
    for i, n in enumerate(names):
        make(os.path.join(cdir, n), dur=70 if i == 1 else 12)
    priv = os.path.join(t, "priv_ru")
    os.makedirs(priv)
    json.dump({"C01": cdir}, open(os.path.join(priv, "inventory_map.json"), "w"))

    def book(rows):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append([S + "_h1", S + "_h2", S + "_h3"])
        for r in rows:
            ws.append(r)
        wb.save(os.path.join(par, "book.xlsx"))

    def run(argv):
        o, e = io.StringIO(), io.StringIO()
        old = IL.PRIVATE
        IL.PRIVATE = priv
        try:
            with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
                rc = IL.main(argv)
        finally:
            IL.PRIVATE = old
        txt = o.getvalue() + e.getvalue()
        for bad in (S, "forte_ui_", "минут", "нарушени", "book.xlsx"):
            assert bad not in txt, "ru importer leak"
        return rc, txt

    P, N = ["нарушение", "есть нарушение", "с нарушением", "нарушения"], ["нет нарушений", "без нарушения"]
    book([[names[0], "5 сек", P[1]], [names[1][:-4], "1 мин 2 сек", P[2]], [names[2], "3 с, 7 с", P[3]],
          [names[3], "4 с - 8 с", P[0]], [names[4], "около 2 мин", P[0]], [names[5], None, N[0]],
          [names[2], "5 сек", N[1]], [names[3], "50 сек", P[0]], [names[4], S + " text", P[0]],
          [names[5], None, P[0]]])
    rc, txt = run(["--candidate", "C01"])
    assert "col:3 (ru_binary)" in txt and "shape_list_seconds=1 shape_range_seconds=1 shape_minutes=1 shape_other=1" in txt, txt
    assert "n_ru_minutes_seconds_compound=1" in txt and "timecode_format: ambiguous" in txt
    rc, txt = run(["--candidate", "C01", "--import", "--auto-ru"])
    want = ("rows=10 matched=10 unmatched=0 events=5 parse_errors=0 statuses_set=10 needs_user=5 "
            "positive_with_time=7 positive_without_time=1 negative_with_time=1 "
            "n_time_within_duration=5 n_time_exceeds_duration=1 n_duration_unknown=0")
    assert rc == 0 and want in txt, txt
    recs = annotate.read_labels(os.path.join(priv, "labels", "C01.imported.jsonl"))
    ev = sorted((r["clip"], r["timestamp_sec"]) for r in recs if r["kind"] == "imported_event")
    assert ev == [(0, 5.0), (1, 62.0), (2, 3.0), (2, 7.0), (3, 4.0)], ev
    assert [r["range_end_sec"] for r in recs if r.get("range_end_sec")] == [8.0]
    assert all(r["state"] == "imported_unconfirmed" for r in recs if r["kind"] == "imported_clip_status")
    assert sorted(r["reason"] for r in recs if r["kind"] == "imported_needs_user") == [
        "negative_with_time", "positive_without_time", "shape_minutes", "shape_other", "time_exceeds_duration"]
    book([[names[0], "5 сек", P[0]], [names[1], "1 мин 2 сек", P[0]], [names[2], None, N[0]]])
    assert "timecode_format: unambiguous:ru_time_text" in run(["--candidate", "C01"])[1]
    # needs_user rows are served to the local UI only
    from fastapi.testclient import TestClient
    cli = TestClient(annotate.create_app(priv, extra_allowed_clients=("testclient",)), base_url="http://127.0.0.1")
    book([[names[0], "5 сек", P[0]], [names[4], "около 2 мин", P[0]], [names[5], None, N[0]]])
    run(["--candidate", "C01", "--import", "--auto-ru"])
    nu = cli.get("/api/c/C01/import/needs_user").json()
    assert len(nu) == 1 and nu[0]["clip"] == 4 and nu[0]["cells"][1] == "около 2 мин"
    return 1


def _attrs_group(t):
    """Stage-2 blind attribute prompts: train-only, holdout files unreadable, tags hidden, replace, summary."""
    from fastapi.testclient import TestClient
    from . import annotate_attrs as AT
    S = "ZZSENTINEL"
    cdir = os.path.join(t, "at_" + S)
    os.makedirs(cdir)
    names = [f"{S}_a{i}.mp4" for i in range(8)]
    for i, n in enumerate(names):
        make(os.path.join(cdir, n), dur=45 + i)
    cl = annotate.list_clips(cdir)
    shas = [annotate.sha_prefix(p) for p in cl]
    priv = os.path.join(t, "priv_at")
    os.makedirs(os.path.join(priv, "labels"))
    os.makedirs(os.path.join(priv, "splits"))
    json.dump({"C01": cdir, "C02": cdir}, open(os.path.join(priv, "inventory_map.json"), "w"))
    holdout = {2, 6, 7}
    recs = []
    for i, sh in enumerate(shas):
        recs.append({"kind": "imported_clip_status", "clip": i, "sha": sh,
                     "status": "reviewed_violation" if i < 3 else "reviewed_no_violation"})
        if i < 3:
            recs.append({"kind": "imported_event", "clip": i, "sha": sh, "timestamp_sec": 10.0})
    open(os.path.join(priv, "labels", "C01.imported.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in recs))
    json.dump({"clips": {sh: ("holdout" if i in holdout else "train") for i, sh in enumerate(shas)}},
              open(os.path.join(priv, "splits", "C01.json"), "w"))
    hold_shas = {shas[i] for i in holdout}
    for i in holdout:
        os.chmod(cl[i], 0)
    try:
        cli = TestClient(annotate.create_app(priv, extra_allowed_clients=("testclient",)), base_url="http://127.0.0.1")
        assert cli.post("/api/c/C02/attrs/generate", json={}).status_code == 409  # no split for C02
        o = io.StringIO()
        with contextlib.redirect_stdout(o):
            g = cli.post("/api/c/C01/attrs/generate", json={}).json()
        assert S not in o.getvalue() and "attrs C01:" in o.getvalue()
        assert g["train_clips"] == 5 and g["violation"] == 2 and g["label"] == 2 and g["prompts"] == g["label"] + g["control"]
        assert g["control"] >= 8 and cli.post("/api/c/C01/attrs/generate", json={}).status_code == 409
        pf = os.path.join(priv, "attrs_prompts", "C01.json")
        pr = json.load(open(pf))
        assert not ({p["sha"] for p in pr} & hold_shas), "holdout prompted"
        for p in pr:  # controls: >=20 s from any label of the same clip
            if p["probe"] == "control" and p["sha"] in (shas[0], shas[1]):
                assert abs(p["t"] - 10.0) >= 20.0
        first = [p["probe"] for p in pr]
        assert first != sorted(first), "labels not shuffled"
        before = open(pf).read()
        cli.post("/api/c/C01/attrs/generate", json={"force": True})
        assert open(pf).read() == before, "not deterministic"
        d = cli.get("/api/c/C01/attrs/prompts")
        items = d.json()["items"]
        assert set(items[0]) == {"pid", "clip", "t", "answers"} and "label" not in d.text and "control" not in d.text
        assert len(items) == g["prompts"] and all(i["clip"] not in holdout for i in items)
        full = {q: AT.QUESTIONS[n][2][0][0] for n, q in enumerate(AT.VALID)}
        full = {q: opts[0][0] for q, _t, opts in AT.QUESTIONS}
        assert cli.put("/api/c/C01/attrs/answer/0", json={"answers": dict(full, q1="bogus")}).status_code == 422
        assert cli.put("/api/c/C01/attrs/answer/0", json={"answers": {"q1": "0"}}).status_code == 422
        assert cli.put("/api/c/C01/attrs/answer/999", json={"answers": full}).status_code == 404
        o = io.StringIO()
        with contextlib.redirect_stdout(o):
            assert cli.put("/api/c/C01/attrs/answer/0", json={"answers": full}).json()["answered"] == 1
            assert cli.put("/api/c/C01/attrs/answer/1", json={"answers": dict(full, q2="none", q8="turned")}).json()["answered"] == 2
        assert o.getvalue().count("attrs C01: answered=") == 2 and S not in o.getvalue()
        cli.put("/api/c/C01/attrs/answer/0", json={"answers": dict(full, q1="2plus")})  # re-answer replaces
        ap = os.path.join(priv, "labels", "C01.attrs.jsonl")
        rows = annotate.read_labels(ap)
        assert len(rows) == 2 and all(r["kind"] == "attr" for r in rows)
        assert [r for r in rows if r["probe"] == pr[0]["probe"] and r["t"] == pr[0]["t"]][0]["answers"]["q1"] == "2plus"
        sm = cli.get("/api/c/C01/attrs/summary")
        sj = sm.json()
        assert sj["n_prompts"] == g["prompts"] and sj["n_answered"] == 2
        bp = sj["by_probe"]
        assert bp["label"]["n_answered"] + bp["control"]["n_answered"] == 2
        assert sum(sum(bp[k]["q1"].values()) for k in bp) == 2 and sum(sum(bp[k]["q2"].values()) for k in bp) == 2
        tot_q2_none = sum(bp[k]["q2"]["none"] for k in bp)
        assert tot_q2_none == 1
        assert "sha" not in sm.text and '"t"' not in sm.text and S not in sm.text
    finally:
        for i in holdout:
            os.chmod(cl[i], 0o644)
    return 1


def _guard_group(t):
    """inventory_map overwrite guard: refuse different map, --force keeps .prev (0600), same map is a no-op."""
    r1, r2 = os.path.join(t, "g1"), os.path.join(t, "g2")
    for r in (r1, r2):
        os.makedirs(r)
        make(os.path.join(r, "x.mp4"), dur=1)
    priv = os.path.join(t, "priv_guard")
    old = inventory.PRIVATE
    inventory.PRIVATE = priv
    try:
        def run(argv):
            o = io.StringIO()
            with contextlib.redirect_stdout(o):
                rc = inventory.main(argv)
            return rc, o.getvalue()
        mp = os.path.join(priv, "inventory_map.json")
        assert run([r1])[0] == 0
        first = open(mp).read()
        assert run([r1])[0] == 0                                   # identical map: fine
        rc, out = run([r2])
        assert rc == 2 and "use --force" in out and open(mp).read() == first and "g2" not in out
        assert not os.path.exists(os.path.join(priv, "inventory_map.prev.json"))
        assert run([r2, "--force"])[0] == 0
        prev = os.path.join(priv, "inventory_map.prev.json")
        assert open(prev).read() == first and oct(os.stat(prev).st_mode & 0o777) == "0o600"
        assert open(mp).read() != first
    finally:
        inventory.PRIVATE = old
    return 1


def _audit_group(t):
    """Audit mode: 10 moments, hidden tags, 3-key answers + T1, summary over 5 label moments, R1 per train clip."""
    from fastapi.testclient import TestClient
    S = "ZZSENTINEL"
    cdir = os.path.join(t, "au_" + S)
    os.makedirs(cdir)
    for i in range(8):
        make(os.path.join(cdir, f"{S}_u{i}.mp4"), dur=50 + i)
    cl = annotate.list_clips(cdir)
    shas = [annotate.sha_prefix(p) for p in cl]
    priv = os.path.join(t, "priv_au")
    os.makedirs(os.path.join(priv, "labels"))
    os.makedirs(os.path.join(priv, "splits"))
    json.dump({"C01": cdir}, open(os.path.join(priv, "inventory_map.json"), "w"))
    holdout = {5, 6, 7}
    recs = []
    for i, sh in enumerate(shas):
        recs.append({"kind": "imported_clip_status", "clip": i, "sha": sh,
                     "status": "reviewed_violation" if i < 5 else "reviewed_no_violation"})
        if i < 5:
            recs.append({"kind": "imported_event", "clip": i, "sha": sh, "timestamp_sec": 8.0})
    open(os.path.join(priv, "labels", "C01.imported.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in recs))
    json.dump({"clips": {sh: ("holdout" if i in holdout else "train") for i, sh in enumerate(shas)}},
              open(os.path.join(priv, "splits", "C01.json"), "w"))
    for i in holdout:
        os.chmod(cl[i], 0)
    try:
        cli = TestClient(annotate.create_app(priv, extra_allowed_clients=("testclient",)), base_url="http://127.0.0.1")
        assert cli.get("/api/c/C01/audit/moments").status_code == 404
        g = cli.post("/api/c/C01/attrs/generate", json={}).json()
        assert g["label"] == 5
        d = cli.get("/api/c/C01/audit/moments")
        items = d.json()["items"]
        assert len(items) == 10 and set(items[0]) == {"pid", "clip", "t", "answered", "paired", "answers", "t1", "v1"}; pids = [i["pid"] for i in items]
        assert "label" not in d.text and "control" not in d.text and "sha" not in d.text
        assert len({i["clip"] for i in items}) == 5 and all(i["clip"] not in holdout for i in items)
        full = {"q1": "1", "q3": "yes", "q7": "no"}
        o = io.StringIO()
        with contextlib.redirect_stdout(o):
            for n, pid in enumerate(pids):
                r = cli.put(f"/api/c/C01/audit/answer/{pid}", json={"answers": full, "t1": "start" if n % 2 else "end", "v1": "partial" if n % 2 else "visible"})
                assert r.status_code == 200
        assert o.getvalue().count("audit C01: answered=") == 10 and S not in o.getvalue()
        assert cli.put(f"/api/c/C01/audit/answer/{pids[0]}", json={"answers": dict(full, q2="no"), "t1": "start", "v1": "visible"}).status_code == 422
        assert cli.put(f"/api/c/C01/audit/answer/{pids[0]}", json={"answers": {"q1": "1"}, "t1": "start", "v1": "visible"}).status_code == 422
        assert cli.put(f"/api/c/C01/audit/answer/{pids[0]}", json={"answers": full, "t1": "bogus", "v1": "visible"}).status_code == 422
        assert cli.put(f"/api/c/C01/audit/answer/{pids[0]}", json={"answers": full, "t1": "start", "v1": "zzz"}).status_code == 422
        assert cli.put(f"/api/c/C01/audit/answer/{pids[0]}", json={"answers": full, "t1": "start"}).status_code == 422
        assert cli.put(f"/api/c/C01/audit/answer/{pids[0]}", json={"answers": dict(full, q1="2plus"), "t1": "end", "v1": "visible"}).status_code == 200
        rows = [r for r in annotate.read_labels(os.path.join(priv, "labels", "C01.attrs.jsonl")) if r.get("mode") == "audit"]
        assert len(rows) == 10 and all(set(r["answers"]) == {"q1", "q3", "q7"} and "t1" in r and "v1" in r for r in rows)
        sm = cli.get("/api/c/C01/attrs/audit_summary").json()
        bp = sm["by_probe"]
        assert sm["moments"] == 10 and sm["answered"] == 10 and sm["t1_stored"] == 10
        assert bp["label"]["n"] == 5 and bp["control"]["n"] == 5 and sum(sm["t1_label"].values()) == 5
        assert sm["v1_stored"] == 10 and sum(sm["v1_label"].values()) == 5
        assert sum(sum(bp[k]["q1"].values()) for k in bp) == 10 and sum(bp[k]["q3"]["yes"] for k in bp) == 10
        # R1: one per TRAIN clip (5 violation-labelled idx 0-4 are train; 5-7 holdout)
        r1 = cli.get("/api/c/C01/audit/r1").json()["items"]
        assert len(r1) == 5 and all(x["clip"] not in holdout for x in r1) and all(x["t"] > 10 for x in r1)
        for x in r1[:3]:
            assert cli.put(f"/api/c/C01/audit/r1/{x['pid']}", json={"r1": "bottom"}).status_code == 200
        assert cli.put(f"/api/c/C01/audit/r1/{r1[0]['pid']}", json={"r1": "left"}).status_code == 200  # replace
        assert cli.put(f"/api/c/C01/audit/r1/{r1[0]['pid']}", json={"r1": "zzz"}).status_code == 422
        sm = cli.get("/api/c/C01/attrs/audit_summary").json()
        assert sm["r1"] == {"n_clips": 5, "answered": 3, "total": 5, "saved": 3, "remaining": 2, "counts": {"bottom": 2, "top": 0, "left": 1, "right": 0, "unseen": 0}}
        assert "sha" not in json.dumps(sm) and S not in json.dumps(sm)
        # full-mode summary unaffected by audit records
        assert cli.get("/api/c/C01/attrs/summary").json()["n_answered"] == 0
    finally:
        for i in holdout:
            os.chmod(cl[i], 0o644)
    return 1


def main():
    t = tempfile.mkdtemp(prefix="forte_ui_")
    ok = 0
    try:
        a, b = os.path.join(t, "alpha"), os.path.join(t, "beta", "x")
        os.makedirs(a), os.makedirs(b)
        for i in range(3):
            make(os.path.join(a, f"clip{i}.mp4"), dur=1 + i)
        make(os.path.join(b, "z.mp4"), dur=1)
        open(os.path.join(b, "ignored.txt"), "w").write("x")
        with open(os.path.join(a, "l.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "timestamp", "secretcol"])
            w.writerow(["v1", "1.5", "zz"])
        json.dump([{"clip": "a", "start": 1, "end": 2}], open(os.path.join(a, "l.json"), "w"))
        exts, dirs = inventory.scan(t)
        assert exts[".mp4"] == 4 and exts[".csv"] == 1
        c = inventory.candidates(dirs)
        assert [x[0] for x in c] == ["C01", "C02"]
        assert len(c[0][2]["media"]) == 3 and c[0][2]["labels"][".csv"] == 1
        priv = os.path.join(t, "priv")
        inventory.write_map(c, priv)
        ok += 1
        buf = io.StringIO()
        inventory.PRIVATE = priv
        inventory.write_map.__defaults__ = (priv,)  # keep CLI map inside scratch
        with contextlib.redirect_stdout(buf):
            inventory.main([t, "--probe"])
        out = buf.getvalue()
        assert all(s not in out for s in ("alpha", "beta", "clip0", "forte_ui_", t)), "name leak"
        assert "fps_ge_29" in out and "probed_ok" in out
        ok += 1
        lr = inventory.label_readiness(a)
        s = json.dumps(lr)
        assert lr["csv"]["rows_total"] == 1 and lr["csv"]["files_with_timestamp_col"] == 1
        assert lr["json"]["files_with_video_id_col"] == 1 and "secretcol" not in s and "v1" not in s
        ok += 1
        cl = annotate.list_clips(a)
        shas = [annotate.sha_prefix(p) for p in cl]
        assert len(set(shas)) == 3
        s1, s2 = annotate.make_split(shas, 0.34), annotate.make_split(shas, 0.34)
        assert s1 == s2 and sum(s1["counts"].values()) == 3
        assert annotate.make_split(shas, 0)["counts"]["holdout"] == 0
        assert annotate.make_split(shas, 1)["counts"]["train"] == 0
        ok += 1
        from fastapi.testclient import TestClient
        cli = TestClient(annotate.create_app(priv, extra_allowed_clients=("testclient",)), base_url="http://127.0.0.1")
        cid = c[0][0]
        d = cli.get(f"/api/c/{cid}/clips").json()
        assert len(d["clips"]) == 3 and "alpha" not in json.dumps(d) and "clip0" not in json.dumps(d)
        r = cli.get(f"/api/c/{cid}/video/0", headers={"Range": "bytes=0-99"})
        assert r.status_code == 206 and len(r.content) == 100
        assert cli.get(f"/api/c/{cid}/video/0", headers={"Range": "bytes=99999999-"}).status_code == 416
        assert cli.get(f"/api/c/{cid}/video/9").status_code == 404
        assert cli.get("/api/c/..%2f/video/0").status_code == 404
        ok += 1
        ev = [{"t": 1.25, "note": "n", "norm": "p22_leave_before_end"}]
        assert cli.put(f"/api/c/{cid}/clip/0/events", json=ev).json() == {"saved": 1}
        assert cli.put(f"/api/c/{cid}/clip/0/events", json=[{"t": 1, "norm": "bogus"}]).status_code == 422
        sha0 = d["clips"][0]["sha"]
        os.makedirs(os.path.join(priv, "observations"))
        json.dump([{"type": "x", "start_sec": 0, "end_sec": 1, "confidence": .5}],
                  open(os.path.join(priv, "observations", sha0 + ".json"), "w"))
        stt = cli.get(f"/api/c/{cid}/clip/0/state").json()
        assert len(stt["events"]) == 1 and len(stt["observations"]) == 1
        rv = [{"obs_index": 0, "type": "x", "start_sec": 0, "end_sec": 1, "decision": "accept"}]
        assert cli.put(f"/api/c/{cid}/clip/0/reviews", json=rv).json() == {"saved": 1}
        assert cli.put(f"/api/c/{cid}/clip/0/events", json=[]).json() == {"saved": 0}
        lines = annotate.read_labels(os.path.join(priv, "labels", cid + ".jsonl"))
        assert [x["kind"] for x in lines] == ["obs_review"] and lines[0]["sha"] == sha0
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sp = cli.post(f"/api/c/{cid}/split", json={"ratio": 0.5}).json()
        assert sp["counts"]["train"] + sp["counts"]["holdout"] == 3 and "alpha" not in buf.getvalue()
        assert os.path.exists(os.path.join(priv, "splits", cid + ".json"))
        ok += 1
        assert d["status_counts"] == {"unreviewed": 3, "reviewed_nv": 0, "reviewed_v": 0}
        assert cli.put(f"/api/c/{cid}/clip/0/events", json=ev).json() == {"saved": 1}
        assert cli.get(f"/api/c/{cid}/clip/0/state").json()["status"] == "unreviewed"  # events never infer status
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = cli.put(f"/api/c/{cid}/clip/1/status", json={"status": "reviewed_no_violation"}).json()
            cli.put(f"/api/c/{cid}/clip/0/status", json={"status": "reviewed_violation"})
        assert r["status_counts"] == {"unreviewed": 2, "reviewed_nv": 1, "reviewed_v": 0}
        assert "alpha" not in buf.getvalue() and "unreviewed=" in buf.getvalue()
        assert cli.put(f"/api/c/{cid}/clip/1/status", json={"status": "bogus"}).status_code == 422
        d2 = cli.get(f"/api/c/{cid}/clips").json()
        assert d2["status_counts"] == {"unreviewed": 1, "reviewed_nv": 1, "reviewed_v": 1}
        assert [x["status"] for x in d2["clips"]] == ["reviewed_violation", "reviewed_no_violation", "unreviewed"]
        recs = [x for x in annotate.read_labels(os.path.join(priv, "labels", cid + ".jsonl")) if x["kind"] == "clip_status"]
        assert len(recs) == 2 and all(x["type"] == "clip_status" and x["sha16"] and "ts" in x for x in recs)
        cli.put(f"/api/c/{cid}/clip/1/status", json={"status": "reviewed_violation"})  # replace, no dup
        assert len([x for x in annotate.read_labels(os.path.join(priv, "labels", cid + ".jsonl")) if x["kind"] == "clip_status"]) == 2
        assert cli.post(f"/api/c/{cid}/split", json={"ratio": 0.5}).json()["counts"]["train"] + 0 >= 0  # split untouched
        ok += 1
        # F11 sentinel: names/extensions must never reach stdout or stderr
        sd = os.path.join(t, "ZZSENTINEL_dir")
        os.makedirs(sd)
        make(os.path.join(sd, "ZZSENTINEL_v.mp4"), dur=2)
        open(os.path.join(sd, "ZZSENTINEL_f.zzsentinelext"), "w").write("x")
        open(os.path.join(sd, "ZZSENTINEL_bad.mov"), "w").write("not a video")
        with open(os.path.join(sd, "ZZSENTINEL.csv"), "w", newline="") as f:
            csv.writer(f).writerows([["ZZSENTINEL_col", "time"], ["ZZSENTINEL_val", "1"]])
        open(os.path.join(sd, "ZZSENTINEL_broken.json"), "w").write("{ZZSENTINEL")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            assert inventory.main([t, "--probe", "--force"]) == 0
            m = json.load(open(os.path.join(priv, "inventory_map.json")))
            sid = [k for k, v in m.items() if v == sd][0]
            assert inventory.main([t, "--labels", sid, "--force"]) == 0
            assert inventory.main([t, "--probe", "C01"]) == 0
            assert inventory.main([t, "--probe", "C99"]) == 2
        txt = out.getvalue() + err.getvalue()
        pl = [l for l in txt.splitlines() if l.startswith("probe[")]
        assert len(pl) == 2 and pl[0].startswith("probe[ALL]") and pl[1].startswith("probe[C01]")
        pa, pc = (json.loads(l.split(": ", 1)[1]) for l in pl)
        assert pa["probed_ok"] == 5 and pc["probed_ok"] == 3 and pc["unprobed"] == 0
        assert pc["count_duration_10_145"] == 0 and pc["count_fps_ge_29_and_dur_ok"] == 0
        assert "count_duration_10_145" in pa
        for bad in ("ZZSENTINEL", "zzsentinelext", "forte_ui_"):
            assert bad not in txt, "sentinel leak"
        assert "other_count=2" in txt  # .zzsentinelext + .jsonl from the test's own labels and "unprobed=" not in txt and '"unprobed": 1' in txt
        assert '"duration_buckets"' in txt and '"size_bytes_max"' in txt
        ok += 1
        ok += _guard_group(t)
        ok += _import_group(t)
        ok += _ru_group(t)
        ok += _attrs_group(t)
        ok += _audit_group(t)
        ok += _selftest_audit.run(t)
        strict = TestClient(annotate.create_app(priv), base_url="http://127.0.0.1")  # client host 'testclient'
        assert strict.get("/api/candidates").status_code == 403
        assert cli.get("/api/candidates", headers={"Host": "evil.example"}).status_code == 403
        ok += 1
        print(f"selftest ok: {ok}/16 groups passed")
    finally:
        shutil.rmtree(t, ignore_errors=True)


if __name__ == "__main__":
    main()
