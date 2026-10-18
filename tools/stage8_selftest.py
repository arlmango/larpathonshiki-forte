"""Synthetic self-test for tools/stage8_audit.py (no real data, no video, no real labels).

usage: python -m tools.stage8_selftest DIR
Builds isolated LOCAL_PRIVATE_DIR fixtures under DIR with tools.stage7_selftest.build (17 train + 4 holdout dummy
clip files, holdout chmod 000; the gate never opens clip files or caches), then writes a SYNTHETIC interval snapshot
whose payloads are produced by the real schema validator (annotate_intervals.clean_payload) and checks:
  fail_real_like  2 numeric-interval positives, 3 unknown, 1 same-clip confident control, 0 external,
                  + 1 record on a holdout sha and 1 superseded revision -> STOP_INSUFFICIENT_DATA, exact counts
  pass            4 numeric + 1 unknown; 1 same-clip + 2 external confident, 1 external low -> GATE_PASS
  fail_controls   4 numeric (+1 unknown-boundary only) ; 2 external confident, 1 same-clip low, 1 overlapping
                  -> STOP (controls 2 < 3 only)
  ctl_in_unknown  4 numeric + 1 unknown-flagged task with a good/sure same-clip control -> C 0,
                  n_same_clip_controls_in_unknown_tasks 1, STOP (controls only)
  refusals        snapshot sha mismatch, lock mismatch, second run (summary exists) -> rc 2, no new summary
  privacy         open() spy: no holdout path, no cache, no clip file opened; --print-lock opens no private file;
                  holdout mode 000; no sha16/sentinel/fixture path/planted interval time in stdout, stderr, summary;
                  summary mode 0400
"""
import builtins
import contextlib
import hashlib
import io
import json
import os
import stat
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import stage7_selftest as S7T  # noqa: E402
from tools.annotate_intervals import clean_payload, task_id  # noqa: E402

DUR = 17.0                                  # <= the shortest fixture clip; intervals validated against it
TIMES = (3.317, 7.913, 9.271, 12.643, 1.187, 4.429, 13.571, 15.733)   # planted, must never appear in outputs


def pos_payload(intervals=(), flags=(), ctls=(), done=True):
    p = {"intervals": [dict({"start_unknown": False, "end_unknown": False, "start_uncertain": False,
                             "end_uncertain": False, "visibility": "good", "confidence": "sure"}, **iv)
                       for iv in intervals],
         "clip_flags": {k: k in flags for k in ("not_visible", "unknown", "event_unconfirmed")},
         "same_clip_controls": [dict({"visibility": "good", "confidence": "sure", "comparable_visible": True,
                                      "no_violation": True}, **c) for c in ctls],
         "control_not_found": not ctls, "done": done}
    return clean_payload("positive", p, DUR, {"viewed": True})


def ctl_payload(parent, k, vis="good", conf="sure", answer=None):
    p = {"intervals": [] if answer else [{"start": TIMES[4], "end": TIMES[5], "visibility": vis, "confidence": conf,
                                          "comparable_visible": True, "no_violation": True}],
         "answer": answer, "done": True}
    out = clean_payload("control", p, DUR, {"viewed": True, "parent": parent})
    out["k"] = k
    return out


def fixture(base, name, spec):
    root, priv, sh, hold = S7T.build(base, name)
    pos = [sh[i] for i in range(17) if _status(priv, sh[i]) == "reviewed_violation"]
    neg = [sh[i] for i in range(17) if _status(priv, sh[i]) == "reviewed_no_violation"]
    for d in ("snapshots",):
        os.makedirs(os.path.join(priv, "labels", d), exist_ok=True)
    recs = []
    for i, (s, pl) in enumerate(zip(pos, spec["pos"])):
        tid = task_id(s, "positive", i)
        if i == 0:   # superseded revision: must be ignored (latest rev wins)
            recs.append({"task_id": tid, "sha16": s, "role": "positive", "rev": 1, "ts": 0,
                         "payload": pos_payload(flags=("unknown",))})
        recs.append({"task_id": tid, "sha16": s, "role": "positive", "rev": 2, "ts": 1, "payload": pl})
    for j, (vis, conf, ans) in enumerate(spec.get("ext", [])):
        parent = task_id(pos[j], "positive", j)
        recs.append({"task_id": task_id(neg[j], "control", 0), "sha16": neg[j], "role": "control", "rev": 1, "ts": 2,
                     "payload": ctl_payload(parent, 0, vis, conf, ans)})
    if spec.get("holdout_record"):
        recs.append({"task_id": task_id(sh[17], "positive", 9), "sha16": sh[17], "role": "positive", "rev": 1,
                     "ts": 3, "payload": pos_payload([{"start": TIMES[0], "end": TIMES[1]}])})
    recs.insert(0, {"task_id": "_assign", "sha16": "", "role": "assignment", "rev": 1, "ts": 0, "payload": {}})
    rel = "labels/snapshots/C01.intervals.v1.TEST.jsonl"
    raw = "".join(json.dumps(r, sort_keys=True) + "\n" for r in recs).encode()
    with open(os.path.join(priv, rel), "wb") as f:
        f.write(raw)
    os.chmod(os.path.join(priv, rel), 0o400)
    for rp in ("labels/C01.attrs.jsonl", "labels/C01.audit_r1.jsonl", "train_probe/stage7_summary.json"):
        with open(os.path.join(priv, rp), "w") as f:
            f.write("{}\n")
    return root, priv, sh, hold, rel, hashlib.sha256(raw).hexdigest()


def _status(priv, sha):
    with open(os.path.join(priv, "labels", "C01.imported.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["sha16"] == sha and r["kind"] == "imported_clip_status":
                return r["status"]
    return None


IV = [{"start": TIMES[0], "end": TIMES[1], "start_uncertain": True}, {"start": TIMES[2], "end": TIMES[3],
                                                                      "end_uncertain": True}]
SC = {"start": TIMES[6], "end": TIMES[7]}
SPECS = {
    "fail_real_like": {"pos": [pos_payload([IV[0]]), pos_payload([IV[1]], ctls=[SC]),
                               pos_payload(flags=("unknown",)), pos_payload(flags=("not_visible",)),
                               pos_payload(flags=("event_unconfirmed",))], "holdout_record": True,
                       "expect": {"outcome": "STOP_INSUFFICIENT_DATA", "n_with_numeric_intervals": 2,
                                  "n_intervals": 2, "n_boundary_uncertain": 2, "n_good_sure": 2, "n_unknown_tasks": 3,
                                  "n_same_clip_controls_confirmed_confident": 1, "n_external_controls": 0,
                                  "n_records_excluded_not_train": 1, "n_positive_tasks_done": 5, "unmet": 2}},
    "pass": {"pos": [pos_payload([IV[0]], ctls=[SC]), pos_payload([IV[1]]), pos_payload([IV[0], IV[1]]),
                     pos_payload([IV[1]]), pos_payload(flags=("unknown",))],
             "ext": [("good", "sure", None), ("good", "sure", None), ("partial", "sure", None)],
             "expect": {"outcome": "GATE_PASS_SEPARATE_PROTOCOL_REQUIRED", "n_with_numeric_intervals": 4,
                        "n_intervals": 5, "n_unknown_tasks": 1, "n_same_clip_controls_confirmed_confident": 1,
                        "n_external_controls": 2, "n_external_controls_low_confidence": 1, "unmet": 0}},
    "fail_controls": {"pos": [pos_payload([IV[0]], ctls=[dict(SC, confidence="unsure")]),
                              pos_payload([IV[1]], ctls=[{"start": TIMES[2], "end": TIMES[6]}]),
                              pos_payload([IV[0]]), pos_payload([IV[1]]),
                              pos_payload([{"start": TIMES[0], "end": None, "end_unknown": True}])],
                      "ext": [("good", "sure", None), ("good", "sure", None), ("good", "sure", "not_comparable")],
                      "expect": {"outcome": "STOP_INSUFFICIENT_DATA", "n_with_numeric_intervals": 4,
                                 "n_boundary_unknown": 1, "n_event_control_overlap": 1,
                                 "n_same_clip_controls_low_or_unconfirmed": 1, "n_external_controls": 2,
                                 "n_external_controls_unclear": 1, "unmet": 1}},
    "ctl_in_unknown": {"pos": [pos_payload([IV[0]]), pos_payload([IV[1]]), pos_payload([IV[0]]), pos_payload([IV[1]]),
                               pos_payload(flags=("unknown",), ctls=[SC])],
                       "expect": {"outcome": "STOP_INSUFFICIENT_DATA", "n_with_numeric_intervals": 4,
                                  "n_same_clip_controls_confirmed_confident": 0,
                                  "n_same_clip_controls_in_unknown_tasks": 1, "n_unknown_tasks": 1, "unmet": 1}},
}


def call(argv):
    from tools import stage8_audit as S8
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = S8.main(argv)
    return rc, out.getvalue(), err.getvalue()


def scenario(base, name, spec):
    from tools import stage8_audit as S8
    root, priv, sh, hold, rel, sha = fixture(base, name, spec)
    S7T.point(priv)
    lock = S8.lock_digest()[0]
    args = ["--candidate", "C01", "--gate-only", "--expect-lock", lock, "--snapshot", rel, "--snapshot-sha256", sha]
    seen, real = [], builtins.open

    def spy(f, *a, **k):
        seen.append(os.path.abspath(str(f)) if isinstance(f, (str, os.PathLike)) else "")
        return real(f, *a, **k)
    builtins.open = spy
    try:
        rc, out, err = call(args)
    finally:
        builtins.open = real
    sp = os.path.join(priv, "train_probe", "stage8_summary.json")
    summ = open(sp).read() if os.path.exists(sp) else ""
    res = json.loads(summ) if summ else {}
    g = res.get("gate_aggregates", {})
    ex = spec["expect"]
    c = {"rc0": rc == 0, "stderr_empty": err == "", "summary_mode_0400": bool(summ) and
         stat.S_IMODE(os.stat(sp).st_mode) == 0o400, "outcome": res.get("outcome") == ex["outcome"],
         "unmet_n": len(g.get("gate", {}).get("unmet", [None] * 9)) == ex["unmet"],
         "counts": all(g.get(k) == v for k, v in ex.items() if k not in ("outcome", "unmet")),
         "snapshot_verified_unchanged": res.get("snapshot") == {"sha256_verified": True, "unchanged": True},
         "immutables_unchanged": res.get("immutable_files", {}).get("status") == "unchanged",
         "holdout_not_opened": not any(os.path.abspath(h) in seen for h in hold),
         "no_cache_or_clip_opened": not any(("cache" in p and p.startswith(priv)) or "clips_" in p for p in seen),
         "holdout_mode_000": all(stat.S_IMODE(os.stat(h).st_mode) == 0 for h in hold)}
    blob = out + err + summ
    c["no_leak"] = (not any(s in blob for s in sh) and S7T.SENT not in blob and root not in blob and priv not in blob
                    and not any(f"{t:.3f}" in blob or f"{t:.2f}" in blob for t in TIMES))
    # refusals: second run, lock mismatch, snapshot mismatch -> rc 2, summary untouched
    m0 = os.stat(sp).st_mtime_ns if summ else None
    r2 = call(args)[0]
    os.chmod(sp, 0o600); os.remove(sp)
    r3 = call(args[:4] + ["0" * 16] + args[5:])[0]
    r4 = call(args[:-1] + ["0" * 64])[0]
    c["refusals_rc2_no_summary"] = (r2, r3, r4) == (2, 2, 2) and not os.path.exists(sp) and m0 is not None
    ok = all(c.values())
    print(f"{name:15s} {'PASS' if ok else 'FAIL'} outcome={res.get('outcome')} "
          f"unmet={len(g.get('gate', {}).get('unmet', []))} opened={len(seen)} "
          + " ".join(k for k, v in c.items() if not v), flush=True)
    return ok, priv, hold


def print_lock(priv):
    seen, real = [], builtins.open

    def spy(f, *a, **k):
        seen.append(os.path.abspath(str(f)) if isinstance(f, (str, os.PathLike)) else "")
        return real(f, *a, **k)
    builtins.open = spy
    try:
        rc, out, err = call(["--print-lock"])
    finally:
        builtins.open = real
    ok = rc == 0 and "lock_sha256_16=" in out and not any(p.startswith(priv) for p in seen)
    print(f"print_lock      {'PASS' if ok else 'FAIL'} private files opened: {sum(p.startswith(priv) for p in seen)}",
          flush=True)
    return ok


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    base = a[0]
    res, priv = [], None
    for name, spec in SPECS.items():
        ok, priv, _hold = scenario(base, name, spec)
        res.append(ok)
    res.append(print_lock(priv))
    for d in os.listdir(base):
        for r, _ds, fs in os.walk(os.path.join(base, d)):
            for f in fs:
                p = os.path.join(r, f)
                if stat.S_IMODE(os.stat(p).st_mode) == 0:
                    os.chmod(p, 0o600)
    print(f"SELFTEST {sum(res)}/{len(res)} PASS")
    return 0 if all(res) else 1


if __name__ == "__main__":
    sys.exit(main())
