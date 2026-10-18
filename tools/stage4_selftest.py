"""Synthetic self-test for tools/stage4_probe.py (no real data). usage: python -m tools.stage4_selftest DIR [--n-perm 200]

Builds an isolated LOCAL_PRIVATE_DIR fixture under DIR: 17 train + 4 holdout synthetic clips (holdout chmod 000),
planted numeric caches, imported labels (holdout records carry events too) and an R1 file (15 agree / 2 mismatch).
Scenarios: planted  - operator absent during the 5 s PRECEDING each positive's label -> must be GO, F1/hit high;
           nosignal - every clip has one 5 s absence at a label-independent time   -> must be NO-GO;
           missing  - planted, one negative cache deleted -> decoded by the child from its synthetic TRAIN clip
                      (tests the decode path only: the decoded clip is a different 'camera', no signal claim).
Every scenario: an open() spy proves no holdout path is opened in-process; holdout files stay mode 000; stdout,
stderr and stage4_summary.json contain no fixture sha, path or sentinel.
"""
import builtins
import contextlib
import io
import json
import os
import shutil
import stat
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SENT = "ZZS4SENT"


def plant(path, rng, y, scenario):
    dur = float(rng.uniform(60, 150))
    t = np.arange(0, dur, 0.25)
    tl = float(rng.uniform(20, dur - 5)) if y else None
    if scenario == "planted":
        a0 = tl - 5 if y else float(rng.uniform(5, dur - 2))
        a1 = tl if y else a0 + 1.0
    else:
        a0 = float(rng.uniform(5, dur - 6)); a1 = a0 + 5
    rows = []
    for i, ti in enumerate(t):
        if not a0 <= ti < a1:
            rows.append([i, 0.42, 0.25, 0.58, 0.65, 0.9, 0])
        if rng.random() < 0.5:
            x = rng.choice([rng.uniform(0.0, 0.12), rng.uniform(0.8, 0.9)])  # clients well outside the zone
            rows.append([i, x, 0.86, x + 0.1, 0.99, 0.8, 0])
    np.savez_compressed(path, t=t.astype(np.float32), grid=(rng.random((len(t), 27, 48)) * 0.05).astype(np.float16),
                        dets=np.array(rows, np.float32).reshape(-1, 7), dur=np.float32(dur))
    return tl


def build(base, scenario, seed=0):
    from tools.annotate import sha_prefix
    from tools.make_synthetic import make
    root = os.path.join(base, scenario)
    shutil.rmtree(root, ignore_errors=True)
    cdir, priv = os.path.join(root, f"clips_{SENT}"), os.path.join(root, "priv")
    os.makedirs(cdir)
    for d in ("labels", "splits", os.path.join("train_probe", "cache")):
        os.makedirs(os.path.join(priv, d))
    paths = []
    for i in range(21):
        p = os.path.join(cdir, f"{SENT}_v{i:02d}.mp4")
        make(p, dur=12 if i == 16 else 1 + 0.1 * i)
        paths.append(p)
    sh = [sha_prefix(p) for p in paths]
    hold = paths[17:]
    for p in hold:
        os.chmod(p, 0)
    json.dump({"C01": cdir}, open(os.path.join(priv, "inventory_map.json"), "w"))
    json.dump({"clips": {s: ("holdout" if i >= 17 else "train") for i, s in enumerate(sh)}},
              open(os.path.join(priv, "splits", "C01.json"), "w"))
    rng = np.random.default_rng(seed)
    recs, r1 = [], []
    for i, s in enumerate(sh):
        y = i < 5 or i in (17, 18)
        recs.append({"kind": "imported_clip_status", "clip": i, "sha16": s,
                     "status": "reviewed_violation" if y else "reviewed_no_violation"})
        if i < 17:
            tl = plant(os.path.join(priv, "train_probe", "cache", f"{s}_4.npz"), rng, y, scenario)
            r1.append({"kind": "audit_r1", "mode": "audit", "clip": i, "sha": s, "t": 1.0,
                       "r1": "bottom" if i in (4, 15) else "top"})
        else:
            tl = 30.0 if y else None
        if tl is not None:
            recs.append({"kind": "imported_event", "clip": i, "sha16": s, "timestamp_sec": round(tl, 3)})
    open(os.path.join(priv, "labels", "C01.imported.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in recs))
    open(os.path.join(priv, "labels", "C01.audit_r1.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in r1))
    if scenario == "missing":
        os.remove(os.path.join(priv, "train_probe", "cache", f"{sh[16]}_4.npz"))
    return root, priv, sh, hold


def run(base, scenario, n_perm):
    from tools import stage4_probe as S4, train_probe as TP
    root, priv, sh, hold = build(base, scenario)
    S4.LP = TP.LP = priv
    os.environ["LOCAL_PRIVATE_DIR"] = priv
    os.environ["SYNTHETIC_DETECTOR"] = "1"
    seen, real_open = [], builtins.open

    def spy(f, *a, **k):
        seen.append(os.path.abspath(str(f)) if isinstance(f, (str, os.PathLike)) else "")
        return real_open(f, *a, **k)
    out, err = io.StringIO(), io.StringIO()
    builtins.open = spy
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = S4.main(["--candidate", "C01", "--n-perm", str(n_perm)])
    finally:
        builtins.open = real_open
    summ_txt = open(os.path.join(priv, "train_probe", "stage4_summary.json")).read() if rc == 0 else ""
    blob = out.getvalue() + err.getvalue() + summ_txt
    res = json.loads(summ_txt) if summ_txt else {}
    checks = {
        "rc0": rc == 0,
        "holdout_never_opened": not any(h in seen for h in map(os.path.abspath, hold)),
        "holdout_mode_000": all(stat.S_IMODE(os.stat(h).st_mode) == 0 for h in hold),
        "no_sha_path_sentinel_in_outputs": not any(s in blob for s in sh) and SENT not in blob and root not in blob,
        "stderr_empty": err.getvalue() == "",
    }
    if res:
        p, d = res["primary_all_units"], res["decision"]
        checks["zone_groups_15_2"] = res["zone_groups"] == {"agree": 15, "mismatch": 2, "no_r1": 0}
        if scenario == "planted":
            checks["GO"] = d["decision_GO"] == 1
            checks["f1_ge_0.9"] = p["outer_clip_f1"] >= 0.9
            checks["hit_ge_0.8"] = p["outer_hit_rate_timestamp_endpoint"] >= 0.8
        else:
            checks["NO_GO"] = d["decision_GO"] == 0
        if scenario == "missing":
            checks["decoded_1"] = res["coverage"]["caches_decoded"] == 1
            checks["none_excluded"] = res["coverage"]["excluded_failed_cache_or_zone"] == 0
            checks["spy_live_saw_train_clip_open"] = any(os.path.dirname(x).endswith(f"clips_{SENT}") for x in seen)
        agg = (f"f1={p['outer_clip_f1']:.3f} hit={p['outer_hit_rate_timestamp_endpoint']:.3f} "
               f"permF1p95={p['perm']['f1']['p95']:.3f} permHITp95={p['perm']['hit_rate']['p95']:.3f} "
               f"always+={p['always_positive_clip_f1']:.3f} GO={d['decision_GO']} "
               f"runtime={res['runtime_sec']['total']:.1f}s")
    else:
        agg = "no summary"
    for h in hold:
        os.chmod(h, stat.S_IRUSR | stat.S_IWUSR)  # allow cleanup
    ok = all(checks.values())
    print(f"{scenario:9s} {'PASS' if ok else 'FAIL'} {agg} " + " ".join(k for k, v in checks.items() if not v))
    return ok


if __name__ == "__main__":
    base = sys.argv[1]
    n = int(sys.argv[sys.argv.index("--n-perm") + 1]) if "--n-perm" in sys.argv else 200
    r = [run(base, s, n) for s in ("planted", "nosignal", "missing")]
    print(f"stage4 selftest {sum(r)}/{len(r)} passed")
    sys.exit(0 if all(r) else 1)
