"""Synthetic self-test for tools/stage5_probe.py (no real data).

usage: python -m tools.stage5_selftest DIR [--n-perm 200] [--scenarios planted,null,long] [--power-reps 5 --power-perm 50]
Builds an isolated LOCAL_PRIVATE_DIR fixture under DIR: 17 train + 4 holdout rendered clips (holdout chmod 000,
holdout label records carry events), stage-4-style zone caches (one train cache deliberately missing ->
zone_fallback), imported labels, R1 (15 agree / 2 mismatch). Real SigLIP2 ONNX embeddings of the rendered frames.
Scenarios: planted - a large cue object in the operator zone during the 8 s before each positive's label -> GO;
           null    - no cue (distractors only)                                                         -> NO-GO;
           long    - a 30 s cue ending at the label -> timestamp-endpoint hit via run_end;
           corrupt - one train clip truncated to 40 % -> decode_incomplete, no cache, counted, no GO (incomplete).
Every scenario: open() spy proves no holdout path is opened in-process; holdout files stay mode 000; stdout, stderr
and stage5_summary.json contain no fixture sha, path or sentinel. Planted also runs the outer-clip perturbation test.
Power (protocol §9, bounded): embedding-level planting of the measured cue direction into null embeddings.
"""
import builtins
import contextlib
import io
import json
import os
import shutil
import stat
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SENT = "ZZS5SENT"
FPS_R, WH = 5, (640, 360)
OP_BOX = (0.40, 0.20, 0.60, 0.70)  # normalized operator box in the rendered frames (cy = 0.45 -> 'top' agrees)


def render(path, dur, rng, cue=None):
    import cv2
    w, h = WH
    bg = tuple(int(v) for v in rng.integers(40, 160, 3))
    ctr = tuple(int(v) for v in rng.integers(30, 200, 3))
    d0 = float(rng.uniform(0, max(0.1, dur - 6)))
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS_R, WH)
    for i in range(int(dur * FPS_R)):
        t = i / FPS_R
        fr = np.full((h, w, 3), bg, np.uint8)
        cv2.rectangle(fr, (40, 230), (600, 340), ctr, -1)
        cx = 320 + int(12 * np.sin(t))
        cv2.ellipse(fr, (cx, 100), (22, 28), 0, 0, 360, (200, 180, 160), -1)
        cv2.rectangle(fr, (cx - 35, 128), (cx + 35, 250), (40, 40, 160), -1)
        if d0 <= t < d0 + 6:  # distractor in EVERY clip: a client figure on the far left
            cv2.ellipse(fr, (90, 110), (22, 28), 0, 0, 360, (190, 170, 150), -1)
            cv2.rectangle(fr, (55, 138), (125, 250), (120, 60, 40), -1)
        if cue and cue[0] <= t < cue[1]:  # cue: large green box + white 'banknote' stacks inside the zone
            cv2.rectangle(fr, (250, 150), (390, 245), (30, 200, 40), -1)
            for k in range(4):
                cv2.rectangle(fr, (262 + 30 * k, 170), (284 + 30 * k, 230), (245, 245, 245), -1)
        vw.write(fr)
    vw.release()


def zone_cache(path, dur):
    t = np.arange(0, dur, 0.25)
    rows = [[i, *OP_BOX, 0.9, 0] for i in range(len(t))]
    np.savez_compressed(path, t=t.astype(np.float32), grid=np.zeros((len(t), 27, 48), np.float16),
                        dets=np.array(rows, np.float32).reshape(-1, 7), dur=np.float32(dur))


def build(base, scenario, seed=0):
    from tools.annotate import sha_prefix
    root = os.path.join(base, scenario)
    shutil.rmtree(root, ignore_errors=True)
    cdir, priv = os.path.join(root, f"clips_{SENT}"), os.path.join(root, "priv")
    os.makedirs(cdir)
    for d in ("labels", "splits", os.path.join("train_probe", "cache")):
        os.makedirs(os.path.join(priv, d))
    rng = np.random.default_rng(seed)
    paths, meta = [], []
    for i in range(21):
        y = i < 5 or i in (17, 18)
        dur = float(rng.uniform(40, 60) if (scenario == "long" and y) else rng.uniform(15, 60))
        tl = float(rng.uniform(32 if scenario == "long" else 10, dur - 1)) if y else None
        cue = None
        if y and i < 17 and scenario == "planted":
            cue = (tl - 8, tl)
        elif y and i < 17 and scenario == "long":
            cue = (tl - 30, tl)
        p = os.path.join(cdir, f"{SENT}_v{i:02d}.mp4")
        render(p, dur, rng, cue)
        if scenario == "corrupt" and i == 10:  # truncated train negative: registered as-is (sha of the bad file)
            sz = os.path.getsize(p)
            with open(p, "r+b") as f:
                f.truncate(int(sz * 0.4))
        paths.append(p)
        meta.append((y, tl, dur))
    sh = [sha_prefix(p) for p in paths]
    hold = paths[17:]
    for p in hold:
        os.chmod(p, 0)
    json.dump({"C01": cdir}, open(os.path.join(priv, "inventory_map.json"), "w"))
    json.dump({"clips": {s: ("holdout" if i >= 17 else "train") for i, s in enumerate(sh)}},
              open(os.path.join(priv, "splits", "C01.json"), "w"))
    recs, r1 = [], []
    for i, (s, (y, tl, dur)) in enumerate(zip(sh, meta)):
        recs.append({"kind": "imported_clip_status", "clip": i, "sha16": s,
                     "status": "reviewed_violation" if y else "reviewed_no_violation"})
        if tl is not None:
            recs.append({"kind": "imported_event", "clip": i, "sha16": s, "timestamp_sec": round(tl, 3)})
        if i < 17:
            if i != 16:  # one train clip without a zone cache -> zone_fallback
                zone_cache(os.path.join(priv, "train_probe", "cache", f"{s}_4.npz"), dur)
            r1.append({"kind": "audit_r1", "mode": "audit", "clip": i, "sha": s, "t": 1.0,
                       "r1": "bottom" if i in (4, 15) else "top"})
    open(os.path.join(priv, "labels", "C01.imported.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in recs))
    open(os.path.join(priv, "labels", "C01.audit_r1.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in r1))
    open(os.path.join(priv, "train_probe", "stage4_summary.json"), "w").write("{}")
    return root, priv, sh, hold


def _point(priv):
    from tools import stage4_probe as S4, train_probe as TP
    S4.LP = TP.LP = priv
    os.environ["LOCAL_PRIVATE_DIR"] = priv


def run(base, scenario, n_perm, sens):
    from tools import stage5_probe as S5
    root, priv, sh, hold = build(base, scenario)
    _point(priv)
    seen, real_open = [], builtins.open

    def spy(f, *a, **k):
        seen.append(os.path.abspath(str(f)) if isinstance(f, (str, os.PathLike)) else "")
        return real_open(f, *a, **k)
    out, err = io.StringIO(), io.StringIO()
    argv = ["--candidate", "C01", "--n-perm", str(n_perm), "--n-perm-sens", "20"] + ([] if sens else ["--no-sensitivity"])
    builtins.open = spy
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = S5.main(argv)
    finally:
        builtins.open = real_open
    wall = time.time() - t0
    sp = os.path.join(priv, "train_probe", "stage5_summary.json")
    summ = open(sp).read() if rc == 0 and os.path.exists(sp) else ""
    blob = out.getvalue() + err.getvalue() + summ
    res = json.loads(summ) if summ else {}
    checks = {"rc0": rc == 0, "holdout_never_opened": not any(h in seen for h in map(os.path.abspath, hold)),
              "holdout_mode_000": all(stat.S_IMODE(os.stat(h).st_mode) == 0 for h in hold),
              "no_sha_path_sentinel": not any(s in blob for s in sh) and SENT not in blob and root not in blob,
              "stderr_empty": err.getvalue() == ""}
    agg = "no summary"
    if res:
        p, d, c = res["primary_all_units"], res["decision"], res["coverage"]
        if scenario == "corrupt":
            checks["decode_incomplete_1"] = c["embed_child_status"].get("decode_incomplete", 0) == 1
            checks["counted_failed_1_units_16"] = c["excluded_failed_decode_or_embedding"] == 1 and c["units"] == 16
            checks["primary_incomplete_no_GO"] = (not c["primary_complete"]) and d["decision_GO"] == 0 and \
                d["primary_complete_all_units"] == 0 and "INCOMPLETE PRIMARY" in out.getvalue()
            checks["no_cache_for_bad_clip"] = not os.path.exists(
                os.path.join(priv, "train_probe", "cache_emb_siglip2", f"{sh[10]}.npz"))
        else:
            checks["units_17_none_failed"] = c["units"] == 17 and c["excluded_failed_decode_or_embedding"] == 0
        checks["zone_fallback_1"] = c["zone_fallback_full_frame"] == 1
        checks["immutable_unchanged"] = res["immutable_files"]["status"] == "unchanged"
        if scenario == "planted":
            checks["GO"] = d["decision_GO"] == 1
        elif scenario in ("null", "corrupt"):
            checks["NO_GO"] = d["decision_GO"] == 0
        else:
            checks["hit_ge_0.8"] = p["outer_hit_rate_timestamp_endpoint"] >= 0.8
            checks["run_end_chosen_majority"] = p["chosen_configs"]["localisation"].get("run_end", 0) >= 9
        if sens:
            checks["sens_and_ablation_ran"] = bool(res["SENSITIVITY_view_full_only"]) and \
                bool(res["ABLATION_zone_agree_only_NOT_DEPLOYABLE"]) and res["zone_groups"]["agree"] == 14
        agg = (f"f1={p['outer_clip_f1']:.3f} hit={p['outer_hit_rate_timestamp_endpoint']:.3f} "
               f"permF1p95={p['perm']['f1']['p95']:.3f} permHITp95={p['perm']['hit_rate']['p95']:.3f} "
               f"durF1={res['baseline_duration_only_nested']['clip_f1']:.3f} GO={d['decision_GO']} "
               f"loc={p['chosen_configs']['localisation']} nested_s(p50/p95/max)="
               f"{'/'.join(f'{v:.2f}' for v in p['nested_run_sec_p50_p95_max'])} embed_wall_total="
               f"{res['runtime_sec']['load_and_embed']:.0f}s wall={wall:.0f}s")
    for h in hold:
        os.chmod(h, stat.S_IRUSR | stat.S_IWUSR)
    ok = all(checks.values())
    print(f"{scenario:8s} {'PASS' if ok else 'FAIL'} {agg} " + " ".join(k for k, v in checks.items() if not v),
          flush=True)
    return ok, priv


def units_of(priv):
    from tools import stage5_encoder as ENC, stage5_probe as S5
    _point(priv)
    with contextlib.redirect_stdout(io.StringIO()):
        U, _c = S5.load_units("C01", 600, 8192)
    return U, ENC.text_probes()


def perturbation(priv):
    """Outer-clip label/time changes must not change that clip's prediction; inner flips must matter."""
    from tools import stage5_model as M
    U, Ut = units_of(priv)
    P = [M.prep(u, Ut) for u in U]
    cache = M.FoldCache(P)
    y = np.array([u["y"] for u in U]); tl = [u["tl"] for u in U]
    idx = list(range(len(U)))
    base = M.nested(U, P, cache, idx, y, tl)
    changed, inner = 0, 0
    for k in idx:
        for mode in ("flip", "shift"):
            y2, tl2 = y.copy(), list(tl)
            if mode == "flip":
                y2[k] = 1 - y2[k]
                tl2[k] = [U[k]["dur"] / 2] if y2[k] else []
            else:
                tl2[k] = [t + 7.0 for t in tl[k]] if tl[k] else [U[k]["dur"] / 3]
            p2, t2, c2, _r = M.nested(U, P, cache, idx, y2, tl2)
            changed += int(p2[k] != base[0][k] or not np.allclose(t2[k], base[1][k], equal_nan=True) or c2[k] != base[2][k])
            if mode == "flip" and k == 0:
                inner = int(sum(p2[m] != base[0][m] or c2[m] != base[2][m] for m in idx if m != k))
    ok = changed == 0 and inner > 0
    print(f"perturb  {'PASS' if ok else 'FAIL'} outer-clip perturbations={2 * len(idx)} prediction_changes={changed}; "
          f"inner-label flip changed other clips' prediction/config: {inner}", flush=True)
    return ok


def power(priv_planted, priv_null, reps, n_perm, budget_sec=1800.0):
    """Bounded power (protocol §9): add the measured cue direction to null embeddings during the 8 s before
    re-drawn label times, cue visible in 5/5 or 4/5 positives; reps x full nested + n_perm permutations."""
    from tools import stage4_probe as S4, stage5_model as M, stage5_probe as S5
    Up, Ut = units_of(priv_planted)
    delta = {}
    for v in M.VIEWS:  # mean cue embedding minus mean non-cue embedding, positives of the planted fixture
        on = np.vstack([u["E"][v][(u["t"] >= u["tl"][0] - 8) & (u["t"] < u["tl"][0])] for u in Up if u["y"]])
        off = np.vstack([u["E"][v] for u in Up if not u["y"]])
        delta[v] = on.mean(0) - off.mean(0)
    Un, _ = units_of(priv_null)
    rng = np.random.default_rng(7)
    out, deadline = {}, time.time() + budget_sec
    for vis in (5, 4):
        go, t0 = [], time.time()
        for _ in range(reps):
            if time.time() > deadline:
                print("power    incomplete: budget", flush=True)
                break
            U = []
            pos_i = [i for i, u in enumerate(Un) if u["y"]]
            hidden = set(rng.choice(pos_i, 5 - vis, replace=False).tolist()) if vis < 5 else set()
            for i, u in enumerate(Un):
                u2 = dict(u, E={v: u["E"][v].copy() for v in M.VIEWS})
                if u["y"]:
                    tl = float(rng.uniform(10, u["dur"] - 1))
                    u2["tl"] = [tl]
                    if i not in hidden:
                        m = (u["t"] >= tl - 8) & (u["t"] < tl)
                        for v in M.VIEWS:
                            e = u2["E"][v][m] + delta[v]
                            u2["E"][v][m] = e / np.linalg.norm(e, axis=1, keepdims=True)
                U.append(u2)
            P = [M.prep(u, Ut) for u in U]
            cache = M.FoldCache(P)
            with contextlib.redirect_stdout(io.StringIO()):
                r = S5.evaluate(U, P, cache, list(range(len(U))), n_perm, 0)
                dur_f1 = S5.duration_only(U, list(range(len(U))))["clip_f1"]
            go.append(S5.decide(r, dur_f1)["decision_GO"])
        out[vis] = (float(np.mean(go)), time.time() - t0)
        print(f"power    cue visible in {vis}/5 positives: GO rate {np.mean(go) if go else float('nan'):.2f} over "
              f"{len(go)}/{reps} completed reps "
              f"(n_perm {n_perm}), {time.time() - t0:.0f}s", flush=True)
    _ = S4
    return out


if __name__ == "__main__":
    a = sys.argv
    base = a[1]
    n = int(a[a.index("--n-perm") + 1]) if "--n-perm" in a else 200
    scen = a[a.index("--scenarios") + 1].split(",") if "--scenarios" in a else ["planted", "null", "long", "corrupt"]
    reps = int(a[a.index("--power-reps") + 1]) if "--power-reps" in a else 0
    pperm = int(a[a.index("--power-perm") + 1]) if "--power-perm" in a else 50
    pbud = float(a[a.index("--power-budget-sec") + 1]) if "--power-budget-sec" in a else 1800.0
    oks, privs = [], {}
    for s in scen:
        ok, privs[s] = run(base, s, 20 if s == "corrupt" else n, sens=(s == "planted"))
        oks.append(ok)
        if s == "planted":
            oks.append(perturbation(privs[s]))
    if reps and "planted" in privs and "null" in privs:
        power(privs["planted"], privs["null"], reps, pperm, pbud)
    print(f"stage5 selftest {sum(oks)}/{len(oks)} passed")
    sys.exit(0 if all(oks) else 1)
