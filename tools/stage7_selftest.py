"""Synthetic self-test for tools/stage7_wcr.py (no real data, no video).

usage: python -m tools.stage7_selftest DIR [--n-perm 200] [--power-reps 5]
Builds isolated LOCAL_PRIVATE_DIR fixtures under DIR: 17 train + 4 holdout dummy clip files (holdout chmod 000; the
probe never opens clip files), imported labels, split, inventory map, and SYNTHETIC numeric caches written directly:
stage-4-format `<sha>_4.npz` (4 fps motion grid with per-clip scene level, operator person box) and stage-5-format
`cache_emb_siglip2/<sha>.npz` (768-d unit vectors: per-clip scene offset + AR drift + noise).
Label positions U(0.3, 0.95) x duration except `nearend` (U(0.7, 0.95)).
Scenarios (expected): planted (cue in [tl-8, tl] of all 5 positives, both families) -> GO (A_sem GO required);
  null -> NO-GO; scene (positives differ only by a static whole-clip shift) -> NO-GO;
  edge (the same cue at a positive-like position in EVERY clip) -> NO-GO; cue4 (cue in 4/5) -> reported;
  missing (planted, one train embedding cache deleted) -> A_sem incomplete, A_sem no GO;
  nearend (planted, labels near the clip end) -> NO-GO: localisation not identifiable vs the position prior.
Plus: open() spy on an in-process compute (no holdout path opened), holdout mode 000, no sha/path/sentinel in
outputs, summary mode 0600, transplant-null non-degeneracy, event-F1 == evaluate.match_events, outer-clip
perturbation test (0 prediction changes), memory watchdog kill on the real child entry point, bounded power.
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
SENT = "ZZS7SENT"
OP_BOX = (0.40, 0.20, 0.60, 0.70)
D = 768


def durations(rng):
    return np.r_[rng.uniform(17, 32, 15), 60.0, 142.0]


def sem_seq(rng, t, scene=1.0, noise=0.35, drift=0.2):
    s = rng.normal(size=D); s *= scene / np.linalg.norm(s)
    ar, x = np.zeros((len(t), D)), np.zeros(D)
    for k in range(len(t)):
        x = 0.9 * x + rng.normal(size=D) * np.sqrt(0.19); ar[k] = x
    return s + ar * drift / np.sqrt(D) + rng.normal(size=(len(t), D)) * noise / np.sqrt(D)


def mot_cache(path, rng, dur, cue_iv, level_shift=0.0, amp=0.35):
    t = np.arange(0, dur, 0.25)
    lvl = rng.uniform(0.02, 0.15) + level_shift
    g = np.clip(lvl + 0.05 * rng.standard_normal((len(t), 27, 48)), 0, 1)
    rows = []
    for i, ti in enumerate(t):
        j = rng.normal(0, 0.005, 4)
        rows.append([i, *(np.array(OP_BOX) + j), 0.9, 0])
        if rng.random() < 0.3:
            rows.append([i, 0.05, 0.2, 0.2, 0.7, 0.8, 0])
        for a, b in cue_iv:
            if a <= ti <= b:
                g[i, 8:14, 20:28] = np.clip(g[i, 8:14, 20:28] + amp, 0, 1)   # inside the operator zone
    np.savez_compressed(path, t=t.astype(np.float32), grid=g.astype(np.float16),
                        dets=np.array(rows, np.float32).reshape(-1, 7), dur=np.float32(dur))


def build(base, scenario, seed=0, amp=0.15, n_cue=5):
    relpos = (0.7, 0.95) if scenario == "nearend" else (0.3, 0.95)
    from tools.annotate import sha_prefix
    root = os.path.join(base, scenario)
    shutil.rmtree(root, ignore_errors=True)
    cdir, priv = os.path.join(root, f"clips_{SENT}"), os.path.join(root, "priv")
    os.makedirs(cdir)
    for d in ("labels", "splits", os.path.join("train_probe", "cache"), os.path.join("train_probe", "cache_emb_siglip2")):
        os.makedirs(os.path.join(priv, d))
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(21):
        p = os.path.join(cdir, f"{SENT}_v{i:02d}.mp4")
        open(p, "wb").write(rng.bytes(64) + bytes([i]))
        paths.append(p)
    sh = [sha_prefix(p) for p in paths]
    hold = paths[17:]
    for p in hold:
        os.chmod(p, 0)
    dur = durations(rng)
    pos = sorted(rng.choice(np.argsort(dur)[:10], 5, replace=False).tolist())
    tl = {i: float(rng.uniform(*relpos) * dur[i]) for i in pos}
    rels = [tl[i] / dur[i] for i in pos]
    u, v = rng.normal(size=D), rng.normal(size=D)
    u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    for i in range(17):
        t = np.arange(0.5, dur[i], 1.0)
        e = sem_seq(rng, t)
        cue = []
        if scenario in ("planted", "cue4", "missing", "nearend") and i in pos and (scenario != "cue4" or i != pos[0]) and n_cue:
            cue = [(tl[i] - 8, tl[i])]
        if scenario == "edge":
            r = rels[i % 5]; cue = [(r * dur[i] - 8, r * dur[i])]
        for a, b in cue:
            c = u + 0.7 * rng.normal(size=D) / np.sqrt(D); c /= np.linalg.norm(c)
            e[(t >= a) & (t <= b)] += amp * c
        if scenario == "scene" and i in pos:
            e += 0.6 * v
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        if not (scenario == "missing" and i == 3):
            np.savez_compressed(os.path.join(priv, "train_probe", "cache_emb_siglip2", f"{sh[i]}.npz"),
                                t=t.astype(np.float32), E_full=e.astype(np.float16), E_zone=e.astype(np.float16),
                                zone_fallback=np.int8(0), dur_dec=np.float32(dur[i]))
        mot_cache(os.path.join(priv, "train_probe", "cache", f"{sh[i]}_4.npz"), rng, dur[i], cue,
                  level_shift=0.15 if (scenario == "scene" and i in pos) else 0.0)
    json.dump({"C01": cdir}, open(os.path.join(priv, "inventory_map.json"), "w"))
    json.dump({"clips": {s: ("holdout" if i >= 17 else "train") for i, s in enumerate(sh)}},
              open(os.path.join(priv, "splits", "C01.json"), "w"))
    recs = []
    for i, s in enumerate(sh):
        y = i in pos or i in (17, 18)
        recs.append({"kind": "imported_clip_status", "clip": i, "sha16": s,
                     "status": "reviewed_violation" if y else "reviewed_no_violation"})
        if y:
            recs.append({"kind": "imported_event", "clip": i, "sha16": s,
                         "timestamp_sec": round(tl.get(i, 5.0), 3)})
    open(os.path.join(priv, "labels", "C01.imported.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in recs))
    for k in (4, 5):
        open(os.path.join(priv, "train_probe", f"stage{k}_summary.json"), "w").write("{}")
    return root, priv, sh, hold


def point(priv):
    from tools import stage4_probe as S4, train_probe as TP
    S4.LP = TP.LP = priv
    os.environ["LOCAL_PRIVATE_DIR"] = priv


def run(base, scenario, n_perm):
    from tools import stage7_wcr as S7
    root, priv, sh, hold = build(base, scenario)
    point(priv)
    out, err = io.StringIO(), io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = S7.main(["--candidate", "C01", "--n-perm", str(n_perm), "--budget-total-sec", "900"])
    wall = time.time() - t0
    sp = os.path.join(priv, "train_probe", "stage7_summary.json")
    summ = open(sp).read() if rc == 0 and os.path.exists(sp) else ""
    blob = out.getvalue() + err.getvalue() + summ
    res = json.loads(summ) if summ else {}
    checks = {"rc0": rc == 0, "holdout_mode_000": all(stat.S_IMODE(os.stat(h).st_mode) == 0 for h in hold),
              "no_sha_path_sentinel": not any(s in blob for s in sh) and SENT not in blob and root not in blob,
              "stderr_empty": err.getvalue() == ""}
    agg = "no summary"
    if res:
        checks["summary_mode_0600"] = stat.S_IMODE(os.stat(sp).st_mode) == 0o600
        checks["immutable_unchanged"] = res["immutable_files"]["status"] == "unchanged"
        F = res["families"]
        for fam, f in F.items():
            if "transplant_N1" in f:
                n1 = f["transplant_N1"]
                checks[f"{fam}_N1_nondegenerate"] = all(n1[a]["pool_sd"] > 0 and n1[a]["null_distinct"] > 50
                                                        for a in ("relative", "end"))
                checks[f"{fam}_eventF1_matches_evaluate"] = bool(f["event_f1_matches_evaluate"])
        g = {fam: F[fam]["criteria"]["family_GO"] for fam in F}
        if scenario == "planted":
            checks["GO_overall_and_A_sem"] = res["decision_GO"] == 1 and g["A_sem"] == 1
        elif scenario in ("null", "scene", "edge", "nearend"):
            checks["NO_GO"] = res["decision_GO"] == 0
        elif scenario == "missing":
            checks["A_sem_incomplete_no_GO"] = F["A_sem"]["failed_units"] == 1 and \
                F["A_sem"]["criteria"]["7_family_complete"] == 0 and g["A_sem"] == 0
        agg = " | ".join(
            f"{fam}: F1 {f.get('clip_f1', float('nan')):.2f} hit {f.get('hit_timestamp_endpoint', float('nan')):.2f} "
            f"evF1 {f.get('event_f1', float('nan')):.2f} pN2(f1/hit/ev) "
            f"{'/'.join(f'{f['perm_N2'][k]['p_perm']:.3f}' for k in ('f1', 'hit', 'event_f1')) if 'perm_N2' in f else '-'} "
            f"pN1(rel/end) {f['transplant_N1']['relative']['p']:.4f}/{f['transplant_N1']['end']['p']:.4f} "
            f"rho {f['abs_rho_inner_scores_vs_duration_median_max'][0]:.2f} GO {f['criteria']['family_GO']} "
            f"fail={[k for k, v in f['criteria'].items() if not v and k != 'family_GO']}"
            if "perm_N2" in f else f"{fam}: {f.get('status')}" for fam, f in F.items())
        agg += f" | wall {wall:.0f}s child_rss {res['child_peak_rss_mb']:.0f}MB"
    ok = all(checks.values())
    print(f"{scenario:8s} {'PASS' if ok else 'FAIL'} {agg} " + " ".join(k for k, v in checks.items() if not v),
          flush=True)
    return ok, priv, hold


def spy_compute(priv, hold):
    """in-process compute under an open() spy: no holdout path may be opened."""
    from tools import stage7_wcr as S7
    point(priv)
    seen, real = [], builtins.open

    def spy(f, *a, **k):
        seen.append(os.path.abspath(str(f)) if isinstance(f, (str, os.PathLike)) else "")
        return real(f, *a, **k)
    builtins.open = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            S7.compute("C01", 5, time.time() + 300)
    finally:
        builtins.open = real
    ok = not any(h in seen for h in map(os.path.abspath, hold)) and len(seen) > 0
    print(f"spy      {'PASS' if ok else 'FAIL'} in-process compute opened {len(seen)} files, holdout paths opened: "
          f"{sum(h in seen for h in map(os.path.abspath, hold))}", flush=True)
    return ok


def perturbation(priv):
    from tools import stage7_wcr as S7
    point(priv)
    U, _c = S7.load("C01")
    ok_all = True
    for key in ("sem", "mot"):
        V = [u for u in U if u[key] is not None]
        Fm = S7.Family([S7.prep(*u[key]) for u in V])
        y = [u["y"] for u in V]; tl = [u["tl"] for u in V]; dur = [u["dur"] for u in V]
        b = S7.nested(Fm, y, tl, dur)
        changed, inner = 0, 0
        for k in range(len(V)):
            for mode in ("flip", "shift"):
                y2, tl2 = list(y), list(tl)
                if mode == "flip":
                    y2[k] = 1 - y2[k]; tl2[k] = [dur[k] * 0.5] if y2[k] else []
                else:
                    tl2[k] = [t + 7.0 for t in tl[k]] if tl[k] else [dur[k] / 3]
                r = S7.nested(Fm, y2, tl2, dur)
                changed += int(r["pred"][k] != b["pred"][k] or not np.isclose(r["tpred"][k], b["tpred"][k])
                               or not np.isclose(r["score"][k], b["score"][k]))
                if mode == "flip" and y[k]:
                    inner = max(inner, int(sum(not np.isclose(r["score"][m], b["score"][m]) for m in range(len(V))
                                               if m != k)))
        ok = changed == 0 and inner > 0
        ok_all &= ok
        print(f"perturb  {'PASS' if ok else 'FAIL'} {key}: outer perturbations={2 * len(V)} own-prediction changes="
              f"{changed}; a positive-label flip changed other clips' scores: {inner}", flush=True)
    return ok_all


def memtest():
    from tools import stage7_wcr as S7
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = S7.main(["--candidate", "C01", "--n-perm", "1", "--memtest", "300", "--mem-mb", "150"])
    ok = rc == 0 and "memory_cap" in out.getvalue()
    print(f"memtest  {'PASS' if ok else 'FAIL'} {out.getvalue().strip()}", flush=True)
    return ok


def power(base, reps, n_perm):
    """bounded power, in-process, A_sem fixture at amp 0.12 and 0.15, cue in 5/5 and 4/5 (GO of A_sem alone)."""
    from tools import stage7_wcr as S7
    for amp, sc in ((0.12, "planted"), (0.15, "planted"), (0.15, "cue4")):
        go, t0 = [], time.time()
        for r in range(reps):
            _root, priv, _sh, _h = build(os.path.join(base, "power"), sc, seed=100 + r, amp=amp)
            point(priv)
            U, cov = S7.load("C01")
            V = [u for u in U if u["sem"]]
            Fm = S7.Family([S7.prep(*u["sem"]) for u in V])
            f = S7.evaluate_family(Fm, [u["y"] for u in V], [u["tl"] for u in V], [u["dur"] for u in V], n_perm, r,
                                   time.time() + 600, True)
            go.append(f["criteria"]["family_GO"])
        print(f"power    A_sem amp {amp} {'4/5' if sc == 'cue4' else '5/5'}: GO {np.mean(go):.2f} over {reps} reps "
              f"(n_perm {n_perm}, alpha {S7.ALPHA}), {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    base = a[0]
    n_perm = int(a[a.index("--n-perm") + 1]) if "--n-perm" in a else 200
    reps = int(a[a.index("--power-reps") + 1]) if "--power-reps" in a else 5
    t0 = time.time()
    res = []
    for sc in ("planted", "null", "scene", "edge", "cue4", "missing", "nearend"):
        ok, priv, hold = run(base, sc, n_perm)
        res.append(ok if sc != "cue4" else True)
        if sc == "planted":
            res.append(spy_compute(priv, hold))
            res.append(perturbation(priv))
    res.append(memtest())
    if reps:
        power(base, reps, n_perm)
    for d in os.listdir(base):
        for r, _ds, fs in os.walk(os.path.join(base, d)):
            for f in fs:
                p = os.path.join(r, f)
                if stat.S_IMODE(os.stat(p).st_mode) == 0:
                    os.chmod(p, 0o600)
    print(f"SELFTEST {sum(res)}/{len(res)} PASS, wall {time.time() - t0:.0f}s")
    return 0 if all(res) else 1


if __name__ == "__main__":
    sys.exit(main())
