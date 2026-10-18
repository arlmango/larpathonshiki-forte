"""Synthetic self-test for tools/stage10_weak.py (no real data, no video, no real labels).

usage: python -m tools.stage10_selftest DIR [--reps 10] [--reps-null 40]
CLI fixture (tools.stage9_selftest.fixture -> stage7_selftest.build: 17 train + 4 holdout dummy clip files, holdout
chmod 000, a holdout cache at mode 000, a holdout snapshot record, a superseded revision; synthetic 768-d caches):
2 interval positives (one start-uncertain, one partial), 3 unknown positive tasks, 12 negative import rows of which two
are made BYTE-IDENTICAL (same content sha16, two clip indices) -> 11 distinct negative units; planted cue in the
intervals.
  cli_planted    rc 0, summary 0400, outcome OBSERVED, n_pos 2 / n_neg 11, 22 pairs, integrity unchanged, no leak,
                 refusals (second run, wrong lock, wrong snapshot sha, stage-8 summary missing) -> rc 2, no summary
  prelock        --prelock-duration-auc: rc 0, PRELOCK line with n 2/11 and the duration AUC, writes no file, no leak
  open_spy       in-process compute(): no clip file, holdout path/cache or unknown-task cache opened; holdout 000
In memory (256-d generator, cue 0.25, same method code):
  rep_planted    planted cue -> OBSERVED rate >= 0.9, exact hits >= 0.9, mean LPO-AUC >= 0.9
  rep_null       no cue -> per-positive exact hit rate within 3 binomial SE of the mean exact chance share, and the
                 false OBSERVED rate <= uniform both-hit reference + 3 SE
  duration       positives long, negatives short, no cue -> duration separability 1.0 => descriptive-only flag;
                 K-grid max is less length-biased than an all-frames max (|AUC - 0.5| compared, reported)
  scene          positives share a scene-burst direction, negatives have their own, no action cue -> LPO-AUC
                 inflated (>= 0.7) while the top-frame hit rate stays within 3 SE of chance (reported)
  mask_weights   strict margin; outside-interval frames of positive clips never rows; unknown tasks not units;
                 equal clip weights, classes balanced, sum w = n rows; grid K = 20 incl. short clips
  leakage        pair model identical when the held-out pair's labels/features change
  print_lock     opens no private file
"""
import json
import os
import shutil
import stat
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import stage7_selftest as S7T, stage9_head as S9, stage9_selftest as S9T  # noqa: E402
from tools import stage10_weak as W  # noqa: E402

AMP, AMP_MEM, D_MEM, IV_LEN = 0.15, 0.25, 256, 12.0     # fixture parameters (not method parameters)


# ============================================================================ generator (synthetic only)
def seq(rng, t, D, scene=1.0, noise=0.35, drift=0.2):
    s = rng.normal(size=D); s *= scene / np.linalg.norm(s)
    ar, x = np.zeros((len(t), D)), np.zeros(D)
    for k in range(len(t)):
        x = 0.9 * x + rng.normal(size=D) * np.sqrt(0.19); ar[k] = x
    return s + ar * drift / np.sqrt(D) + rng.normal(size=(len(t), D)) * noise / np.sqrt(D)


def bursts(rng, n, duty=0.4, seg=4.0):
    a, on = np.zeros(n), rng.random() < duty
    for k in range(n):
        if rng.random() < 1.0 / seg:
            on = rng.random() < duty
        a[k] = on
    return a


def gen(rng, mode="planted", D=S7T.D, n_lab=2, n_unk=3, n_neg=12, amp=AMP):
    u = rng.normal(size=D); u /= np.linalg.norm(u)
    v = rng.normal(size=D); v /= np.linalg.norm(v)
    clips, pays, times = [], {}, []
    for j in range(n_lab + n_unk + n_neg):
        kind = "lab" if j < n_lab else ("unk" if j < n_lab + n_unk else "neg")
        if mode == "duration":
            dur = float(rng.uniform(100, 140) if kind != "neg" else rng.uniform(12, 40))
        else:
            dur = float(rng.uniform(50, 90) if kind != "neg" else rng.uniform(30, 140))
        t = np.arange(0.5, dur - 0.5, 1.0)
        E = seq(rng, t, D)
        if mode == "scene":
            w = v if kind != "neg" else rng.normal(size=D)
            E += amp * 1.5 * np.outer(bursts(rng, len(t)), w / np.linalg.norm(w))
        c = {"key": j, "kind": kind, "t": t, "dur": dur}
        if kind == "lab":
            s = round(float(rng.uniform(*((0.15, 0.3), (0.55, 0.7))[j % 2])) * dur, 3)   # prior lands off-interval
            iv = {"start": s, "end": round(s + IV_LEN, 3), "start_uncertain": j == 0,
                  "visibility": "partial" if j == 1 else "good"}
            times += [iv["start"], iv["end"]]
            if mode == "planted":
                m = (t >= iv["start"]) & (t <= iv["end"])
                cue = u + 0.7 * rng.normal(size=(m.sum(), D)) / np.sqrt(D)
                E[m] += amp * cue / np.linalg.norm(cue, axis=1, keepdims=True)
            pays[j] = S9T.pos_payload([iv], dur=dur)
        elif kind == "unk":
            pays[j] = S9T.pos_payload(flags=(("unknown", "not_visible", "event_unconfirmed")[j % 3],), dur=dur)
        c["E"] = E / np.linalg.norm(E, axis=1, keepdims=True)
        clips.append(c)
    return clips, pays, times


def in_memory(clips, pays):
    sha = {c["key"]: f"s{c['key']:02d}" for c in clips}
    latest = {f"t{k}": {"task_id": f"t{k}", "sha16": sha[k], "role": "positive", "rev": 1, "payload": p}
              for k, p in pays.items()}
    pos = {sha[c["key"]] for c in clips if c["kind"] != "neg"}
    neg = {sha[c["key"]] for c in clips if c["kind"] == "neg"}
    lab, _g = S9.supervision(latest, pos, neg)
    by = {sha[c["key"]]: c for c in clips}
    U = [{"P": S9.point_prep(by[s]["t"], by[s]["E"]), "dur": by[s]["dur"], "y": 1, "lab": lab[s]} for s in sorted(lab)]
    U += [{"P": S9.point_prep(c["t"], c["E"]), "dur": c["dur"], "y": 0, "lab": None}
          for c in clips if c["kind"] == "neg"]
    return U


# ============================================================================ CLI fixture (+ content-hash dedup)
def fixture(base, name, clips, pays):
    root, priv, sh, hold, unk, rel, ssha = S9T.fixture(base, name, clips, pays)
    lp = os.path.join(priv, "labels", "C01.imported.jsonl")
    recs = [json.loads(x) for x in open(lp)]
    negs = [sh.index(r["sha16"]) for r in recs if r["kind"] == "imported_clip_status"
            and r["status"] == "reviewed_no_violation" and sh.index(r["sha16"]) < 17]
    a, b = sorted(negs)[:2]
    cdir = os.path.join(root, f"clips_{S7T.SENT}")
    shutil.copyfile(os.path.join(cdir, f"{S7T.SENT}_v{a:02d}.mp4"), os.path.join(cdir, f"{S7T.SENT}_v{b:02d}.mp4"))
    for r in recs:
        if r["sha16"] == sh[b]:
            r["sha16"] = sh[a]                      # byte-identical file -> same content hash, second clip index
    open(lp, "w").write("".join(json.dumps(r) + "\n" for r in recs))
    sp = os.path.join(priv, "splits", "C01.json")
    split = json.load(open(sp)); split["clips"].pop(sh[b]); json.dump(split, open(sp, "w"))
    return root, priv, sh, hold, unk, rel, ssha, sh[a]


def leak_free(blob, sh, root, priv, times):
    return (not any(s in blob for s in sh) and S7T.SENT not in blob and root not in blob and priv not in blob
            and not any(f"{t:.3f}" in blob or f"{t:.2f}" in blob for t in times))


def cli(base):
    clips, pays, times = gen(np.random.default_rng(11), "planted")
    root, priv, sh, hold, unk, rel, ssha, used = fixture(base, "s10_planted", clips, pays)
    S7T.point(priv)
    lock = W.lock_digest()[0]
    snap = ["--snapshot", rel, "--snapshot-sha256", ssha]
    args = ["--candidate", "C01", "--run", "--expect-lock", lock] + snap
    tp_dir = os.path.join(priv, "train_probe")
    before = set(os.listdir(tp_dir))
    rcp, outp, errp = call(["--candidate", "C01", "--prelock-duration-auc"] + snap)
    pl = json.loads(outp.split("PRELOCK ", 1)[1]) if "PRELOCK " in outp else {}
    pins = json.loads(outp.split("PINS ", 1)[1].splitlines()[0]) if "PINS " in outp else {}
    args += ["--expect-inputs", pins.get("inputs_sha256_16", "")]
    c_pre = {"rc0": rcp == 0, "stderr_empty": errp == "", "n": (pl.get("n_positive_units"), pl.get("n_negative_units"))
             == (2, 11), "no_file_written": set(os.listdir(tp_dir)) == before,
             "no_leak": leak_free(outp + errp, sh, root, priv, times), "immutables": pl.get("immutables_unchanged"),
             "pins": pins.get("n_caches_in_manifest") == 13 and len(pins.get("inputs_sha256_16", "")) == 16}
    ok_pre = all(c_pre.values())
    print(f"prelock        {'PASS' if ok_pre else 'FAIL'} duration_auc={pl.get('duration_auc_longer_is_positive')} "
          f"separability={pl.get('duration_separability')} " + " ".join(k for k, v in c_pre.items() if not v))
    # in-process compute under an open() spy (the CLI computes in a child the spy cannot see)
    (res, seen) = spy_compute(rel, ssha)
    c_spy = {"holdout_not_opened": not any(os.path.abspath(h) in seen for h in hold),
             "unknown_caches_not_opened": not any(os.path.abspath(u) in seen for u in unk),
             "no_clip_file_opened": not any("clips_" in p for p in seen),
             "holdout_mode_000": all(stat.S_IMODE(os.stat(h).st_mode) == 0 for h in hold)}
    ok_spy = all(c_spy.values()) and res.get("outcome") == W.OBS
    print(f"open_spy       {'PASS' if ok_spy else 'FAIL'} opened={len(seen)} " + " ".join(k for k, v in c_spy.items() if not v))
    rc, out, err = call(args)
    sp = os.path.join(tp_dir, W.SUMMARY)
    summ = open(sp).read() if os.path.exists(sp) else ""
    r = json.loads(summ) if summ else {}
    c = {"rc0": rc == 0, "stderr_empty": err == "", "summary_0400": bool(summ) and stat.S_IMODE(os.stat(sp).st_mode) == 0o400,
         "outcome": r.get("outcome") == W.OBS, "dedup_units": (r.get("n_positive_units"), r.get("n_negative_units"))
         == (2, 11) and r.get("coverage", {}).get("n_train_negative_units_distinct_content") == 11,
         "pairs": (r.get("n_pairs"), r.get("n_pairs_trainable")) == (22, 22),
         "unknown_excluded": r.get("coverage", {}).get("n_positive_units_excluded_no_numeric_interval") == 3,
         "no_weights": r.get("model_artefact_saved") is False and set(os.listdir(tp_dir)) == before | {W.SUMMARY},
         "integrity": r.get("snapshot") == {"sha256_verified": True, "unchanged": True}
         and r.get("inputs", {}).get("verified") is True and r.get("inputs", {}).get("unchanged_after_run") is True
         and r.get("immutable_files", {}).get("status") == "unchanged" and r.get("stage8_summary_unchanged") is True,
         "holdout_record_excluded": r.get("supervision_counts", {}).get("n_records_excluded_not_train") == 1,
         "no_leak": leak_free(out + err + summ, sh, root, priv, times)}
    r2 = call(args)[0]
    os.chmod(sp, 0o600); os.remove(sp)
    r3 = call(args[:4] + ["0" * 16] + args[5:])[0]
    r4 = call(args[:8] + ["0" * 64] + args[9:])[0]
    r6 = call(args[:-1] + ["0" * 16])[0]                                   # wrong inputs digest
    with open(os.path.join(tp_dir, "cache_emb_siglip2", f"{used}.npz"), "ab") as f:
        f.write(b"\0")                                                    # tamper one USED negative cache
    r7 = call(args)[0]
    s8 = os.path.join(tp_dir, "stage8_summary.json")
    s8b = open(s8).read()
    os.remove(s8)
    r5 = call(args)[0]
    open(s8, "w").write(s8b)
    c["refusals_rc2_no_summary"] = (r2, r3, r4, r5, r6) == (2,) * 5 and not os.path.exists(sp)
    c["cache_tamper_refused"] = r7 == 2 and not os.path.exists(sp)
    ok = all(c.values())
    print(f"cli_planted    {'PASS' if ok else 'FAIL'} outcome={r.get('outcome')} lpo_auc={r.get('lpo_auc_clip_grid_max')} "
          f"exact={r.get('localisation', {}).get('exact_interval_hits')} clip={r.get('clip_level', {}).get('f1')} "
          f"runtime={r.get('runtime_sec')}s " + " ".join(k for k, v in c.items() if not v), flush=True)
    return [ok_pre, ok_spy, ok], priv


def call(argv):
    import contextlib
    import io
    o, e = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
        rc = W.main(argv)
    sys.excepthook = sys.__excepthook__
    return rc, o.getvalue(), e.getvalue()


def spy_compute(rel, ssha):
    import builtins
    seen, real = [], builtins.open

    def spy(f, *a, **k):
        seen.append(os.path.abspath(str(f)) if isinstance(f, (str, os.PathLike)) else "")
        return real(f, *a, **k)
    builtins.open = spy
    try:
        res = W.compute("C01", rel, ssha)
    finally:
        builtins.open = real
    return res, seen


# ============================================================================ in-memory checks
def reps(mode, n, seed0, D=D_MEM):
    R = []
    for s in range(n):
        clips, pays, _ = gen(np.random.default_rng(seed0 + s), mode, D=D, amp=AMP_MEM)
        U = in_memory(clips, pays)
        r = W.analyse(U)
        if mode == "duration":
            real = W.clip_score
            W.clip_score = lambda m, P: float(W.frame_scores(m, P).max())       # all-frames max (contrast only)
            try:
                r["allmax_auc"] = W.analyse(U)["lpo_auc_clip_grid_max"]
            finally:
                W.clip_score = real
        R.append(r)
    return R


def hit_stats(R):
    h = sum(r["localisation"]["exact_interval_hits"][0] for r in R)
    n = sum(r["localisation"]["exact_interval_hits"][1] for r in R)
    e = float(np.mean([x for r in R for x in r["localisation"]["chance_exact_share_sorted"]]))
    return h, n, e, np.sqrt(e * (1 - e) / n)


def check_reps(k, kn):
    out = []
    R = reps("planted", k, 1000)
    obs, a = np.mean([r["outcome"] == W.OBS for r in R]), np.mean([r["lpo_auc_clip_grid_max"] for r in R])
    h, n, e, _se = hit_stats(R)
    ok = obs >= 0.9 and a >= 0.9 and h / n >= 0.9
    print(f"rep_planted    {'PASS' if ok else 'FAIL'} observed {obs:.2f}; exact hits {h}/{n} (chance {e:.3f}); "
          f"mean LPO-AUC {a:.3f} (reps {k})"); out.append(ok)
    R = reps("null", kn, 5000)
    h, n, e, se = hit_stats(R)
    obs = np.mean([r["outcome"] == W.OBS for r in R])
    ref = np.mean([r["localisation"]["uniform_all_exact_hit_reference_product"] for r in R])
    a = np.mean([r["lpo_auc_clip_grid_max"] for r in R])
    se_o = np.sqrt(ref * (1 - ref) / len(R))
    ok = abs(h / n - e) <= 3 * se and obs <= ref + 3 * se_o
    print(f"rep_null       {'PASS' if ok else 'FAIL'} hits {h}/{n} = {h / n:.3f} vs chance {e:.3f} (3 SE {3 * se:.3f}); "
          f"observed {obs:.3f} vs uniform both-hit {ref:.3f}; mean LPO-AUC {a:.3f}"); out.append(ok)
    R = reps("duration", k, 7000)
    g = np.mean([abs(r["lpo_auc_clip_grid_max"] - 0.5) for r in R]); m = np.mean([abs(r["allmax_auc"] - 0.5) for r in R])
    flag = all(r["clip_level_descriptive_only"] and r["duration_separability"] == 1.0 for r in R)
    rho = np.median([r["spearman_heldout_clip_score_vs_duration"] for r in R])
    ok = flag and g < m
    print(f"duration       {'PASS' if ok else 'FAIL'} separability 1.0 + descriptive flag {int(flag)}; |AUC-0.5| grid "
          f"{g:.3f} vs all-frames max {m:.3f}; median rho(score, dur) grid {rho:.2f}"); out.append(ok)
    R = reps("scene", k, 9000)
    h, n, e, se = hit_stats(R)
    a = np.mean([r["lpo_auc_clip_grid_max"] for r in R]); obs = np.mean([r["outcome"] == W.OBS for r in R])
    pc = np.median([x for r in R for x in r["localisation"]["in_interval_mean_percentile_sorted"]])
    ok = a >= 0.7 and abs(h / n - e) <= 3 * se
    print(f"scene          {'PASS' if ok else 'FAIL'} LPO-AUC {a:.3f} (inflated by scene identity) while hits {h}/{n} "
          f"= {h / n:.3f} vs chance {e:.3f}; observed {obs:.2f}; median in-interval percentile {pc:.2f}"); out.append(ok)
    return out


def check_mask_weights():
    clips, pays, _ = gen(np.random.default_rng(7), "planted", D=D_MEM)
    U = in_memory(clips, pays)
    c = {"units_exclude_unknown": len(U) == 14 and sum(u["y"] for u in U) == 2,
         "constants": (W.MARGIN, W.LAMBDA, W.K_GRID, W.TOL_UNCERTAIN, W.TOL_CERTAIN) == (5.0, 10.0, 20, 5.0, 2.0)}
    u0 = U[0]
    m, w, y = W.clip_rows(u0)
    iv, t = u0["lab"]["ivs"][0], u0["P"]["t"]
    c["strict_margin"] = iv["su"] and m.any() and t[m].min() >= iv["start"] + 5 - 1e-9 and t[m].max() <= iv["end"]
    c["outside_never_rows"] = not (m & ~((t >= iv["start"]) & (t <= iv["end"]))).any()
    c["clip_weight_1"] = abs(w.sum() - 1) < 1e-12 and abs(W.clip_rows(U[5])[1].sum() - 1) < 1e-12
    cap = {}
    real = S9.fit_logreg
    S9.fit_logreg = lambda X, yy, ww, lam: (cap.update(y=yy, w=ww, X=X), real(X, yy, ww, lam=lam))[1]
    try:
        W.train(U, list(range(1, 14)))
    finally:
        S9.fit_logreg = real
    yy, ww = cap["y"], cap["w"]
    c["classes_balanced_sum_n"] = abs(ww[yy == 1].sum() - ww[yy == 0].sum()) < 1e-9 and abs(ww.sum() - len(yy)) < 1e-9
    nb = [len(U[i]["P"]["t"]) for i in range(2, 14)]
    blocks = np.split(ww[yy == 0], np.cumsum(nb)[:-1])
    c["equal_negative_clip_weights"] = np.allclose([b.sum() for b in blocks], blocks[0].sum())
    known = {0: [0.0, 0.2646], 1: [0.0021, 0.3848], 6: [0.2109, 0.7891], 12: [0.7354, 1.0]}   # n = 12, 95 %
    c["clopper_pearson_known_values"] = all(W.clopper_pearson(kk, 12) == v for kk, v in known.items())
    g = W.grid(11)
    c["grid_k20_short_and_long"] = len(g) == 20 and g.min() == 0 and g.max() == 10 and len(W.grid(300)) == 20
    ok = all(c.values())
    print(f"mask_weights   {'PASS' if ok else 'FAIL'} " + " ".join(k for k, v in c.items() if not v))
    return ok


def check_leak():
    clips, pays, _ = gen(np.random.default_rng(9), "planted", D=D_MEM)
    U = in_memory(clips, pays)
    i, j = 0, 5
    idx = [k for k in range(len(U)) if k not in (i, j)]
    m0 = W.train(U, idx)[0]
    U2 = list(U)
    lb = dict(U[i]["lab"]); lb["ivs"] = [dict(v, start=v["start"] - 9, end=v["end"] - 9) for v in lb["ivs"]]
    rz = np.random.default_rng(1)
    U2[i] = dict(U[i], lab=lb, P={"t": U[i]["P"]["t"], "z": rz.normal(size=U[i]["P"]["z"].shape)})
    U2[j] = dict(U[j], P={"t": U[j]["P"]["t"], "z": rz.normal(size=U[j]["P"]["z"].shape)})
    m1 = W.train(U2, idx)[0]
    ok = all(np.array_equal(m0[x], m1[x]) for x in ("beta", "med", "sd")) and m0["b0"] == m1["b0"]
    print(f"leakage        {'PASS' if ok else 'FAIL'} heldout_pair_perturbed_model_identical={int(ok)}")
    return ok


def check_print_lock(priv):
    import builtins
    seen, real = [], builtins.open
    builtins.open = lambda f, *a, **k: (seen.append(os.path.abspath(str(f))), real(f, *a, **k))[1]
    try:
        rc, out, _ = call(["--print-lock"])
    finally:
        builtins.open = real
    ok = rc == 0 and "lock_sha256_16=" in out and not any(p.startswith(priv) for p in seen)
    print(f"print_lock     {'PASS' if ok else 'FAIL'} private files opened: {sum(p.startswith(priv) for p in seen)}")
    return ok


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    base = a[0]
    k = int(a[a.index("--reps") + 1]) if "--reps" in a else 10
    kn = int(a[a.index("--reps-null") + 1]) if "--reps-null" in a else 40
    res, priv = cli(base)
    res += check_reps(k, kn) + [check_mask_weights(), check_leak(), check_print_lock(priv)]
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
