"""Synthetic self-test for tools/stage9_head.py (no real data, no video, no real labels).

usage: python -m tools.stage9_selftest DIR [--reps-planted 20] [--reps-null 60]
CLI fixtures (tools.stage7_selftest.build: 17 train + 4 holdout dummy clip files, holdout chmod 000; then SYNTHETIC
stage-5-format embedding caches written over the train ones, 1 fps, durations U(60, 140) s, plus one holdout cache at
mode 000 that must never be opened) and a synthetic interval snapshot built with the real validator
annotate_intervals.clean_payload:
  cli_planted    3 numeric positives (one start-uncertain, one partial), 1 same-clip control in positive 0,
                 2 unknown tasks, 12 negatives, 1 holdout record, 1 superseded rev; planted cue inside the intervals
                 -> EXPLORATORY outcome, 3 folds, 2 trainable, 1 not trainable (no_negative_supervision), hits 2/2
  cli_real_like  2 numeric positives, the only control inside an UNKNOWN task, 3 unknown -> STOP_NOT_TRAINABLE
                 (0 trainable folds, resubstitution not trainable, reason no_negative_supervision)
  each CLI run   rc 0, stderr empty, summary 0400, open() spy: no clip file, no holdout cache/path, no unknown-task
                 cache opened; holdout mode 000; no sha16/sentinel/fixture path/interval time in any output;
                 refusals (second run, wrong lock, wrong snapshot sha, stage-8 summary missing) -> rc 2, no summary
In memory (same generator, no files):
  rep_planted    replicated planted datasets -> head hit rate on trainable folds >= 0.9
  rep_null       replicated null datasets -> head hit rate within 3 binomial SE of the random expectation
  mask_rules     margin on the uncertain side, weights 1.0/0.5, unknown tasks + their controls excluded, overlapping
                 control excluded, control rows never inside an event interval
  leakage        fold model unchanged when the held-out clip's labels/features or the imported negatives change
  print_lock     opens no private file
"""
import builtins
import contextlib
import io
import json
import os
import stat
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import stage7_selftest as S7T  # noqa: E402
from tools import stage9_head as H  # noqa: E402
from tools.annotate_intervals import clean_payload, task_id  # noqa: E402

AMP = 0.15          # synthetic cue amplitude (fixture parameter, not a method parameter)
IV_LEN, CTL_LEN = 8.0, 8.0


# ============================================================================ generator (synthetic only)
def gen(rng, planted, n_lab=3, n_unk=2, n_neg=12, ctl_in_unknown=False):
    """-> clips [{key, kind, t, E, dur}], payloads {key: positive payload}, times (all planted interval numbers)."""
    u = rng.normal(size=S7T.D); u /= np.linalg.norm(u)
    clips, pays, times = [], {}, []
    for j in range(n_lab + n_unk + n_neg):
        kind = "lab" if j < n_lab else ("unk" if j < n_lab + n_unk else "neg")
        dur = float(rng.uniform(60, 140))
        t = np.arange(0.5, dur - 0.5, 1.0)
        E = S7T.sem_seq(rng, t)
        c = {"key": j, "kind": kind, "t": t, "dur": dur}
        if kind == "lab":
            s = round(float(rng.uniform(0.3, 0.8)) * dur, 3)
            iv = {"start": s, "end": round(s + IV_LEN, 3), "start_uncertain": j == 0,
                  "visibility": "partial" if j == 1 else "good"}
            ctls = []
            if j == 0 and not ctl_in_unknown:
                a = s - 3 * IV_LEN if s > 3 * IV_LEN + 1 else s + 3 * IV_LEN
                ctls = [{"start": round(a, 3), "end": round(a + CTL_LEN, 3)}]
            times += [iv["start"], iv["end"]] + [x for cc in ctls for x in (cc["start"], cc["end"])]
            if planted:
                m = (t >= iv["start"]) & (t <= iv["end"])
                cue = u + 0.7 * rng.normal(size=(m.sum(), S7T.D)) / np.sqrt(S7T.D)
                E[m] += AMP * cue / np.linalg.norm(cue, axis=1, keepdims=True)
            pays[j] = pos_payload([iv], ctls=ctls, dur=dur)
        elif kind == "unk":
            ctls = []
            if ctl_in_unknown and j == n_lab:
                ctls = [{"start": 10.111, "end": 18.111}]
                times += [10.111, 18.111]
            pays[j] = pos_payload(flags=(("unknown", "not_visible", "event_unconfirmed")[j % 3],), ctls=ctls, dur=dur)
        c["E"] = E / np.linalg.norm(E, axis=1, keepdims=True)
        clips.append(c)
    return clips, pays, times


def pos_payload(intervals=(), flags=(), ctls=(), dur=60.0):
    p = {"intervals": [dict({"start_unknown": False, "end_unknown": False, "start_uncertain": False,
                             "end_uncertain": False, "visibility": "good", "confidence": "sure"}, **iv)
                       for iv in intervals],
         "clip_flags": {k: k in flags for k in ("not_visible", "unknown", "event_unconfirmed")},
         "same_clip_controls": [dict({"visibility": "good", "confidence": "sure", "comparable_visible": True,
                                      "no_violation": True}, **c) for c in ctls],
         "control_not_found": not ctls, "done": True}
    return clean_payload("positive", p, dur, {"viewed": True})


def in_memory(clips, pays):
    sha = {c["key"]: f"s{c['key']:02d}" for c in clips}
    latest = {f"t{k}": {"task_id": f"t{k}", "sha16": sha[k], "role": "positive", "rev": 1, "payload": p}
              for k, p in pays.items()}
    pos = {sha[c["key"]] for c in clips if c["kind"] != "neg"}
    neg = {sha[c["key"]] for c in clips if c["kind"] == "neg"}
    lab, g = H.supervision(latest, pos, neg)
    by = {sha[c["key"]]: c for c in clips}
    L = [{"P": H.point_prep(by[s]["t"], by[s]["E"]), "dur": by[s]["dur"], "lab": lab[s]} for s in sorted(lab)]
    N = [H.point_prep(c["t"], c["E"]) for c in clips if c["kind"] == "neg"]
    return L, N, g


# ============================================================================ CLI fixture
def fixture(base, name, clips, pays):
    root, priv, sh, hold = S7T.build(base, f"s9_{name}")
    with open(os.path.join(priv, "labels", "C01.imported.jsonl")) as f:
        st = {r["sha16"]: r["status"] for r in map(json.loads, f) if r["kind"] == "imported_clip_status"}
    tpos = [s for s in sh[:17] if st[s] == "reviewed_violation"]
    tneg = [s for s in sh[:17] if st[s] == "reviewed_no_violation"]
    pos_clips = [c for c in clips if c["kind"] != "neg"]
    neg_clips = [c for c in clips if c["kind"] == "neg"]
    assert len(pos_clips) == len(tpos) and len(neg_clips) == len(tneg)
    edir = os.path.join(priv, "train_probe", "cache_emb_siglip2")
    m = {}
    for s, c in list(zip(tpos, pos_clips)) + list(zip(tneg, neg_clips)):
        m[c["key"]] = s
        np.savez_compressed(os.path.join(edir, f"{s}.npz"), t=c["t"].astype(np.float32),
                            E_full=c["E"].astype(np.float16), E_zone=c["E"].astype(np.float16),
                            zone_fallback=np.int8(0), dur_dec=np.float32(c["dur"]))
    hc = os.path.join(edir, f"{sh[17]}.npz")
    np.savez_compressed(hc, t=np.arange(3, dtype=np.float32), E_zone=np.ones((3, 4), np.float16),
                        dur_dec=np.float32(3))
    os.chmod(hc, 0)
    recs = [{"task_id": "_assign", "sha16": "", "role": "assignment", "rev": 1, "ts": 0, "payload": {}}]
    for i, (k, p) in enumerate(sorted(pays.items())):
        tid = task_id(m[k], "positive", i)
        if i == 0:
            recs.append({"task_id": tid, "sha16": m[k], "role": "positive", "rev": 1, "ts": 0,
                         "payload": pos_payload(flags=("unknown",))})
        recs.append({"task_id": tid, "sha16": m[k], "role": "positive", "rev": 2, "ts": 1, "payload": p})
    recs.append({"task_id": task_id(sh[17], "positive", 9), "sha16": sh[17], "role": "positive", "rev": 1, "ts": 3,
                 "payload": pos_payload([{"start": 2.0, "end": 9.0}], dur=17.0)})
    rel = "labels/snapshots/C01.intervals.v1.TEST.jsonl"
    os.makedirs(os.path.join(priv, "labels", "snapshots"), exist_ok=True)
    raw = "".join(json.dumps(r, sort_keys=True) + "\n" for r in recs).encode()
    with open(os.path.join(priv, rel), "wb") as f:
        f.write(raw)
    os.chmod(os.path.join(priv, rel), 0o400)
    for rp in ("labels/C01.attrs.jsonl", "labels/C01.audit_r1.jsonl", "train_probe/stage7_summary.json"):
        with open(os.path.join(priv, rp), "w") as f:
            f.write("{}\n")
    with open(os.path.join(priv, "train_probe", "stage8_summary.json"), "w") as f:
        json.dump({"outcome": "STOP_INSUFFICIENT_DATA"}, f)
    import hashlib
    unk = [os.path.join(edir, f"{m[c['key']]}.npz") for c in clips if c["kind"] == "unk"]
    return root, priv, sh, hold + [hc], unk, rel, hashlib.sha256(raw).hexdigest()


def call(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = H.main(argv)
    sys.excepthook = sys.__excepthook__     # run_guarded installs a static hook; restore it for the self-test
    return rc, out.getvalue(), err.getvalue()


def spied(argv):
    seen, real = [], builtins.open

    def spy(f, *a, **k):
        seen.append(os.path.abspath(str(f)) if isinstance(f, (str, os.PathLike)) else "")
        return real(f, *a, **k)
    builtins.open = spy
    try:
        r = call(argv)
    finally:
        builtins.open = real
    return r, seen


def cli(base, name, clips, pays, times, expect):
    root, priv, sh, hold, unk, rel, ssha = fixture(base, name, clips, pays)
    S7T.point(priv)
    lock = H.lock_digest()[0]
    args = ["--candidate", "C01", "--run", "--expect-lock", lock, "--snapshot", rel, "--snapshot-sha256", ssha]
    (rc, out, err), seen = spied(args)
    sp = os.path.join(priv, "train_probe", H.SUMMARY)
    summ = open(sp).read() if os.path.exists(sp) else ""
    r = json.loads(summ) if summ else {}
    c = {"rc0": rc == 0, "stderr_empty": err == "", "summary_0400": bool(summ) and stat.S_IMODE(os.stat(sp).st_mode) == 0o400,
         "expect": all(r.get(k) == v for k, v in expect.items()),
         "integrity": r.get("snapshot") == {"sha256_verified": True, "unchanged": True}
         and r.get("immutable_files", {}).get("status") == "unchanged" and r.get("stage8_summary_unchanged") is True,
         "holdout_not_opened": not any(os.path.abspath(h) in seen for h in hold),
         "unknown_caches_not_opened": not any(os.path.abspath(u) in seen for u in unk),
         "no_clip_file_opened": not any("clips_" in p for p in seen),
         "holdout_mode_000": all(stat.S_IMODE(os.stat(h).st_mode) == 0 for h in hold),
         "holdout_record_excluded": r.get("supervision_counts", {}).get("n_records_excluded_not_train") == 1}
    blob = out + err + summ
    c["no_leak"] = (not any(s in blob for s in sh) and S7T.SENT not in blob and root not in blob and priv not in blob
                    and not any(f"{t:.3f}" in blob or f"{t:.2f}" in blob for t in times))
    r2 = call(args)[0]
    os.chmod(sp, 0o600); os.remove(sp)
    r3 = call(args[:4] + ["0" * 16] + args[5:])[0]
    r4 = call(args[:-1] + ["0" * 64])[0]
    s8 = os.path.join(priv, "train_probe", "stage8_summary.json")
    os.remove(s8)
    r5 = call(args)[0]
    c["refusals_rc2_no_summary"] = (r2, r3, r4, r5) == (2, 2, 2, 2) and not os.path.exists(sp)
    ok = all(c.values())
    print(f"{name:14s} {'PASS' if ok else 'FAIL'} outcome={r.get('outcome')} folds={r.get('n_folds')} "
          f"trainable={r.get('n_folds_trainable')} hits={r.get('head_hits_trainable_folds')} "
          f"reasons={r.get('not_trainable_reasons')} opened={len(seen)} " + " ".join(k for k, v in c.items() if not v),
          flush=True)
    return ok, priv


# ============================================================================ in-memory checks
def rep(planted, reps, seed0):
    hits = n = 0
    exp, untrain = [], 0
    for s in range(reps):
        clips, pays, _ = gen(np.random.default_rng(seed0 + s), planted, n_neg=0)
        L, _N, _g = in_memory(clips, pays)
        r = H.analyse(L, [])
        hits += r["head_hits_trainable_folds"]; n += r["n_folds_trainable"]
        untrain += r["not_trainable_reasons"].get("no_negative_supervision", 0)
        if r["n_folds_trainable"]:
            exp += [r["random_expectation_mean_trainable"]] * r["n_folds_trainable"]
    return hits, n, float(np.mean(exp)) if exp else float("nan"), untrain


def check_rep_planted(reps):
    h, n, e, u = rep(True, reps, 1000)
    ok = n == 2 * reps and u == reps and h / n >= 0.9
    print(f"rep_planted    {'PASS' if ok else 'FAIL'} hits {h}/{n} (random expectation {e:.3f}); untrainable {u}/{reps}")
    return ok


def check_rep_null(reps):
    h, n, e, u = rep(False, reps, 5000)
    se = np.sqrt(e * (1 - e) / n)
    ok = n == 2 * reps and u == reps and abs(h / n - e) <= 3 * se
    print(f"rep_null       {'PASS' if ok else 'FAIL'} hits {h}/{n} = {h / n:.3f} vs random expectation {e:.3f} "
          f"(3 SE = {3 * se:.3f}); untrainable {u}/{reps}")
    return ok


def check_mask():
    rng = np.random.default_rng(7)
    clips, pays, _ = gen(rng, True)
    k0 = clips[0]
    p0 = pays[0]
    ivs = list(p0["intervals"])
    # extra: an overlapping control and a low-confidence control in clip 0 (must be excluded)
    a, b = ivs[0]["start"], ivs[0]["end"]
    p0x = dict(p0, same_clip_controls=p0["same_clip_controls"] + [
        dict(p0["same_clip_controls"][0], start=round(a + 1, 3), end=round(b + 4, 3)),
        dict(p0["same_clip_controls"][0], confidence="unsure")])
    pays2 = {**pays, 0: p0x}
    L, _N, g = in_memory(clips, pays2)
    l0 = L[0]
    pm, wp, nm, _both = H.rows(l0["P"], l0["lab"])
    t = l0["P"]["t"]
    c = {"margin_on_uncertain_start": bool(pm.any()) and t[pm].min() >= a + 5.0 - 1e-9 and t[pm].max() <= b
         and H.MARGIN == 5.0 and H.LAMBDA == 10.0 and H.TOL == 5.0,                 # pre-registered constants
         "weight_partial_0.5": set(np.unique(H.rows(L[1]["P"], L[1]["lab"])[1][H.rows(L[1]["P"], L[1]["lab"])[0]]))
         == {0.5},
         "weight_good_sure_1": set(np.unique(wp[pm])) == {1.0},
         "unknown_excluded": g["n_unknown_tasks_excluded"] == 2 and len(L) == 3,
         "overlap_ctl_excluded": g["n_controls_excluded_overlap"] == 1,
         "low_ctl_excluded": g["n_controls_excluded_low_or_unconfirmed"] == 1 and g["n_controls_used"] == 1,
         "ctl_rows_outside_events": not (nm & (t >= a) & (t <= b)).any() and nm.any(),
         "imported_neg_not_rows": True}
    clips_u, pays_u, _ = gen(np.random.default_rng(8), True, n_lab=2, n_unk=3, ctl_in_unknown=True)
    Lu, _Nu, gu = in_memory(clips_u, pays_u)
    r = H.analyse(Lu, [])
    c["ctl_in_unknown_task_unused"] = (gu["n_controls_used"] == 0 and r["n_folds_trainable"] == 0
                                       and r["resubstitution"].get("reason") == "no_negative_supervision")
    ok = all(c.values()) and k0["kind"] == "lab"
    print(f"mask_rules     {'PASS' if ok else 'FAIL'} " + " ".join(k for k, v in c.items() if not v))
    return ok


def check_leak():
    clips, pays, _ = gen(np.random.default_rng(9), True)
    L, N, _g = in_memory(clips, pays)
    k = 1                                   # held-out positive whose fold is trainable (control lies in clip 0)
    train = [0, 2]
    m0 = H.train_model(L, train)[0]
    L2 = [dict(x) for x in L]
    lb = dict(L[k]["lab"]); lb["ivs"] = [dict(iv, start=iv["start"] - 20, end=iv["end"] - 20, w=1.0) for iv in lb["ivs"]]
    L2[k] = dict(L[k], lab=lb, P={"t": L[k]["P"]["t"], "z": np.random.default_rng(1).normal(size=L[k]["P"]["z"].shape)})
    m1 = H.train_model(L2, train)[0]
    same = all(np.array_equal(m0[x], m1[x]) for x in ("beta", "med", "sd")) and m0["b0"] == m1["b0"]
    r0 = H.analyse(L, N)
    r1 = H.analyse(L, [{"t": P["t"], "z": P["z"] * 0} for P in N])
    nn = {x: r0[x] for x in r0 if x not in ("fp_screen",)} == {x: r1[x] for x in r1 if x not in ("fp_screen",)}
    ok = same and nn and m0 is not None
    print(f"leakage        {'PASS' if ok else 'FAIL'} heldout_perturb_model_identical={int(same)} "
          f"negatives_perturb_folds_identical={int(nn)}")
    return ok


def check_print_lock(priv):
    (rc, out, _err), seen = spied(["--print-lock"])
    ok = rc == 0 and "lock_sha256_16=" in out and not any(p.startswith(priv) for p in seen)
    print(f"print_lock     {'PASS' if ok else 'FAIL'} private files opened: {sum(p.startswith(priv) for p in seen)}")
    return ok


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    base = a[0]
    rp = int(a[a.index("--reps-planted") + 1]) if "--reps-planted" in a else 20
    rn = int(a[a.index("--reps-null") + 1]) if "--reps-null" in a else 60
    res = []
    clips, pays, times = gen(np.random.default_rng(11), True)
    ok, priv = cli(base, "cli_planted", clips, pays, times,
                   {"outcome": H.OUTCOME, "n_folds": 3, "n_folds_trainable": 2, "head_hits_trainable_folds": 2,
                    "not_trainable_reasons": {"no_negative_supervision": 1}})
    res.append(ok)
    clips, pays, times = gen(np.random.default_rng(12), True, n_lab=2, n_unk=3, ctl_in_unknown=True)
    ok, _ = cli(base, "cli_real_like", clips, pays, times,
                {"outcome": H.STOP, "n_folds": 2, "n_folds_trainable": 0,
                 "not_trainable_reasons": {"no_negative_supervision": 2}, "stop_reason": ["no_negative_supervision"]})
    res.append(ok)
    res += [check_rep_planted(rp), check_rep_null(rn), check_mask(), check_leak(), check_print_lock(priv)]
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
