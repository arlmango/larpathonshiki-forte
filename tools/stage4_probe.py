"""STAGE 4 (TRAIN caches only): preceding-window candidates, nested clip-LOO, permutation control.

usage: python -m tools.stage4_probe --candidate C02 [--n-perm 1000] [--budget-sec 300]
Implements docs/PROTOCOL_STAGE4.md exactly (incl. the §3 pre-run amendment: inner clip-LOO selection).
Units only via local_safe.select_split_units(side "train"); with all caches present the dry-run form is used
and NO clip file is opened. A missing cache is decoded by the train_probe child (job via stdin, stderr
DEVNULL) for that TRAIN clip only. stdout and <LP>/train_probe/stage4_summary.json carry AGGREGATES ONLY
(no shas, paths, clip names, per-clip rows or per-clip times). Audit files are read-only; their sha256 is
checked before/after and only unchanged/CHANGED is printed. Localization = timestamp-endpoint (±5 s of the
imported label second), NOT event onset/interval and NOT a legal violation.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import train_probe as TP  # noqa: E402
from tools.local_safe import Refused, run_guarded, select_split_units  # noqa: E402

LP = TP.LP
FEATURES = ("n_persons", "n_in_zone", "zone_empty", "n_outside_zone", "n_in_ring", "motion_global", "motion_zone",
            "motion_ring", "person_area_mean", "person_area_max", "zone_box_area", "min_person_dist",
            "zone_to_other_dist", "objects_near")
WINDOWS, DIRS, AGGS, OFFSETS = (5.0, 15.0), (1, -1), ("mean", "max"), (0.0, -2.5)
CFGS = [(f, w, s, a, o) for f in FEATURES for w in WINDOWS for s in DIRS for a in AGGS for o in OFFSETS]
TOL, FPS = 5.0, 4.0
# coordination-recorded hashes (C02 only); anything else: before/after comparison only
EXPECTED = {"C02": {"attrs": "ab7a86b165346141ac3d548752d617b00ca223f43937f25be0106e8d43f4896f",
                    "r1": "09247337a21dc4010d3e47ca9173f368c7ea4c298dab8effc5018e9f7c97ac35",
                    "map": "e413108f8b7f0080"}}
PROTOCOL = os.path.join(ROOT, "docs", "PROTOCOL_STAGE4.md")


def cfg_name(c):
    f, w, s, a, o = CFGS[c]
    return f"{f}|{w:g}s|{'high' if s > 0 else 'low'}|{a}|off{o:g}"


# ============================================================================ scan (label-free)
def scan(t, x, f, W, sign, agg, off):
    """Slide over the WHOLE clip. Window for scan point tau = samples with time in (tau+off-W, tau+off],
    valid iff >= max(2, ceil(0.5*W*f)) samples. -> (clip score = max stat, earliest argmax tau)."""
    v = sign * np.asarray(x, np.float64)
    if not len(t):
        return -np.inf, np.nan
    end = t + off
    hi = np.searchsorted(t, end, side="right")
    lo = np.searchsorted(t, end - W, side="right")
    n = hi - lo
    ok = n >= max(2, math.ceil(0.5 * W * f))
    if not ok.any():
        return -np.inf, np.nan
    if agg == "mean":
        c = np.concatenate([[0.0], np.cumsum(v)])
        st = (c[hi] - c[lo]) / np.maximum(n, 1)
    else:
        L = int(n.max())
        idx = lo[:, None] + np.arange(L)[None]
        st = np.where(idx < hi[:, None], v[np.clip(idx, 0, len(v) - 1)], -np.inf).max(1)
    st = np.where(ok, np.round(st, 9), -np.inf)
    k = int(np.argmax(st))  # first index of the max = earliest tau
    return float(st[k]), float(t[k])


# ============================================================================ nested selection
def fit_thr(s, y):
    """s (C,n), y (n,) -> thr (C,): maximize clip F1 over candidate thresholds = the scores; ties -> higher."""
    P = s[:, None, :] >= s[:, :, None]  # [c, k, i] = s_i >= s_k
    pos = y.astype(bool)
    f1 = 2 * (P & pos).sum(2) / np.maximum(1, P.sum(2) + pos.sum())
    o = np.lexsort((s, f1), axis=-1)[:, -1]
    return s[np.arange(len(s)), o]


def auc_rows(s, y):
    pos, neg = s[:, y == 1], s[:, y == 0]
    if not pos.shape[1] or not neg.shape[1]:
        return np.full(len(s), 0.5)
    d = pos[:, :, None] - neg[:, None, :]
    return (np.nan_to_num(d, nan=0.0) > 0).mean((1, 2)) + 0.5 * ((pos[:, :, None] == neg[:, None, :]).mean((1, 2)))


def hits(T, tl, y):
    """H[c,i] = positive clip i localized within ±TOL of one of its label times (label-free T)."""
    H = np.zeros(T.shape, bool)
    for i in range(T.shape[1]):
        if y[i] and len(tl[i]):
            d = np.min(np.abs(T[:, i][:, None] - np.asarray(tl[i], float)[None]), axis=1)
            H[:, i] = np.nan_to_num(d, nan=np.inf) <= TOL
    return H


def nested(S, T, y, H):
    """Honest nested clip-LOO (protocol §3 amended): for outer clip k, select c* by the AGGREGATED inner
    held-out predictions (inner F1, inner hit rate, AUC, lowest index) of an inner clip-LOO over the 16
    outer-train clips, refit theta on all 16, predict k once. -> pred, t_pred, chosen."""
    C, N = S.shape
    pred, tpred, ch = np.zeros(N, bool), np.full(N, np.nan), np.zeros(N, int)
    for k in range(N):
        tr = np.flatnonzero(np.arange(N) != k)
        ip = np.zeros((C, len(tr)), bool)
        for jj, j in enumerate(tr):
            it = tr[tr != j]
            ip[:, jj] = S[:, j] >= fit_thr(S[:, it], y[it])
        yt = y[tr].astype(bool)
        f1 = 2 * (ip & yt).sum(1) / np.maximum(1, ip.sum(1) + yt.sum())
        hr = H[:, tr][:, yt].mean(1) if yt.any() else np.zeros(C)
        c = int(np.lexsort((-np.arange(C), auc_rows(S[:, tr], y[tr]), hr, f1))[-1])
        thr = fit_thr(S[c:c + 1, tr], y[tr])[0]
        pred[k], tpred[k], ch[k] = S[c, k] >= thr, T[c, k], c
    return pred, tpred, ch


def clip_f1(pred, y):
    tp = int((pred & (y == 1)).sum())
    d = int(pred.sum()) + int((y == 1).sum())
    return 2 * tp / d if d else 0.0


def hit_rate(tpred, tl, y):
    h = [bool(np.isfinite(tpred[i]) and min(abs(tpred[i] - t) for t in tl[i]) <= TOL)
         for i in range(len(y)) if y[i] and len(tl[i])]
    return float(np.mean(h)) if h else float("nan")


def evaluate(S, T, y, tl, dur, n_perm, seed):
    """Primary nested run + permutation null of (outer F1, outer hit rate) under the SAME procedure."""
    pred, tpred, ch = nested(S, T, y, hits(T, tl, y))
    m = TP.metrics(pred, y, tpred, tl)
    obs_f1, obs_hr = clip_f1(pred, y), hit_rate(tpred, tl, y)
    rng = np.random.default_rng(seed)
    nf, nh = [], []
    for _ in range(n_perm):
        pi = rng.permutation(len(y))
        yp = y[pi]
        tlp = [[t / dur[pi[i]] * dur[i] for t in tl[pi[i]]] if yp[i] else [] for i in range(len(y))]
        pp, tp_, _c = nested(S, T, yp, hits(T, tlp, yp))
        nf.append(clip_f1(pp, yp)); nh.append(hit_rate(tp_, tlp, yp))
    nf, nh = np.array(nf), np.array(nh, float)

    def null(v, obs):
        return {"p50": float(np.nanpercentile(v, 50)), "p95": float(np.nanpercentile(v, 95)),
                "n_null_ge_observed": int((v >= obs - 1e-12).sum()),
                "p_perm": float((1 + (v >= obs - 1e-12).sum()) / (1 + len(v)))}
    cnt = collections.Counter(cfg_name(c) for c in ch)
    rho = [spearman(S[c], dur) for c in ch]
    return {"n_units": int(len(y)), "n_pos": int(y.sum()), "outer": m, "outer_clip_f1": obs_f1,
            "outer_hit_rate_timestamp_endpoint": obs_hr,
            "random_time_hit_expectation": random_time(tl, dur, y),
            "always_positive": TP.metrics(np.ones(len(y), bool), y, np.full(len(y), np.nan), tl),
            "always_positive_clip_f1": clip_f1(np.ones(len(y), bool), y),
            "perm": {"n": int(n_perm), "seed": seed, "f1": null(nf, obs_f1), "hit_rate": null(nh, obs_hr)},
            "chosen_configs": {"distinct": len(cnt), "top": cnt.most_common(5),
                               **{d: dict(collections.Counter(str(CFGS[c][k]) for c in ch))
                                  for k, d in enumerate(("feature", "window", "direction", "agg", "offset"))}},
            "chosen_score_vs_duration_abs_spearman_median_max": [float(np.nanmedian(np.abs(rho))),
                                                                 float(np.nanmax(np.abs(rho)))]}


def random_time(tl, dur, y):
    ex, ap = [], []
    for i in range(len(y)):
        if y[i] and len(tl[i]):
            iv = sorted((max(0.0, t - TOL), min(dur[i], t + TOL)) for t in tl[i])
            cov, cur = 0.0, None
            for a, b in iv:
                if cur and a <= cur[1]:
                    cur = (cur[0], max(cur[1], b))
                else:
                    cov += (cur[1] - cur[0]) if cur else 0.0
                    cur = (a, b)
            cov += max(0.0, cur[1] - cur[0])
            ex.append(cov / dur[i]); ap.append(2 * TOL * len(tl[i]) / dur[i])
    return {"clipped_exact": float(np.mean(ex)) if ex else float("nan"),
            "unclipped_10s_over_duration": float(np.mean(ap)) if ap else float("nan")}


def _rank(v):
    v = np.asarray(v, float)
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    r[o] = np.arange(len(v), dtype=float)
    for u in np.unique(v):  # average ties
        m = v == u
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ============================================================================ inputs
def sha256_or_none(p):
    try:
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def audit_hashes(cid):
    return {"attrs": sha256_or_none(os.path.join(LP, "labels", f"{cid}.attrs.jsonl")),
            "r1": sha256_or_none(os.path.join(LP, "labels", f"{cid}.audit_r1.jsonl")),
            "map": sha256_or_none(os.path.join(LP, "inventory_map.json"))}


def hash_status(cid, before, after):
    exp, out = EXPECTED.get(cid, {}), {}
    for k in before:
        b, a = before[k], after[k]
        if b is None and a is None:
            out[k] = "absent"
        elif b != a or (k in exp and not (b or "").startswith(exp[k])):
            out[k] = "CHANGED"
        else:
            out[k] = "unchanged"
    return out


def load_units(cid, budget):
    units, cnt = select_split_units(LP, cid, "train", dry_run=True)  # dry run: no clip file opened
    lab = [(sha, int(st == "reviewed_violation"), ev) for sha, _p, st, ev in units
           if st in ("reviewed_violation", "reviewed_no_violation")]
    cache = os.path.join(LP, "train_probe", "cache")
    os.makedirs(cache, exist_ok=True)
    out = lambda sha: os.path.join(cache, f"{sha}_{FPS:g}.npz")  # noqa: E731
    missing = [sha for sha, _y, _e in lab if not os.path.exists(out(sha))]
    decoded, failed = 0, 0
    if missing:  # decode ONLY missing TRAIN clips (side checked before open, sha re-checked after)
        full, _c = select_split_units(LP, cid, "train")
        paths = {sha: p for sha, p, _s, _e in full}
        for sha in missing:
            if sha in paths and TP.run_child(paths[sha], out(sha), FPS, budget):
                decoded += 1
    U = []
    for sha, y, ev in lab:
        if not os.path.exists(out(sha)):
            failed += 1
            continue
        z = np.load(out(sha))
        r = TP.features(z)
        if r is None:
            failed += 1
            continue
        t, F, cov = r
        d = float(z["dur"])
        U.append({"sha": sha, "y": y, "tl": [float(v) for v in ev], "t": t.astype(np.float64), "dur": d,
                  "f": len(t) / max(d, 1e-6), "F": {k: F[k] for k in FEATURES}, "zone": cov["zone"]})
    cov = {"label_units_selected": len(lab), "skipped_by_selector": cnt["skipped"] + (len(units) - len(lab)),
           "caches_present_at_start": len(lab) - len(missing), "caches_decoded": decoded,
           "excluded_failed_cache_or_zone": failed}
    return U, cov


def zone_groups(cid, U):
    """R1 frame half vs estimated-zone centre. -> (agree mask, counts). Audit file READ ONLY."""
    r1 = {}
    try:
        with open(os.path.join(LP, "labels", f"{cid}.audit_r1.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("kind") == "audit_r1":
                    r1[r.get("sha")] = r.get("r1")
    except OSError:
        pass
    agree = np.zeros(len(U), bool)
    missing = 0
    for i, u in enumerate(U):
        z, a = u["zone"], r1.get(u["sha"])
        cx, cy = (z[0] + z[2]) / 2, (z[1] + z[3]) / 2
        if a is None:
            missing += 1
        agree[i] = {"bottom": cy > 0.5, "top": cy < 0.5, "left": cx < 0.5, "right": cx > 0.5}.get(a, False)
    return agree, {"agree": int(agree.sum()), "mismatch": int(len(U) - agree.sum() - missing), "no_r1": missing}


# ============================================================================ main
def _fmt(v):
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _main(argv=None):
    ap = argparse.ArgumentParser(description="stage 4 train-only probe (aggregates only)")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--budget-sec", type=float, default=300.0)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    proto = sha256_or_none(PROTOCOL)
    h0 = audit_hashes(a.candidate)
    try:
        U, cov = load_units(a.candidate, a.budget_sec)
    except (Refused, KeyError, OSError, ValueError) as e:
        print("refused: " + (str(e) if isinstance(e, Refused) else "inputs missing or malformed"), file=sys.stderr)
        return 2
    if len(U) < 6 or len({u["y"] for u in U}) < 2:
        print(f"refused: too few usable train units ({len(U)})", file=sys.stderr)
        return 2
    y = np.array([u["y"] for u in U])
    tl = [u["tl"] for u in U]
    dur = np.array([u["dur"] for u in U])
    S = np.zeros((len(CFGS), len(U))); T = np.zeros_like(S)
    for c, (f, w, s, ag, o) in enumerate(CFGS):
        for i, u in enumerate(U):
            S[c, i], T[c, i] = scan(u["t"], u["F"][f], u["f"], w, s, ag, o)
    t_scan = time.time() - t0
    lt = [t for u in U if u["y"] for t in u["tl"]]
    pos = [u for u in U if u["y"] and u["tl"]]
    cov.update({"units": len(U), "pos": int(y.sum()), "neg": int((1 - y).sum()), "pos_with_label_time": len(pos),
                "label_times": len(lt),
                "share_label_times_le_duration": float(np.mean([x <= u["dur"] for u in pos for x in u["tl"]]))
                if lt else float("nan"),
                **{f"pos_with_full_preceding_window_{W}s": sum(any(W <= t <= u["dur"] for t in u["tl"]) for u in pos)
                   for W in (5, 15, 30)},
                "duration_sec_min_med_max": [float(dur.min()), float(np.median(dur)), float(dur.max())],
                "samples_per_clip_min_med_max": [int(min(len(u["t"]) for u in U)),
                                                 float(np.median([len(u["t"]) for u in U])),
                                                 int(max(len(u["t"]) for u in U))],
                "n_configs": len(CFGS)})
    fire = {f: [float(np.mean([np.ptp(u["F"][f]) > 0 for u in U])), float(np.mean([(u["F"][f] > 0).mean() for u in U]))]
            for f in FEATURES}
    prim = evaluate(S, T, y, tl, dur, a.n_perm, 0)
    Sd = np.vstack([dur, -dur]); Td = np.full_like(Sd, np.nan)
    pd_, _t, chd = nested(Sd, Td, y, np.zeros(Sd.shape, bool))
    dur_only = TP.metrics(pd_, y, np.full(len(y), np.nan), tl)
    dur_only["sign_chosen_counts"] = {"longer=violation": int((chd == 0).sum()), "shorter=violation": int((chd == 1).sum())}
    conf = {"pointbiserial_r_duration_vs_label": float(np.corrcoef(dur, y)[0, 1]),
            "auc_duration_vs_label": float(auc_rows(dur[None], y)[0])}
    agree, zc = zone_groups(a.candidate, U)
    abl = None
    if agree.sum() >= 6 and len(set(y[agree])) == 2:
        sub = np.flatnonzero(agree)
        abl = evaluate(S[:, sub], T[:, sub], y[sub], [tl[i] for i in sub], dur[sub], a.n_perm, 1)
    p = prim
    crit = {"f1_gt_always_positive": int(p["outer_clip_f1"] > p["always_positive_clip_f1"]),
            "f1_gt_perm_p95": int(p["outer_clip_f1"] > p["perm"]["f1"]["p95"]),
            "hit_gt_random_time": int(np.nan_to_num(p["outer_hit_rate_timestamp_endpoint"]) >
                                      p["random_time_hit_expectation"]["clipped_exact"]),
            "hit_gt_perm_p95": int(np.nan_to_num(p["outer_hit_rate_timestamp_endpoint"]) > p["perm"]["hit_rate"]["p95"])}
    crit["decision_GO"] = int(all(crit.values()))
    h1 = audit_hashes(a.candidate)
    hs = hash_status(a.candidate, h0, h1)
    runtime = time.time() - t0
    res = {"protocol": "PROTOCOL_STAGE4 amended-s3 inner-LOO", "coverage": cov, "firing_nonconst_meanfrac": fire,
           "primary_all_units": prim, "baseline_duration_only_nested": dur_only, "duration_confound": conf,
           "zone_groups": zc, "ABLATION_zone_agree_only_NOT_DEPLOYABLE": abl, "decision": crit,
           "runtime_sec": {"total": runtime, "scan": t_scan}}
    os.makedirs(os.path.join(LP, "train_probe"), exist_ok=True)
    with open(os.path.join(LP, "train_probe", "stage4_summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    _print(res, proto, hs)
    return 0


def _line(name, m):
    c, e = m["clip"], m["event_5s"]
    return (f"{name:40s} clip TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']} P={_fmt(c['precision'])} "
            f"R={_fmt(c['recall'])} F1={_fmt(c['f1'])} | event±5s TP={e['tp']} FP={e['fp']} FN={e['fn']} "
            f"P={_fmt(e['precision'])} R={_fmt(e['recall'])} F1={_fmt(e['f1'])}")


def _block(title, r):
    print(f"\n{title}  (n={r['n_units']}, pos={r['n_pos']})")
    print(_line("nested outer folds", r["outer"]))
    print(_line("always-positive", r["always_positive"]))
    rt = r["random_time_hit_expectation"]
    print(f"timestamp-endpoint hit rate (±5 s, all positives, verdict-independent) = "
          f"{_fmt(r['outer_hit_rate_timestamp_endpoint'])}; random-time expectation clipped={_fmt(rt['clipped_exact'])} "
          f"10s/dur={_fmt(rt['unclipped_10s_over_duration'])}")
    for k in ("f1", "hit_rate"):
        q = r["perm"][k]
        print(f"permutation null ({r['perm']['n']}x full nested) {k:8s}: p50={_fmt(q['p50'])} p95={_fmt(q['p95'])} "
              f"n_null>=obs={q['n_null_ge_observed']} p_perm={_fmt(q['p_perm'])}")
    cc = r["chosen_configs"]
    print(f"configs chosen across outer folds: distinct={cc['distinct']} top={cc['top']}")
    print(f"  by window={cc['window']} direction={cc['direction']} agg={cc['agg']} offset={cc['offset']}")
    print(f"  chosen score vs duration |spearman| median/max = "
          f"{'/'.join(_fmt(v) for v in r['chosen_score_vs_duration_abs_spearman_median_max'])}")


def _print(res, proto, hs):
    print(f"protocol_doc_sha256_16={(proto or 'missing')[:16]}")
    print("COVERAGE " + json.dumps(res["coverage"]))
    print("\nFIRING (frac clips non-constant, mean frac samples > 0)")
    for f, (a, b) in res["firing_nonconst_meanfrac"].items():
        print(f"  {f:20s} {a:6.3f} {b:6.3f}")
    _block("PRIMARY (all units) nested clip-LOO with inner-LOO selection", res["primary_all_units"])
    print(_line("baseline duration-only (nested)", res["baseline_duration_only_nested"]),
          res["baseline_duration_only_nested"]["sign_chosen_counts"])
    print("DURATION CONFOUND " + json.dumps(res["duration_confound"]))
    print("ZONE GROUPS (R1 half vs estimated-zone centre) " + json.dumps(res["zone_groups"]))
    if res["ABLATION_zone_agree_only_NOT_DEPLOYABLE"]:
        _block("ABLATION zone-agree units only (NOT deployable, not used for decision)",
               res["ABLATION_zone_agree_only_NOT_DEPLOYABLE"])
    print("\nDECISION " + json.dumps(res["decision"]) + (" -> GO (one locked holdout eval, after review)"
                                                         if res["decision"]["decision_GO"] else " -> NO-GO"))
    print("AUDIT sha256 " + " ".join(f"{k}={v}" for k, v in hs.items()))
    print(f"runtime_sec total={res['runtime_sec']['total']:.1f} scan={res['runtime_sec']['scan']:.1f}")


def main(argv=None):
    return run_guarded(_main, argv)


if __name__ == "__main__":
    sys.exit(main())
