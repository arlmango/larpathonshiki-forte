"""STAGE 9 (TRAIN only, EXPLORATORY): small L2 logistic head on FROZEN point-level SigLIP2 zone embeddings.

usage: python -m tools.stage9_head --print-lock
       python -m tools.stage9_head --candidate C02 --run --expect-lock <lock16>
Implements docs/PROTOCOL_STAGE9.md exactly (one attempt). No clip file, frame or holdout record is opened: units via
local_safe.select_split_units(side "train", dry_run=True) (through stage8_audit.train_sets); only existing stage-5
embedding caches `<LP>/train_probe/cache_emb_siglip2/<sha16>.npz` of the labelled train positives and the train
imported negatives are read. Unknown-task clips are not loaded at all.
- Features (pre-registered, coordinator constraint): POINT level = one row per decoded frame (1 fps, t = the frame's
  own time), L2-normalised E_zone minus the clip's own per-dimension median (label-free, that clip only). NO temporal
  averaging, so a row's temporal support is its single frame and the mask below is strict by construction.
- Rows: positive = frame time inside a numeric interval shrunk by MARGIN on each uncertain side; weight 1.0 if
  good+sure else 0.5. Negative = frame time inside a confirmed-confident same-clip control and inside no event
  interval. Everything else (incl. unknown tasks, imported negatives) is never a row of the loss.
- Model: L2 logistic regression (IRLS, fixed lambda 10, intercept unpenalised, classes rebalanced to equal total
  weight), scaler (median / robust sd) fitted on the fold's training clips only.
- Split: leave-one-labelled-positive-clip-out; a fold without positive or negative rows is NOT TRAINABLE (reported).
stdout and the summary (<LP>/train_probe/stage9_summary.json, O_EXCL, 0400) carry aggregates only: no shas, paths,
names, per-clip rows, interval or label times.
"""
import argparse
import collections
import json
import os
import resource
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import stage8_audit as S8  # noqa: E402
from tools.annotate_intervals import classify  # noqa: E402
from tools.local_safe import Refused, run_guarded  # noqa: E402

PROTOCOL = os.path.join(ROOT, "docs", "PROTOCOL_STAGE9.md")
LOCKED = (PROTOCOL, os.path.join(ROOT, "tools", "stage9_head.py"), os.path.join(ROOT, "tools", "stage9_selftest.py"))
DEPS = tuple(os.path.join(ROOT, "tools", f) for f in ("stage8_audit.py", "local_safe.py", "annotate_intervals.py"))
LAMBDA = 10.0            # one value, no grid
MARGIN = 5.0             # seconds excluded on an uncertain side
TOL = 5.0                # localisation tolerance (s)
W_GOOD_SURE, W_OTHER = 1.0, 0.5
MIN_STEPS = 3
MAX_CACHE_BYTES = 64 << 20
BUDGET_SEC = 300.0
SUMMARY, STAGE8_SUMMARY = "stage9_summary.json", "stage8_summary.json"
OUTCOME = "EXPLORATORY_ONLY_VALIDATION_UNAVAILABLE"
STOP = "STOP_NOT_TRAINABLE"      # no fold and no resubstitution model has both positive and negative rows
DISCLOSURE = ("stage 9: exploratory fit of a small head; train set reused 5+ times (stages 4, 5, 7, human audit, 8b); "
              "units = clips; at most one held-out positive per fold; no test, interval estimate or threshold is "
              "meaningful; resubstitution numbers are not quality; imported negatives are clip-level, unverified "
              "normal; unknown tasks are not negatives; holdout closed; production stays fail-closed")


def lp():
    return S8.lp()


def lock_digest():
    import hashlib
    h, parts = hashlib.sha256(), {}
    for p in LOCKED + DEPS:
        try:
            d = S8.fsha(p)
        except OSError:
            d = "missing"
        parts[os.path.basename(p)] = d[:16]
        h.update(os.path.basename(p).encode() + b"\0" + d.encode() + b"\n")
    return h.hexdigest()[:16], parts


# ============================================================================ supervision (snapshot -> rows spec)
def supervision(latest, pos, neg):
    """-> (lab {sha: {"ivs": [...], "ctl": [(s, e)], "all": raw intervals}}, counts). Train classes only."""
    g = collections.Counter()
    lab = {}
    for r in latest.values():
        role, p, sha = r.get("role"), r.get("payload") or {}, r.get("sha16")
        if role not in ("positive", "control"):
            continue
        if role == "control":
            g["n_external_control_tasks_not_used" if sha in neg else "n_records_excluded_not_train"] += 1
            continue
        if sha not in pos:
            g["n_records_excluded_not_train"] += 1
            continue
        k = classify("positive", p)
        if k is None:
            g["n_positive_tasks_not_done"] += 1
            continue
        if k == "unknown":
            g["n_unknown_tasks_excluded"] += 1
            continue
        raw = p.get("intervals") or []
        ivs = []
        for iv in raw:
            if not S8.numeric(iv):
                g["n_intervals_excluded_unknown_boundary"] += 1
                continue
            gs = iv.get("visibility") == "good" and iv.get("confidence") == "sure"
            ivs.append({"start": float(iv["start"]), "end": float(iv["end"]), "w": W_GOOD_SURE if gs else W_OTHER,
                        "su": bool(iv.get("start_uncertain")), "eu": bool(iv.get("end_uncertain"))})
            g["n_intervals_weight_1" if gs else "n_intervals_weight_0.5"] += 1
        if not ivs:
            g["n_positive_tasks_without_numeric_interval"] += 1
            continue
        ctl = []
        for c in p.get("same_clip_controls") or []:
            if S8.numeric(c) and any(S8.overlap(e, c) for e in raw if e.get("start") is not None or e.get("end") is not None):
                g["n_controls_excluded_overlap"] += 1
            elif S8.confident_ctl(c):
                ctl.append((float(c["start"]), float(c["end"])))
            else:
                g["n_controls_excluded_low_or_unconfirmed"] += 1
        g["n_controls_used"] += len(ctl)
        lab[sha] = {"ivs": ivs, "ctl": ctl, "all": raw}
    return lab, g


# ============================================================================ point-level features + rows
def point_prep(t, E):
    """frozen point-level features: L2-normalised rows minus the clip's own per-dimension median (label-free)."""
    E = np.asarray(E, np.float64)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    return {"t": np.asarray(t, np.float64), "z": E - np.median(E, 0)}


def in_any(t, spans):
    m = np.zeros(len(t), bool)
    for a, b in spans:
        m |= (t >= a) & (t <= b)
    return m


def rows(P, lb):
    """-> (pos mask, pos weights, neg mask, n rows excluded as both). Support of a row = its frame time only."""
    t = P["t"]
    wp = np.zeros(len(t))
    for iv in lb["ivs"]:
        a, b = iv["start"] + (MARGIN if iv["su"] else 0.0), iv["end"] - (MARGIN if iv["eu"] else 0.0)
        m = (t >= a) & (t <= b)
        wp[m] = np.maximum(wp[m], iv["w"])
    ev = [(e["start"] if e.get("start") is not None else 0.0, e["end"] if e.get("end") is not None else np.inf)
          for e in lb["all"]]
    ng = in_any(t, lb["ctl"]) & ~in_any(t, ev)
    both = (wp > 0) & ng
    return (wp > 0) & ~both, np.where(both, 0.0, wp), ng & ~both, int(both.sum())


def scaler(Zs):
    X = np.vstack(Zs)
    med = np.median(X, 0)
    mad = 1.4826 * np.median(np.abs(X - med), 0)
    return med, np.maximum(np.maximum(mad, 0.1 * X.std(0)), 1e-6)


def fit_logreg(X, y, w, lam=LAMBDA, iters=100, tol=1e-10):
    """L2 logistic regression by IRLS with an unpenalised intercept. Exact reduction to the row space of X
    (representer theorem: the penalised optimum lies in span of the rows). -> (beta (d,), b0, converged)."""
    X, y, w = np.asarray(X, float), np.asarray(y, float), np.asarray(w, float)
    _u, s, Vt = np.linalg.svd(X, full_matrices=False)
    keep = s > 1e-10 * max(1.0, s.max() if len(s) else 1.0)
    V = Vt[keep].T
    A = np.c_[np.ones(len(X)), X @ V]
    R = lam * np.eye(A.shape[1]); R[0, 0] = 0.0
    b = np.zeros(A.shape[1])

    def obj(b):
        f = A @ b
        return float(np.sum(w * (np.logaddexp(0.0, f) - y * f)) + 0.5 * b @ R @ b)
    conv, cur = False, obj(b)
    for _ in range(iters):
        f = A @ b
        p = 1.0 / (1.0 + np.exp(-f))
        g = A.T @ (w * (p - y)) + R @ b
        H = (A.T * (w * p * (1 - p))) @ A + R + 1e-12 * np.eye(len(b))
        step, lr = np.linalg.solve(H, g), 1.0
        while lr > 1e-6 and obj(b - lr * step) > cur + 1e-12:
            lr /= 2
        b = b - lr * step
        new = obj(b)
        if abs(cur - new) < tol * max(1.0, abs(cur)) or np.max(np.abs(lr * step)) < tol:
            conv, cur = True, new
            break
        cur = new
    return V @ b[1:], float(b[0]), conv


def train_model(L, idx):
    """fit on labelled clips `idx` (cache present). -> (model dict) or (None, reason). Rows/scaler from idx only."""
    Xs, ys, ws, Zs = [], [], [], []
    npos = nneg = nboth = 0
    for i in idx:
        P = L[i]["P"]
        pm, wp, nm, both = rows(P, L[i]["lab"])
        nboth += both
        Zs.append(P["z"])
        Xs += [P["z"][pm], P["z"][nm]]
        ys += [np.ones(pm.sum()), np.zeros(nm.sum())]
        ws += [wp[pm], np.ones(nm.sum())]
        npos += int(pm.sum()); nneg += int(nm.sum())
    info = {"n_pos_rows": npos, "n_neg_rows": nneg, "n_rows_excluded_both": nboth}
    if not idx or npos == 0:
        return None, "no_positive_rows_under_strict_mask", info
    if nneg == 0:
        return None, "no_negative_supervision", info
    med, sd = scaler(Zs)
    X = (np.vstack(Xs) - med) / sd
    y, w = np.concatenate(ys), np.concatenate(ws)
    wsum, wpos = w.sum(), w[y == 1].sum()
    w = np.where(y == 1, w * wsum / (2 * wpos), w * wsum / (2 * (wsum - wpos)))   # equal total weight per class
    beta, b0, conv = fit_logreg(X, y, w)
    return {"beta": beta, "b0": b0, "med": med, "sd": sd, "converged": conv}, None, info


def score(m, P):
    return ((P["z"] - m["med"]) / m["sd"]) @ m["beta"] + m["b0"]


def spans(lb, pad=0.0):
    return [(iv["start"] - pad, iv["end"] + pad) for iv in lb["ivs"]]


def hit_at(tp, lb):
    return bool(np.isfinite(tp) and in_any(np.array([tp]), spans(lb, TOL))[0])


def argmax_t(s, t):
    return float(t[int(np.argmax(s))]) if len(s) else float("nan")       # earliest argmax


def random_expectation(lb, dur):
    cov, e = 0.0, -np.inf
    for a, b in sorted((max(0.0, a), min(dur, b)) for a, b in spans(lb, TOL)):
        a = max(a, e); cov += max(0.0, b - a); e = max(e, b)
    return cov / dur if dur > 0 else float("nan")


def interval_percentile(s, P, lb):
    """share of rest frames (outside every interval +- TOL) scoring below the mean score inside the intervals."""
    t = P["t"]
    ins, rest = in_any(t, spans(lb)), ~in_any(t, spans(lb, TOL))
    if not ins.any() or not rest.any():
        return float("nan")
    m = s[ins].mean()
    return float(((s[rest] < m).sum() + 0.5 * (s[rest] == m).sum()) / rest.sum())


def prior_time(L, train, k):
    """frozen stage-7 relative rule; tl := midpoint of each training positive's first numeric interval."""
    rel = [(L[i]["lab"]["ivs"][0]["start"] + L[i]["lab"]["ivs"][0]["end"]) / 2 / L[i]["dur"] for i in train]
    return float(np.median(rel)) * L[k]["dur"] if rel else float("nan")


# ============================================================================ analysis (pure, no I/O)
def _q(v):
    v = [x for x in v if np.isfinite(x)]
    return [round(float(np.min(v)), 4), round(float(np.median(v)), 4), round(float(np.max(v)), 4)] if v else None


def analyse(L, N, deadline=None):
    """L: labelled positives [{"P": point_prep or None, "dur", "lab"}]; N: imported negatives [P or None]."""
    ok = [i for i in range(len(L)) if L[i]["P"] is not None]
    folds, pat, reasons = [], collections.Counter(), collections.Counter()
    for k in range(len(L)):
        if deadline and time.time() > deadline:
            raise Refused("time budget exceeded")
        if L[k]["P"] is None:
            reasons["test_cache_missing"] += 1
            continue
        train = [i for i in ok if i != k]
        m, why, info = train_model(L, train)
        ph = hit_at(prior_time(L, [i for i in range(len(L)) if i != k], k), L[k]["lab"])
        f = {"prior_hit": ph, "random": random_expectation(L[k]["lab"], L[k]["dur"]), **info}
        if m is None:
            reasons[why] += 1
            pat["not_trainable"] += 1
        else:
            s = score(m, L[k]["P"])
            h = hit_at(argmax_t(s, L[k]["P"]["t"]), L[k]["lab"])
            f.update(head_hit=h, pct=interval_percentile(s, L[k]["P"], L[k]["lab"]), conv=m["converged"])
            pat[("head_hit" if h else "head_miss") + "_" + ("prior_hit" if ph else "prior_miss")] += 1
        folds.append(f)
    tr = [f for f in folds if "head_hit" in f]
    out = {"n_labelled_positive_clips": len(L), "n_labelled_with_cache": len(ok), "n_folds": len(L),
           "n_folds_trainable": len(tr), "not_trainable_reasons": dict(reasons), "fold_patterns": dict(pat),
           "head_hits_trainable_folds": sum(f["head_hit"] for f in tr),
           "prior_hits_trainable_folds": sum(f["prior_hit"] for f in tr),
           "prior_hits_all_evaluable_folds": [sum(f["prior_hit"] for f in folds), len(folds)],
           "random_expectation_mean_trainable": round(float(np.mean([f["random"] for f in tr])), 3) if tr else None,
           "interval_mean_percentile_trainable": sorted(round(f["pct"], 3) for f in tr if np.isfinite(f["pct"])),
           "train_rows_per_trainable_fold_pos_neg": sorted([f["n_pos_rows"], f["n_neg_rows"]] for f in tr),
           "train_rows_per_untrainable_fold_pos_neg": sorted([f["n_pos_rows"], f["n_neg_rows"]] for f in folds
                                                             if "head_hit" not in f),
           "irls_converged_all": all(f["conv"] for f in tr)}
    # ---- resubstitution (all labelled rows) + FP screen on imported negatives: NOT validation
    m, why, info = train_model(L, ok)
    rs = {"label": "RESUBSTITUTION_ONLY_NOT_VALIDATION", "rows_pos_neg": [info["n_pos_rows"], info["n_neg_rows"]],
          "n_rows_excluded_both": info["n_rows_excluded_both"]}
    fp = {"label": "imported negatives are clip-level, unverified normal; screen only", "n_negatives": len(N),
          "n_scored": 0}
    if m is None:
        rs.update(trainable=False, reason=why)
    else:
        S = {i: score(m, L[i]["P"]) for i in ok}
        rs.update(trainable=True, converged=m["converged"],
                  hits=[sum(hit_at(argmax_t(S[i], L[i]["P"]["t"]), L[i]["lab"]) for i in ok), len(ok)],
                  interval_mean_percentile=sorted(round(interval_percentile(S[i], L[i]["P"], L[i]["lab"]), 3)
                                                  for i in ok))
        imax = [float(S[i][in_any(L[i]["P"]["t"], spans(L[i]["lab"]))].max()) for i in ok
                if in_any(L[i]["P"]["t"], spans(L[i]["lab"])).any()]
        nmax = [float(score(m, P).max()) for P in N if P is not None]
        fp.update(n_scored=len(nmax), negative_clip_max_min_med_max=_q(nmax),
                  positive_interval_max_min_med_max=_q(imax),
                  n_negative_clipmax_ge_min_positive_interval_max=int(sum(v >= min(imax) for v in nmax)) if imax else None)
    out["resubstitution"] = rs
    out["fp_screen"] = fp
    return out


# ============================================================================ I/O (train caches only)
def load_emb(sha):
    """-> point_prep dict + dur, or None (missing / oversized / invalid = failed unit, never a negative)."""
    path = os.path.join(lp(), "train_probe", "cache_emb_siglip2", f"{sha}.npz")
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > MAX_CACHE_BYTES:
            return None
        with np.load(path, allow_pickle=False) as ze:
            t, E, dur = ze["t"].astype(np.float64), ze["E_zone"].astype(np.float64), float(ze["dur_dec"])
    except (OSError, ValueError, KeyError):
        return None
    if (E.ndim != 2 or len(t) != len(E) or len(t) < MIN_STEPS or not np.isfinite(E).all() or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0) or not dur > 0):
        return None
    return point_prep(t, E), dur


def compute(cid, rel, sha, deadline=None):
    path, sha = S8.snapshot_spec(cid, rel, sha)
    latest = S8.read_snapshot(path, sha)
    pos, neg, cnt = S8.train_sets(cid)
    lab, g = supervision(latest, pos, neg)
    L, miss = [], collections.Counter()
    for s in sorted(lab):
        r = load_emb(s)
        miss["labelled_cache_missing_or_invalid"] += r is None
        L.append({"P": r[0] if r else None, "dur": r[1] if r else None, "lab": lab[s]})
    N = []
    for s in sorted(neg):
        r = load_emb(s)
        miss["negative_cache_missing_or_invalid"] += r is None
        N.append(r[0] if r else None)
    res = analyse(L, N, deadline)
    res["supervision_counts"] = {k: int(v) for k, v in sorted(g.items())}
    res["coverage"] = {"n_train_positive_units": len(pos), "n_train_negative_units": len(neg),
                       "selector_skipped": int(cnt["skipped"]), **{k: int(v) for k, v in miss.items()}}
    res["validation"] = (f"validation unavailable: {len(L)} labelled positive clip(s), "
                         f"{sum(bool(x['lab']['ctl']) for x in L)} clip(s) with a confirmed control, "
                         f"{res['n_folds_trainable']} trainable fold(s) with one held-out positive each; "
                         "no inference, interval estimate or threshold is meaningful")
    trainable = res["n_folds_trainable"] > 0 or res["resubstitution"].get("trainable")
    res["outcome"] = OUTCOME if trainable else STOP
    if not trainable:
        res["stop_reason"] = sorted(set(res["not_trainable_reasons"]) | {res["resubstitution"].get("reason", "")} - {""})
    return res


# ============================================================================ CLI
def _stage8_closed():
    p = os.path.join(lp(), "train_probe", STAGE8_SUMMARY)
    try:
        with open(p, encoding="utf-8") as f:
            o = json.load(f).get("outcome")
    except (OSError, ValueError, AttributeError):
        raise Refused("stage 8 not closed: gate summary missing or malformed") from None
    if o not in ("STOP_INSUFFICIENT_DATA", "GATE_PASS_SEPARATE_PROTOCOL_REQUIRED"):
        raise Refused("stage 8 not closed: unexpected gate outcome")
    return S8.fsha(p)


def _main(argv=None):
    ap = argparse.ArgumentParser(description="stage 9 exploratory head (aggregates only)")
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--print-lock", action="store_true")
    m.add_argument("--run", action="store_true")
    ap.add_argument("--candidate", default="")
    ap.add_argument("--expect-lock", default="")
    ap.add_argument("--snapshot", default=None, help="relative to the private dir (non-pinned candidates only)")
    ap.add_argument("--snapshot-sha256", default=None)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    lock, parts = lock_digest()
    print(f"lock_sha256_16={lock} files={json.dumps(parts, sort_keys=True)}")
    if a.print_lock:
        return 0
    if a.expect_lock != lock:
        raise Refused("lock digest mismatch: protocol or code changed since the lock")
    out = os.path.join(lp(), "train_probe", SUMMARY)
    if os.path.lexists(out):
        raise Refused("summary already exists: single pre-registered run")
    spath, ssha = S8.snapshot_spec(a.candidate, a.snapshot, a.snapshot_sha256)
    g0 = _stage8_closed()
    h0, s0 = S8.immutables(a.candidate), (S8.fsha(spath) if os.path.exists(spath) else "absent")
    r = compute(a.candidate, a.snapshot, a.snapshot_sha256, deadline=t0 + BUDGET_SEC)
    h1, s1, g1 = S8.immutables(a.candidate), (S8.fsha(spath) if os.path.exists(spath) else "absent"), _stage8_closed()
    r.update(protocol="PROTOCOL_STAGE9", lock_sha256_16=lock, locked_files_sha256_16=parts,
             runtime_sec=round(time.time() - t0, 2), disclosure=DISCLOSURE,
             maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1),
             fixed={"lambda": LAMBDA, "margin_sec": MARGIN, "tol_sec": TOL, "features": "point_level_no_window",
                    "weights_good_sure_other": [W_GOOD_SURE, W_OTHER]},
             snapshot={"sha256_verified": s0 == ssha, "unchanged": s0 == s1},
             stage8_summary_unchanged=g0 == g1,
             immutable_files={"n": h0[0], "match_recorded_sha16": h1[1],
                              "status": "unchanged" if h0[2] == h1[2] else "CHANGED"})
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "w") as f:
        json.dump(r, f, indent=1, sort_keys=True, default=float)
    print("STAGE9 " + json.dumps({k: v for k, v in r.items() if k not in ("locked_files_sha256_16", "disclosure")},
                                 sort_keys=True, default=float))
    print(f"OUTCOME={r['outcome']}; {r['validation']}")
    print(f"snapshot verified={int(s0 == ssha)} unchanged={int(s0 == s1)}; stage8 summary unchanged={int(g0 == g1)}; "
          f"immutables n={h0[0]} match_recorded={h1[1]} {r['immutable_files']['status']}; summary written (0400)")
    return 0


def main(argv=None):
    return run_guarded(_main, argv)


if __name__ == "__main__":
    sys.exit(main())
