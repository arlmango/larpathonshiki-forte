"""TRAIN-ONLY feature probe: which observable per-sample signals relate to the label timestamps?

usage: python -m tools.train_probe --candidate C02 [--fps 4] [--budget-sec 300] [--force]

HOLDOUT IS FROZEN: a clip file is opened only after an explicit check that its sha16 is on the
train side and its index is not claimed by any non-train label record; its sha is re-checked after
opening. Extraction runs per clip in a CHILD process (job via stdin, never argv; stderr DEVNULL;
SIGKILL at --budget-sec), caching numeric timelines (boxes/classes/motion grid, no images) under
<LP>/train_probe/cache/. stdout and <LP>/train_probe/summary.json carry ONLY aggregates.
No clip names, paths, shas or per-clip rows are ever printed or written to the summary.
"""
import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:  # script form `python tools/train_probe.py` (the -m form already has it)
    sys.path.insert(0, ROOT)
LP = os.environ.get("LOCAL_PRIVATE_DIR") or os.path.join(ROOT, "local_private")
# COCO ids (YOLOX order): person, backpack, handbag, suitcase, bottle, cup, chair, tv, laptop,
# mouse, keyboard, cell phone, book
CLASSES = (0, 24, 26, 28, 39, 41, 56, 62, 63, 64, 66, 67, 73)
GROUPS = {"bag_near": (24, 26, 28), "phone_near": (67,), "bottle_cup_near": (39, 41), "book_near": (73,),
          "electronics_near": (62, 63, 64, 66)}
TOL = 5.0


from tools.local_safe import Refused  # noqa: E402


# ============================================================================ child: extraction
def child_stdin():
    """Child entry: job arrives as ONE JSON line on stdin (never argv: argv is visible via ps)."""
    job = json.loads(sys.stdin.readline())
    return child(job["path"], job["out"], float(job["fps"]))


def child(path, out, fps):
    import cv2
    from app import detector as D, pipeline
    rules = json.load(open(pipeline.DEFAULT_RULESET))
    gw, gh = rules["motion_grid"]
    with D.quiet_native_stderr():
        det = D.load("")
        pr = pipeline.probe(path, 10 ** 6)
        cap = cv2.VideoCapture(path)
        step = max(1, int(round(pr.fps / fps)))
        ts, grids, rows, prev, idx = [], [], [], None, 0
        while cap.grab():
            if idx % step == 0:
                ok, fr = cap.retrieve()
                if not ok:
                    break
                small = pipeline._resize_w(fr, 640)
                h, w = small.shape[:2]
                if hasattr(det, "detect_classes"):
                    d = det.detect_classes(small, CLASSES)
                else:  # synthetic self-test detector: persons only (class 0)
                    p = det.detect(small)
                    d = np.hstack([p, np.zeros((len(p), 1), np.float32)])
                g = cv2.GaussianBlur(cv2.cvtColor(pipeline._resize_w(small, 320), cv2.COLOR_BGR2GRAY), (5, 5), 0)
                grid = np.zeros((gh, gw), np.float32) if prev is None else cv2.resize(
                    (cv2.absdiff(g, prev) > rules["motion_pixel_thresh"]).astype(np.float32), (gw, gh),
                    interpolation=cv2.INTER_AREA)
                prev = g
                for r in d:
                    rows.append([len(ts), r[0] / w, r[1] / h, r[2] / w, r[3] / h, r[4], r[5]])
                ts.append(idx / pr.fps)
                grids.append(grid.astype(np.float16))
            idx += 1
        cap.release()
    np.savez_compressed(out, t=np.array(ts, np.float32), grid=np.array(grids),
                        dets=np.array(rows, np.float32).reshape(-1, 7), dur=np.float32(pr.duration))
    return 0


def run_child(path, out, fps, budget):
    tmp = out + ".part.npz"
    p = subprocess.Popen([sys.executable, "-m", "tools.train_probe", "--child"], cwd=ROOT,
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True, close_fds=True)
    with contextlib.suppress(OSError):
        p.stdin.write((json.dumps({"path": path, "out": tmp, "fps": fps}) + "\n").encode())
        p.stdin.close()
    with contextlib.suppress(subprocess.TimeoutExpired):
        p.wait(timeout=budget)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    if p.returncode == 0 and os.path.exists(tmp):
        os.replace(tmp, out)
        return True
    with contextlib.suppress(OSError):
        os.remove(tmp)
    return False


# ============================================================================ train selection
def train_units(cid):
    """-> ([(sha, path, label, [t..])], n_skipped) for TRAIN shas only, via the shared fail-closed
    selector (side + index checked BEFORE opening; holdout records discarded; sha re-checked after
    opening; unreadable or mismatching files counted, never named)."""
    from tools.local_safe import select_split_units
    units, c = select_split_units(LP, cid, "train")
    out = [(sha, path, int(st == "reviewed_violation"), ev) for sha, path, st, ev in units
           if st in ("reviewed_violation", "reviewed_no_violation")]
    return out, c["skipped"] + c["unreadable"] + c["index_mismatch"] + (len(units) - len(out))


# ============================================================================ features
def features(npz):
    from app import detector as D, pipeline
    rules = json.load(open(pipeline.DEFAULT_RULESET))
    t, grid, dets = npz["t"], npz["grid"].astype(np.float32), npz["dets"]
    S = len(t)
    per = [dets[dets[:, 0] == i] for i in range(S)]
    persons = [p[p[:, 6] == 0][:, 1:6] for p in per]
    zone, support = D.estimate_zone([{"boxes": b} for b in persons], rules)
    if zone is None:
        return None
    ring = D.pad(zone, (rules["zone"]["activity_ring_scale"] - 1) / 2)
    gh, gw = grid.shape[1:]
    zm = D.cell_mask(zone, gw, gh)
    rm = D.cell_mask(ring, gw, gh) & ~zm
    F = {k: np.zeros(S, np.float32) for k in (
        "n_persons", "n_in_zone", "zone_empty", "n_outside_zone", "n_in_ring", "motion_global", "motion_zone",
        "motion_ring", "person_area_mean", "person_area_max", "zone_box_area", "zone_box_aspect",
        "zone_person_score", "min_person_dist", "zone_to_other_dist", "chair_in_zone", "objects_near",
        *GROUPS)}
    for i in range(S):
        b = persons[i]
        inz = D.inside(b, zone) if len(b) else np.zeros(0, bool)
        inr = (D.inside(b, ring) & ~inz) if len(b) else np.zeros(0, bool)
        F["n_persons"][i], F["n_in_zone"][i] = len(b), inz.sum()
        F["zone_empty"][i] = float(inz.sum() == 0)
        F["n_outside_zone"][i], F["n_in_ring"][i] = (~inz).sum(), inr.sum()
        F["motion_global"][i] = grid[i].mean()
        F["motion_zone"][i] = grid[i][zm].mean()
        F["motion_ring"][i] = grid[i][rm].mean() if rm.any() else 0
        if len(b):
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            F["person_area_mean"][i], F["person_area_max"][i] = area.mean(), area.max()
            cx, cy = D.centers(b)
            if len(b) > 1:
                dd = np.hypot(cx[:, None] - cx[None], cy[:, None] - cy[None]) + np.eye(len(b)) * 9
                F["min_person_dist"][i] = dd.min()
            else:
                F["min_person_dist"][i] = 1.5
            if inz.any():
                j = np.argmax(np.where(inz, area, -1))
                bw, bh = b[j, 2] - b[j, 0], b[j, 3] - b[j, 1]
                F["zone_box_area"][i], F["zone_box_aspect"][i] = area[j], bw / max(bh, 1e-6)
                F["zone_person_score"][i] = b[j, 4]
                others = np.hypot(cx - cx[j], cy - cy[j])[np.arange(len(b)) != j]
                F["zone_to_other_dist"][i] = others.min() if len(others) else 1.5
            else:
                F["zone_to_other_dist"][i] = 1.5
        else:
            F["min_person_dist"][i] = F["zone_to_other_dist"][i] = 1.5
        o = per[i][per[i][:, 6] != 0]
        if len(o):
            near = D.inside(o[:, 1:6], ring)
            F["objects_near"][i] = near.sum()
            for k, cl in GROUPS.items():
                F[k][i] = (near & np.isin(o[:, 6], cl)).sum()
            F["chair_in_zone"][i] = (D.inside(o[:, 1:6], zone) & (o[:, 6] == 56)).sum()
    cov = {"samples": S, "zone_support_ratio": (support or 0) / max(S, 1),
           "zone_occupancy": float((F["n_in_zone"] > 0).mean()),
           "zone": [float(v) for v in zone]}  # in-memory only (stage4 R1 ablation); never written to summary
    return t, F, cov


# ============================================================================ statistics
def auc(pos, neg):  # Mann-Whitney: P(pos > neg) + 0.5 P(tie) over all pairs (clip-level vectors)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    scale = max(1.0, abs(a.mean()), abs(b.mean()))
    return float((a.mean() - b.mean()) / s) if s > 1e-6 * scale else float("nan")  # degenerate variance


def rolling(x, n):
    n = max(1, min(n, len(x)))
    c = np.concatenate([[0], np.cumsum(x, dtype=float)])
    return (c[n:] - c[:-n]) / n  # value k covers samples k..k+n-1


def clip_score(t, x, med, cfg, fps):
    tr, sign, win = cfg
    v = (x - med) if tr == "dev" else x
    r = sign * rolling(v, int(round(win * fps)))
    k = int(r.argmax())
    n = max(1, min(int(round(win * fps)), len(x)))
    return float(r[k]), float(t[min(len(t) - 1, k + n // 2)])


def fit_threshold(s, y):
    best = (-1.0, np.inf)
    for thr in np.unique(s):
        p = s >= thr
        tp = (p & (y == 1)).sum()
        f1 = 2 * tp / max(1, p.sum() + (y == 1).sum())
        if f1 > best[0] or (f1 == best[0] and thr > best[1]):
            best = (f1, thr)
    return best[1], best[0]


def metrics(pred, y, tpred, tlab, n_failed=0):
    """Clip and event metrics with tools/evaluate.py semantics (match_events, ±5 s, all events,
    one predicted event per predicted-violation clip). Extraction failures are counted
    separately (never as no_violation)."""
    from tools.evaluate import _prf, match_events
    c = dict(tp=0, fp=0, fn=0, tn=0)
    e = dict(tp=0, fp=0, fn=0)
    for i in range(len(y)):
        pt = [float(tpred[i])] if pred[i] and np.isfinite(tpred[i]) else []
        c["tp" if y[i] and pred[i] else "fp" if pred[i] else "fn" if y[i] else "tn"] += 1
        m = match_events(list(tlab[i]) if y[i] else [], pt, TOL)
        e["tp"] += len(m); e["fp"] += len(pt) - len(m); e["fn"] += (len(tlab[i]) if y[i] else 0) - len(m)
    return {"clip": {**c, **_prf(c["tp"], c["fp"], c["fn"])}, "event_5s": {**e, **_prf(e["tp"], e["fp"], e["fn"])},
            "extraction_failed_excluded": n_failed}


# ============================================================================ analysis
def _paired(units, f, W, times=None):
    """Per positive clip: mean of f within ±W s of its label time(s) vs mean of the rest of the
    SAME clip. -> (n_clips, paired AUC = P(win > rest) + 0.5 P(tie), dz, win means)."""
    diffs, wins = [], []
    for k, u in enumerate(units):
        tl = u["tl"] if times is None else [times[k]]
        m = np.zeros(len(u["t"]), bool)
        for t0 in tl:
            m |= np.abs(u["t"] - t0) <= W
        if m.any() and (~m).any():
            wins.append(float(u["F"][f][m].mean()))
            diffs.append(wins[-1] - float(u["F"][f][~m].mean()))
    d = np.array(diffs)
    if not len(d):
        return 0, float("nan"), float("nan"), []
    pa = float((d > 0).mean() + 0.5 * (d == 0).mean())
    dz = float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else 0.0
    return len(d), pa, dz, wins


def analyse(units, rng, n_failed=0):
    """All statistics are CLIP-level (unit = distinct sha16); temporal samples are never pooled."""
    feats = list(units[0]["F"])
    pos = [u for u in units if u["y"] == 1 and u["tl"]]
    neg = [u for u in units if u["y"] == 0]
    A = []
    for f in feats:
        row = {"feature": f}
        for W in (5, 15):
            n, pa, dz, wins = _paired(pos, f, W)
            negm = [float(u["F"][f].mean()) for u in neg]
            row.update({f"n_clips_w{W}": n, f"paired_auc_win_vs_rest_w{W}": pa, f"dz_w{W}": dz,
                        f"auc_poswin_vs_negclip_mean_w{W}": auc(wins, negm),
                        f"d_poswin_vs_negclip_mean_w{W}": cohen_d(wins, negm)})
        null = []
        for _ in range(200):  # same statistic at random pseudo-label times in the same clips
            _n, pa, _dz, _w = _paired(pos, f, 5, [rng.uniform(0, u["t"][-1]) for u in pos])
            null.append(max(pa, 1 - pa))
        row["null95_paired_auc_w5_random_times"] = float(np.nanpercentile(null, 95))
        A.append(row)
    # clip-level: violation clip vs negative clip, sliding-window max of feature (raw / deviation)
    cfgs = [(f, tr, s, w) for f in feats for tr in ("raw", "dev") for s in (1, -1) for w in (4, 10)]
    y = np.array([u["y"] for u in units])
    Sc = np.zeros((len(cfgs), len(units))); Tc = np.zeros_like(Sc)
    for ci, (f, tr, s, w) in enumerate(cfgs):
        for i, u in enumerate(units):
            Sc[ci, i], Tc[ci, i] = clip_score(u["t"], u["F"][f], np.median(u["F"][f]), (tr, s, w), u["fps"])
    aucs = np.array([auc(Sc[c, y == 1], Sc[c, y == 0]) for c in range(len(cfgs))])
    tl = [u["tl"] for u in units]
    loc = np.array([np.mean([min(abs(Tc[c, i] - t) for t in tl[i]) <= TOL for i in range(len(units))
                             if y[i] == 1 and tl[i]] or [np.nan]) for c in range(len(cfgs))])
    perm = []
    for _ in range(200):
        yp = rng.permutation(y)
        perm.append(np.nanmax([auc(Sc[c, yp == 1], Sc[c, yp == 0]) for c in range(len(cfgs))]))
    B = []
    for c in np.argsort(-np.nan_to_num(aucs))[:15]:
        f, tr, s, w = cfgs[c]
        B.append({"feature": f, "transform": tr, "direction": "high" if s > 0 else "low", "window_sec": w,
                  "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()), "auc_clip": float(aucs[c]),
                  "d_clip": cohen_d(Sc[c, y == 1], Sc[c, y == 0]), "peak_within_5s_of_label": float(loc[c])})
    # Task B: LOO (each unit = one distinct sha16) for top-3 configs, plus fully nested LOO
    def loo(ci_fixed=None):
        pred = np.zeros(len(units), bool); tp_ = np.zeros(len(units)); chosen = []
        for i in range(len(units)):
            tr_ = np.arange(len(units)) != i
            if ci_fixed is None:
                best = max(range(len(cfgs)), key=lambda c: (fit_threshold(Sc[c, tr_], y[tr_])[1],
                                                            auc(Sc[c, tr_ & (y == 1)], Sc[c, tr_ & (y == 0)])))
            else:
                best = ci_fixed
            thr, _ = fit_threshold(Sc[best, tr_], y[tr_])
            pred[i], tp_[i] = Sc[best, i] >= thr, Tc[best, i]
            chosen.append(best)
        m = metrics(pred, y, tp_, tl, n_failed)
        m["distinct_configs_chosen"] = len(set(chosen))
        return m
    top3 = [int(c) for c in np.argsort(-np.nan_to_num(aucs))[:3]]
    P = []
    for c in top3:
        f, tr, s, w = cfgs[c]
        P.append({"pattern": f"{f}|{tr}|{'high' if s > 0 else 'low'}|{w}s", "auc_clip_train": float(aucs[c]),
                  **loo(c)})
    nested = loo(None)
    base = metrics(np.ones(len(units), bool), y, np.full(len(units), np.nan), tl, n_failed)
    fire = [{"feature": f, "n_clips": len(units),
             "frac_clips_ever_gt0": float(np.mean([(u["F"][f] > 0).any() for u in units])),
             "frac_clips_nonconstant": float(np.mean([np.ptp(u["F"][f]) > 0 for u in units])),
             "frac_pos_clips_ever_gt0": float(np.mean([(u["F"][f] > 0).any() for u in units if u["y"] == 1])),
             "frac_neg_clips_ever_gt0": float(np.mean([(u["F"][f] > 0).any() for u in units if u["y"] == 0])),
             "mean_frac_samples_gt0": float(np.mean([(u["F"][f] > 0).mean() for u in units]))} for f in feats]
    return {"firing_rate": fire, "window_vs_rest_clip_paired": A, "clip_level_top15_OPTIMISTIC_EXPLORATORY": B,
            "clip_level_null": {"n_configs": len(cfgs), "perm_max_auc_p50": float(np.percentile(perm, 50)),
                                "perm_max_auc_p95": float(np.percentile(perm, 95))},
            "loo_top3_config_chosen_on_full_train_OPTIMISTIC_EXPLORATORY": P,
            "TRAIN_ESTIMATE_nested_loo_select_feature_window_direction_threshold": nested,
            "baseline_always_violation": base}


def _fmt(v):
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main(argv=None):
    from tools.local_safe import run_guarded
    return run_guarded(_main, argv)


def _main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--child"]:
        return child_stdin()
    ap = argparse.ArgumentParser(description="train-only feature probe (aggregates only)")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--budget-sec", type=float, default=300.0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    try:
        units, skipped = train_units(a.candidate)
    except (Refused, KeyError, OSError, ValueError) as e:
        print("refused: " + (str(e) if isinstance(e, Refused) else "inputs missing or malformed"), file=sys.stderr)
        return 2
    cache = os.path.join(LP, "train_probe", "cache")
    os.makedirs(cache, exist_ok=True)
    runtimes, failed, U = [], 0, []
    load0 = os.getloadavg()
    for sha, path, y, tl in units:
        out = os.path.join(cache, f"{sha}_{a.fps:g}.npz")
        if a.force or not os.path.exists(out):
            t0 = time.time()
            ok = run_child(path, out, a.fps, a.budget_sec)
            runtimes.append(time.time() - t0)
            if not ok:
                failed += 1
                continue
        z = np.load(out)
        r = features(z)
        if r is None:
            failed += 1
            continue
        t, F, cov = r
        U.append({"y": y, "tl": tl, "t": t, "F": F, "cov": cov, "fps": len(t) / max(float(z["dur"]), 1e-6)})
    load1 = os.getloadavg()
    if len(U) < 4 or len({u["y"] for u in U}) < 2:
        print(f"refused: too few usable train units ({len(U)})", file=sys.stderr)
        return 2
    res = analyse(U, np.random.default_rng(0), failed)
    npers = np.concatenate([u["F"]["n_persons"] for u in U])
    cov = {"train_units": len(U), "violation_units": sum(u["y"] for u in U),
           "violation_units_with_timestamp": sum(1 for u in U if u["y"] and u["tl"]),
           "negative_units": sum(1 - u["y"] for u in U), "extraction_failed": failed, "label_units_skipped": skipped,
           "samples_per_clip_min_med_max": [int(min(u["cov"]["samples"] for u in U)),
                                            float(np.median([u["cov"]["samples"] for u in U])),
                                            int(max(u["cov"]["samples"] for u in U))],
           "zone_support_ratio_median": float(np.median([u["cov"]["zone_support_ratio"] for u in U])),
           "zone_occupancy_min_med_max": [float(min(u["cov"]["zone_occupancy"] for u in U)),
                                          float(np.median([u["cov"]["zone_occupancy"] for u in U])),
                                          float(max(u["cov"]["zone_occupancy"] for u in U))],
           "frac_samples_0_1_2plus_persons": [float((npers == 0).mean()), float((npers == 1).mean()),
                                              float((npers >= 2).mean())],
           "runtime_sec_p50_p95_max": [float(np.percentile(runtimes, q)) for q in (50, 95, 100)] if runtimes else None,
           "loadavg_1m_before_after": [round(load0[0], 2), round(load1[0], 2)]}
    res["coverage"] = cov
    os.makedirs(os.path.join(LP, "train_probe"), exist_ok=True)
    with open(os.path.join(LP, "train_probe", "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("COVERAGE " + json.dumps(cov))
    print("\nA0  FIRING RATE per feature (n = train clips): ever>0 / non-constant / ever>0 in pos / in neg / "
          "mean fraction of samples > 0")
    for r in res["firing_rate"]:
        print(f"{r['feature']:22s} {r['n_clips']:3d} {r['frac_clips_ever_gt0']:6.3f} {r['frac_clips_nonconstant']:6.3f} "
              f"{r['frac_pos_clips_ever_gt0']:6.3f} {r['frac_neg_clips_ever_gt0']:6.3f} {r['mean_frac_samples_gt0']:6.3f}")
    print("\nA1  WINDOW vs REST, paired per violation clip (n = clips); pAUC = P(window mean > rest mean); "
          "null95 = 95th pct at random times")
    print(f"{'feature':22s} {'n':>3s} {'pAUC5':>6s} {'pAUC15':>6s} {'dz5':>6s} {'null95':>6s} {'AUCneg5':>7s} {'dneg5':>6s}")
    for r in sorted(res["window_vs_rest_clip_paired"],
                    key=lambda r: -abs(np.nan_to_num(r["paired_auc_win_vs_rest_w5"], nan=0.5) - 0.5)):
        print(f"{r['feature']:22s} {r['n_clips_w5']:3d} {_fmt(r['paired_auc_win_vs_rest_w5']):>6s} "
              f"{_fmt(r['paired_auc_win_vs_rest_w15']):>6s} {_fmt(r['dz_w5']):>6s} "
              f"{_fmt(r['null95_paired_auc_w5_random_times']):>6s} {_fmt(r['auc_poswin_vs_negclip_mean_w5']):>7s} "
              f"{_fmt(r['d_poswin_vs_negclip_mean_w5']):>6s}")
    nl = res["clip_level_null"]
    print(f"\nA2  CLIP LEVEL violation vs negative clip (sliding-window max), OPTIMISTIC EXPLORATORY (chosen on full "
          f"train), top 15 of {nl['n_configs']} configs; label-permutation max-AUC p50={nl['perm_max_auc_p50']:.3f} "
          f"p95={nl['perm_max_auc_p95']:.3f}")
    for r in res["clip_level_top15_OPTIMISTIC_EXPLORATORY"]:
        print(f"{r['feature']:22s} {r['transform']:3s} {r['direction']:4s} {r['window_sec']:>2d}s "
              f"pos={r['n_pos']} neg={r['n_neg']} AUC={r['auc_clip']:.3f} d={_fmt(r['d_clip'])} "
              f"peak±5s={_fmt(r['peak_within_5s_of_label'])}")

    def line(name, m):
        c, e = m["clip"], m["event_5s"]
        return (f"{name:46s} clip TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']} P={_fmt(c['precision'])} "
                f"R={_fmt(c['recall'])} F1={_fmt(c['f1'])} | event±5s TP={e['tp']} FP={e['fp']} FN={e['fn']} "
                f"P={_fmt(e['precision'])} R={_fmt(e['recall'])}")
    print("\nB1  OPTIMISTIC EXPLORATORY: config chosen on full train, threshold refit per leave-one-clip-out fold")
    for r in res["loo_top3_config_chosen_on_full_train_OPTIMISTIC_EXPLORATORY"]:
        print(line(r["pattern"], r))
    print("\nB2  TRAIN ESTIMATE: nested leave-one-clip-out (feature, transform, direction, window and threshold "
          "all chosen inside each fold)")
    nm = res["TRAIN_ESTIMATE_nested_loo_select_feature_window_direction_threshold"]
    print(line(f"nested (distinct configs chosen: {nm['distinct_configs_chosen']})", nm))
    print(line("reference: always predict violation", res["baseline_always_violation"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
