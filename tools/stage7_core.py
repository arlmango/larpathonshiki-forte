"""STAGE 7 numeric core (docs/PROTOCOL_STAGE7.md §2-§7): within-clip contrast against unlabeled background.

Pure numpy, no I/O. Everything learned (direction, robust scale, threshold) is fitted on training clips only;
the robust scale is label-free and cached by training set. Used by tools/stage7_wcr.py and tools/stage7_selftest.py.
"""
import time

import numpy as np

from tools import stage4_probe as S4, train_probe as TP

W, MINV, TOL = 5.0, 3, 5.0
A_SPAN, GUARD, MIN_B = (-10.0, 2.0), (-15.0, 7.0), 3
ALPHA = 0.05 / 2                                   # Bonferroni over the 2 pre-registered families
N1_DRAWS = 10000
MOT_NAMES = [f"zone_sub{a}{b}" for b in range(3) for a in range(3)] + [
    "motion_ring", "motion_global", "n_in_zone", "n_in_ring", "zone_to_other_dist", "person_area_max", "phone_near"]


# ============================================================================ label-free per-clip features
def windows(t):
    hi = np.arange(1, len(t) + 1)
    lo = np.searchsorted(t, t - W, side="right")
    return lo, hi, (hi - lo) >= MINV


def wmean(x, lo, hi):
    c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, 0)])
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)[:, None]


def prep(t, x):
    """clip-median centring, then trailing 5 s window mean. -> dict(t, z, ok)."""
    x = np.asarray(x, np.float64)
    c = x - np.median(x, 0)
    lo, hi, ok = windows(t)
    return {"t": np.asarray(t, np.float64), "z": wmean(c, lo, hi), "ok": ok}


def mot_timeline(npz):
    """stage-4 cache -> (t at 1 s bins k+0.5, 16 features) or None if the label-free zone is unavailable."""
    r = TP.features(npz)
    if r is None:
        return None
    t4, F, cov = r
    x0, y0, x1, y1 = cov["zone"]
    grid = npz["grid"].astype(np.float32)
    gh, gw = grid.shape[1:]
    cx, cy = (np.arange(gw) + 0.5) / gw, (np.arange(gh) + 0.5) / gh
    cols = []
    for b in range(3):
        for a in range(3):
            mx = (cx >= x0 + a * (x1 - x0) / 3) & (cx < x0 + (a + 1) * (x1 - x0) / 3)
            my = (cy >= y0 + b * (y1 - y0) / 3) & (cy < y0 + (b + 1) * (y1 - y0) / 3)
            m = my[:, None] & mx[None, :]
            cols.append(grid[:, m].mean(1) if m.any() else np.zeros(len(t4)))
    for k in MOT_NAMES[9:]:
        cols.append(F[k] if k != "phone_near" else F["phone_near"])
    X = np.stack(cols, 1).astype(np.float64)
    k = np.floor(np.asarray(t4, np.float64)).astype(int)
    tb, xb = [], []
    for b in np.unique(k):
        m = k == b
        tb.append(b + 0.5); xb.append(X[m].mean(0))
    return np.array(tb), np.array(xb)


# ============================================================================ WCR core (pure numpy)
class Family:
    """Per-clip prepared sequences. The per-dimension scale is ROBUST and label-free: sd_d = max(1.4826 MAD_d,
    0.1 SD_d, 1e-6) over all valid steps of the training clips, cached by training set (labels never enter)."""

    def __init__(self, preps):
        self.P = preps
        self.Z = [p["z"][p["ok"]] for p in preps]
        self._sd = {}

    def scale(self, excl):
        key = frozenset(excl)
        if key not in self._sd:
            X = np.vstack([self.Z[m] for m in range(len(self.P)) if m not in key])
            mad = 1.4826 * np.median(np.abs(X - np.median(X, 0)), 0)
            self._sd[key] = np.maximum(np.maximum(mad, 0.1 * X.std(0)), 1e-6)
        return self._sd[key]

    def delta(self, i, tl):
        """within-clip contrast of positive clip i: mean z over A minus mean z over B (guard band excluded)."""
        p = self.P[i]
        t, z, ok = p["t"], p["z"], p["ok"]
        a = np.zeros(len(t), bool)
        b = ok.copy()
        for x in tl:
            a |= (t >= x + A_SPAN[0]) & (t <= x + A_SPAN[1])
            b &= (t < x + GUARD[0]) | (t > x + GUARD[1])
        a &= ok
        if not a.any():
            return None
        return z[a].mean(0) - (z[b].mean(0) if b.sum() >= MIN_B else 0.0)

    def fit(self, excl, D):
        """unit-norm direction from all clips except `excl` (D: within-clip deltas of positives). -> (w, sd)."""
        sd = self.scale(excl)
        ds = [D[i] for i in range(len(self.P)) if i not in excl and i in D]
        w = (np.mean(ds, 0) if ds else np.zeros_like(sd)) / sd
        nw = np.linalg.norm(w)
        return (w / nw if nw > 0 else w), sd

    def seq(self, i, w, sd):
        p = self.P[i]
        return np.where(p["ok"], (p["z"] / sd) @ w, -np.inf)


def peak(s, t):
    if not np.isfinite(s).any():
        return -np.inf, np.nan
    k = int(np.argmax(s))                                   # earliest argmax
    return float(s[k]), float(t[k])


def urank(s, t, ok, tau):
    """within-clip percentile of s at the valid step nearest tau (ties half)."""
    v = np.flatnonzero(ok)
    if len(v) < 2:
        return np.nan
    j = v[np.argmin(np.abs(t[v] - tau))]
    sv = s[v]
    return float(((sv < s[j]).sum() + 0.5 * ((sv == s[j]).sum() - 1)) / (len(v) - 1))


def fit_thr_mid(s, y):
    """F1-max threshold placed at the MIDPOINT between adjacent training scores (candidates: midpoints of the sorted
    distinct scores and min-1); ties -> higher threshold. Avoids a zero-margin cut at a training score."""
    s = np.asarray(s, float); yv = np.asarray(y).astype(bool)
    u = np.unique(s[np.isfinite(s)])
    if not len(u):
        return np.inf
    cand = np.r_[u[0] - 1.0, (u[1:] + u[:-1]) / 2]
    best, thr = -1.0, cand[0]
    for c in cand:
        p = s >= c
        f1 = 2 * (p & yv).sum() / max(1, p.sum() + yv.sum())
        if f1 >= best - 1e-12:
            best, thr = f1, c
    return float(thr)


def hit_of(tp, tls):
    return bool(np.isfinite(tp) and len(tls) and min(abs(tp - x) for x in tls) <= TOL)


def nested(Fm, y, tl, dur, keep_pairs=False):
    """Protocol §4: outer clip-LOO; direction w_{-k}; inner held-out scores with w_{-{k,j}}; F1-max threshold.
    -> dict(pred, tpred, score, rho, pairs)."""
    N = len(Fm.P)
    D = {}
    for i in range(N):
        if y[i] and len(tl[i]):
            d = Fm.delta(i, tl[i])
            if d is not None:
                D[i] = d
    yv = np.asarray(y)
    pred, tp, so, rho = np.zeros(N, bool), np.full(N, np.nan), np.full(N, -np.inf), np.full(N, np.nan)
    rho_neg = np.full(N, np.nan)  # DIAGNOSTIC ONLY (protocol §3): never read by any criterion/threshold/selection
    pairs, outer = {}, {}
    for k in range(N):
        tr = [m for m in range(N) if m != k]
        sin = []
        for j in tr:
            key = frozenset((k, j))
            if key not in pairs:
                pairs[key] = Fm.fit(key, D)
            sin.append(peak(Fm.seq(j, *pairs[key]), Fm.P[j]["t"])[0])
        sin = np.array(sin)
        thr = fit_thr_mid(sin, yv[tr])
        outer[k] = Fm.fit({k}, D)
        so[k], tp[k] = peak(Fm.seq(k, *outer[k]), Fm.P[k]["t"])
        pred[k] = so[k] >= thr
        rho[k] = S4.spearman(np.nan_to_num(sin, neginf=-1e9), [dur[m] for m in tr])
        ng = [i for i, m in enumerate(tr) if not yv[m]]
        rho_neg[k] = S4.spearman(np.nan_to_num(sin[ng], neginf=-1e9), [dur[tr[i]] for i in ng]) if len(ng) > 2 else np.nan
    return {"pred": pred, "tpred": tp, "score": so, "rho": rho, "rho_neg": rho_neg, "pairs": pairs if keep_pairs else None,
            "outer": outer if keep_pairs else None}


def stats(r, y, tl):
    yv = np.asarray(y).astype(bool)
    pred, tp = r["pred"], r["tpred"]
    h = np.array([bool(yv[i]) and hit_of(tp[i], tl[i]) for i in range(len(yv))])
    hr = float(h[yv].mean()) if yv.any() else float("nan")
    tpe = int((pred & h).sum())
    nlab = sum(len(tl[i]) for i in range(len(yv)) if yv[i])
    ef1 = 2 * tpe / (2 * tpe + (int(pred.sum()) - tpe) + (nlab - tpe)) if (pred.sum() + nlab) else 0.0
    mae = float(np.mean([min(abs(tp[i] - x) for x in tl[i]) if np.isfinite(tp[i]) else np.inf
                         for i in range(len(yv)) if yv[i] and len(tl[i])]))
    return S4.clip_f1(pred, yv.astype(int)), hr, float(ef1), mae


def permute(y, tl, dur, rng):
    pi = rng.permutation(len(y))
    yp = np.asarray(y)[pi]
    return yp, [[t / dur[pi[i]] * dur[i] for t in tl[pi[i]]] if yp[i] else [] for i in range(len(y))]


def transplant(Fm, r, y, tl, dur, rng):
    """N1 (protocol §6): positives' within-clip label rank vs negatives scored by the SAME fold's pair model
    w_{-{k,j}} at the transplanted position (relative and end-anchored). -> per-anchor dict."""
    pos = [k for k in range(len(y)) if y[k] and len(tl[k])]
    neg = [j for j in range(len(y)) if not y[j]]
    q = []
    for k in pos:
        s = Fm.seq(k, *r["outer"][k])
        q.append(urank(s, Fm.P[k]["t"], Fm.P[k]["ok"], tl[k][0]))
    Q = float(np.nanmean(q))
    out = {"Q_observed": Q}
    for anchor in ("relative", "end"):
        pool = []
        for k in pos:
            row = []
            for j in neg:
                s = Fm.seq(j, *r["pairs"][frozenset((k, j))])
                tau = tl[k][0] / dur[k] * dur[j] if anchor == "relative" else max(0.0, dur[j] - (dur[k] - tl[k][0]))
                row.append(urank(s, Fm.P[j]["t"], Fm.P[j]["ok"], tau))
            pool.append(np.array(row))
        pool = np.array(pool)
        pool_in = np.c_[pool, np.array(q)]  # S7-1: each positive's observed u_k joins its own reference pool (13)
        draws = np.nanmean(np.stack([pool_in[m][rng.integers(0, pool_in.shape[1], N1_DRAWS)]
                                     for m in range(len(pos))]), 0)
        ge = int((draws >= Q - 1e-12).sum())
        out[anchor] = {"p": (1 + ge) / (1 + N1_DRAWS), "null_mean": float(np.mean(draws)),
                       "null_p95": float(np.percentile(draws, 95)), "pool_sd": float(np.nanstd(pool)),
                       "pool_distinct": int(len(np.unique(np.round(pool[np.isfinite(pool)], 9)))),
                       "null_distinct": int(len(np.unique(np.round(draws, 9))))}
    return out


def position_prior(y, tl, dur):
    """nested label-free position priors: relative and end-anchored (median over training positives)."""
    pos = [k for k in range(len(y)) if y[k] and len(tl[k])]
    out = {}
    for anchor in ("relative", "end"):
        err, hits = [], []
        for k in pos:
            o = [i for i in pos if i != k]
            if anchor == "relative":
                th = float(np.median([tl[i][0] / dur[i] for i in o])) * dur[k]
            else:
                th = dur[k] - float(np.median([dur[i] - tl[i][0] for i in o]))
            err.append(min(abs(th - x) for x in tl[k])); hits.append(min(abs(th - x) for x in tl[k]) <= TOL)
        out[anchor] = {"mae_sec": float(np.mean(err)), "hit": float(np.mean(hits))}
    return out


def random_time(P, y, tl, dur):
    ex_u, ex_s = [], []
    for i in range(len(y)):
        if y[i] and len(tl[i]):
            iv = sorted((max(0.0, x - TOL), min(dur[i], x + TOL)) for x in tl[i])
            cov, e = 0.0, -1.0
            for a, b in iv:
                a = max(a, e); cov += max(0.0, b - a); e = max(e, b)
            ex_u.append(cov / dur[i])
            tv = P[i]["t"][P[i]["ok"]]
            ex_s.append(float(np.mean([hit_of(x, tl[i]) for x in tv])) if len(tv) else 0.0)
    return float(np.mean(ex_u)), float(np.mean(ex_s))


def duration_only(y, tl, dur):
    d = np.asarray(dur, float)
    Sd = np.vstack([d, -d])
    p, _t, ch = S4.nested(Sd, np.full_like(Sd, np.nan), np.asarray(y), np.zeros(Sd.shape, bool))
    return {"clip_f1": S4.clip_f1(p, np.asarray(y)), "shorter_chosen": int((ch == 1).sum())}


def null_summary(v, obs):
    v = np.asarray(v, float)
    ge = int((v >= obs - 1e-12).sum())
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "n_null_ge_observed": ge,
            "p_perm": float((1 + ge) / (1 + len(v)))} if len(v) else {"p_perm": float("nan")}


def evaluate_family(Fm, y, tl, dur, n_perm, seed, deadline, complete):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    r = nested(Fm, y, tl, dur, keep_pairs=True)
    t_run = time.time() - t0
    f1, hr, ef1, mae = stats(r, y, tl)
    n1 = transplant(Fm, r, y, tl, dur, rng)
    null, rts = [], [t_run]
    for _ in range(n_perm):
        if time.time() + 2 * max(rts) > deadline:
            break
        t1 = time.time()
        yp, tlp = permute(y, tl, dur, rng)
        null.append(stats(nested(Fm, list(yp), tlp, dur), list(yp), tlp)[:3])
        rts.append(time.time() - t1)
    null = np.array(null).reshape(-1, 3)
    pc = len(null) == n_perm
    ap = S4.clip_f1(np.ones(len(y), bool), np.asarray(y))
    ru, rs = random_time(Fm.P, y, tl, dur)
    pp = position_prior(y, tl, dur)
    do = duration_only(y, tl, dur)
    perm = {"n": n_perm, "n_completed": len(null), "complete": pc, "f1": null_summary(null[:, 0], f1),
            "hit": null_summary(null[:, 1], hr), "event_f1": null_summary(null[:, 2], ef1)}
    rho_med = float(np.nanmedian(np.abs(r["rho"]))) if np.isfinite(r["rho"]).any() else 1.0
    yv = np.asarray(y).astype(bool)
    c = {"1_f1_gt_always_positive": f1 > ap, "2_f1_gt_duration_only": f1 > do["clip_f1"],
         "3_f1_perm_p_le_alpha": pc and perm["f1"]["p_perm"] <= ALPHA, "4_hit_gt_random_time_max": hr > max(ru, rs),
         "5_hit_perm_p_le_alpha": pc and perm["hit"]["p_perm"] <= ALPHA, "6_abs_rho_median_lt_0.5": rho_med < 0.5,
         "7_family_complete": bool(complete), "8_event_f1_perm_p_le_alpha": pc and perm["event_f1"]["p_perm"] <= ALPHA,
         "9_transplant_p_le_alpha_both_anchors": max(n1["relative"]["p"], n1["end"]["p"]) <= ALPHA,
         "10_mae_lt_position_prior_both_anchors": mae < min(pp["relative"]["mae_sec"], pp["end"]["mae_sec"])}
    c = {k: int(bool(v)) for k, v in c.items()}
    c["family_GO"] = int(all(c.values()))
    m = TP.metrics(r["pred"], np.asarray(y), r["tpred"], tl)
    return {"outer": m, "clip_f1": f1, "hit_timestamp_endpoint": hr, "event_f1": ef1, "mae_sec": mae,
            "n_pred_positive": int(r["pred"].sum()), "always_positive_f1": ap, "duration_only_nested": do,
            "random_time_uniform": ru, "random_time_support": rs, "position_prior": pp, "transplant_N1": n1,
            "DIAGNOSTIC_signed_rho_median_all_and_negatives_only":
                [float(np.nanmedian(r["rho"])) if np.isfinite(r["rho"]).any() else None,
                 float(np.nanmedian(r["rho_neg"])) if np.isfinite(r["rho_neg"]).any() else None],
            "perm_N2": perm, "abs_rho_inner_scores_vs_duration_median_max":
                [rho_med, float(np.nanmax(np.abs(r["rho"]))) if np.isfinite(r["rho"]).any() else 1.0],
            "criteria": c, "nested_run_sec": t_run, "perm_sec_p50": float(np.median(rts)),
            "n_pos_with_delta": int(sum(1 for i in range(len(y)) if y[i] and Fm.delta(i, tl[i]) is not None)),
            "event_f1_matches_evaluate": abs(m["event_5s"]["f1"] - ef1) < 1e-3 if m["event_5s"]["f1"] is not None
            else ef1 == 0.0}
