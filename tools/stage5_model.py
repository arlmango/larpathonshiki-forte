"""STAGE 5 numeric core (docs/PROTOCOL_STAGE5.md §2-§6): per-step features, IRLS logistic, fully nested clip-LOO.

Pure numpy, no I/O. Units are dicts {t, E: {"full": (n,768), "zone": (n,768)}, y, tl, dur}. Everything that is
fitted (PCA, standardisation, logistic model, thresholds, configuration choice) is fitted on training clips only.
Label-free fold objects (PCA, standardisation, feature matrices) are cached by training set and reused across
permutations; only the logistic fits, thresholds and selection are redone per labelling.
"""
import itertools
import math

import numpy as np

from tools import stage4_probe as S4

W, MIN_VALID, D_PCA, TGT, RUN_MARGIN, TOL = 5.0, 3, 8, (-10.0, 2.0), 1.0, 5.0
VIEWS, FAMS, LAMS, LOCS = ("full", "zone"), ("E", "T"), (1.0, 10.0, 100.0), ("peak", "run_end")
GFITS = [(v, f, lam) for v in VIEWS for f in FAMS for lam in LAMS]          # 12 distinct fits
CFGS = [(g, loc) for g in range(len(GFITS)) for loc in range(len(LOCS))]    # 24 configurations


def cfg_name(c):
    g, loc = CFGS[c]
    v, f, lam = GFITS[g]
    return f"{v}|{f}|lam{lam:g}|{LOCS[loc]}"


# ============================================================================ label-free per-clip preparation
def windows(t):
    """-> (lo, hi, valid): window for step i = steps with time in (t_i - W, t_i]."""
    hi = np.arange(1, len(t) + 1)
    lo = np.searchsorted(t, t - W, side="right")
    return lo, hi, (hi - lo) >= MIN_VALID


def wmean(x, lo, hi):
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, 0)])
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1).reshape((-1,) + (1,) * (x.ndim - 1))


def prep(u, U_text):
    """Per clip and view: label-free scalars (change, dist-to-median), text-probe base z, window structure."""
    lo, hi, ok = windows(u["t"])
    out = {"lo": lo, "hi": hi, "valid": ok}
    for v in VIEWS:
        e = u["E"][v].astype(np.float64)
        ch = np.concatenate([[np.nan], np.linalg.norm(np.diff(e, axis=0), axis=1)])
        c = np.where(np.isnan(ch), 0.0, ch)
        cnt = wmean(np.isfinite(ch).astype(float), lo, hi)
        change = wmean(c, lo, hi) / np.maximum(cnt, 1e-9) * (cnt > 0)
        dmed = wmean(np.linalg.norm(e - np.median(e, 0), axis=1), lo, hi)
        out[v] = {"e": e, "scal": np.stack([change, dmed], 1), "zT": e @ U_text.T}
    return out


def temporal(z, P):
    """base z (n,d) -> [level, residual, scalars] (n, 2d+2)."""
    lev = wmean(z, P["lo"], P["hi"])
    return np.hstack([lev, lev - np.median(z, 0)])


class FoldCache:
    """Label-free fold objects keyed by (training set, view, family): standardised feature matrices of ALL units."""

    def __init__(self, preps):
        self.P, self.c = preps, {}

    def X(self, train, v, f):
        key = (train, v, f)
        if key not in self.c:
            P = self.P
            if f == "E":
                E = np.vstack([P[i][v]["e"] for i in train])
                mu = E.mean(0)
                _u, _s, Vt = np.linalg.svd(E - mu, full_matrices=False)
                V = Vt[:D_PCA].T
                base = [(P[i][v]["e"] - mu) @ V for i in range(len(P))]
            else:
                base = [P[i][v]["zT"] for i in range(len(P))]
            F = [np.hstack([temporal(base[i], P[i]), P[i][v]["scal"]]) for i in range(len(P))]
            T = np.vstack([F[i][P[i]["valid"]] for i in train])
            m, s = T.mean(0), T.std(0)
            s[s == 0] = 1.0
            self.c[key] = [(x - m) / s for x in F]
        return self.c[key]


# ============================================================================ model
def irls(X, y, w, lam, it=30):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    b = np.zeros(d + 1)
    R = lam * np.eye(d + 1)
    R[-1, -1] = 0.0
    for _ in range(it):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ b, -30, 30)))
        g = Xb.T @ (w * (p - y)) + R @ b
        H = (Xb * (w * p * (1 - p) + 1e-9)[:, None]).T @ Xb + R
        s = np.linalg.solve(H, g)
        b -= s
        if np.abs(s).max() < 1e-7:
            break
    return b


def targets(P, u_t, y, tl):
    """Per clip -> (row mask, row label). Positive clip: valid steps with t in [tl-10, tl+2] (label 1), other steps
    UNUSED. Negative clip: all valid steps (label 0)."""
    ok = P["valid"]
    if not y:
        return ok, np.zeros(ok.sum())
    m = np.zeros(len(u_t), bool)
    for t0 in tl:
        m |= (u_t >= t0 + TGT[0]) & (u_t <= t0 + TGT[1])
    m &= ok
    return m, np.ones(m.sum())


def fit_score(Xs, train, tg, lam, heldout):
    """Fit on the training clips' target rows; -> {held-out clip: logits over its steps}."""
    X = np.vstack([Xs[i][tg[i][0]] for i in train])
    yy = np.concatenate([tg[i][1] for i in train])
    npos, nneg = yy.sum(), len(yy) - yy.sum()
    if npos == 0 or nneg == 0:
        b = np.zeros(X.shape[1] + 1)
    else:
        w = np.where(yy > 0, len(yy) / (2 * npos), len(yy) / (2 * nneg))
        b = irls(X, yy, w, lam)
    return {i: Xs[i] @ b[:-1] + b[-1] for i in heldout}


def localise(t, logit, valid):
    """-> (clip score, t_peak, t_run_end). Label-free."""
    s = np.where(valid, logit, -np.inf)
    if not np.isfinite(s).any():
        return -np.inf, np.nan, np.nan
    k = int(np.argmax(s))
    M, e = s[k], k
    while e + 1 < len(s) and valid[e + 1] and s[e + 1] >= M - RUN_MARGIN:
        e += 1
    return float(M), float(t[k]), float(t[e])


# ============================================================================ nested procedure
def inner_outer_scores(U, P, cache, idx, y, tl, gsel=None):
    """All model scores needed by the nested scheme on unit subset idx (global indices).
    -> Sin[g, k, j] (model trained on idx minus {k, j}, scoring j), Tin[g, loc, k, j], Sout[g, k], Tout[g, loc, k]."""
    gs = range(len(GFITS)) if gsel is None else gsel
    N, G = len(idx), len(GFITS)
    Sin = np.full((G, N, N), np.nan); Tin = np.full((G, 2, N, N), np.nan)
    Sout = np.full((G, N), np.nan); Tout = np.full((G, 2, N), np.nan)
    tg = {i: targets(P[i], U[i]["t"], y[i], tl[i]) for i in idx}
    for a, b in itertools.combinations(range(N), 2):
        train = tuple(idx[m] for m in range(N) if m not in (a, b))
        for g in gs:
            v, f, lam = GFITS[g]
            sc = fit_score(cache.X(train, v, f), train, tg, lam, (idx[a], idx[b]))
            for (k, j) in ((a, b), (b, a)):
                s, tp, te = localise(U[idx[j]]["t"], sc[idx[j]], P[idx[j]]["valid"])
                Sin[g, k, j], Tin[g, 0, k, j], Tin[g, 1, k, j] = s, tp, te
    for k in range(N):
        train = tuple(idx[m] for m in range(N) if m != k)
        for g in gs:
            v, f, lam = GFITS[g]
            sc = fit_score(cache.X(train, v, f), train, tg, lam, (idx[k],))
            s, tp, te = localise(U[idx[k]]["t"], sc[idx[k]], P[idx[k]]["valid"])
            Sout[g, k], Tout[g, 0, k], Tout[g, 1, k] = s, tp, te
    return Sin, Tin, Sout, Tout


def _hit(tp, tls):
    return bool(np.isfinite(tp) and len(tls) and min(abs(tp - x) for x in tls) <= TOL)


def nested(U, P, cache, idx, y, tl, cfg_ok=None):
    """Protocol §5 on subset idx (y, tl indexed globally). -> pred, tpred, chosen cfg per outer fold, per-fold rho."""
    idx = list(idx)
    N = len(idx)
    cfgs = [c for c in range(len(CFGS)) if cfg_ok is None or cfg_ok(c)]
    gsel = sorted({CFGS[c][0] for c in cfgs})
    Sin, Tin, Sout, Tout = inner_outer_scores(U, P, cache, idx, y, tl, gsel)
    yl = np.array([y[i] for i in idx]); tll = [tl[i] for i in idx]
    pred, tpred, ch, rho = np.zeros(N, bool), np.full(N, np.nan), np.zeros(N, int), np.full(N, np.nan)
    for k in range(N):
        tr = np.array([m for m in range(N) if m != k])
        yt = yl[tr]
        key = []
        for c in cfgs:
            g, loc = CFGS[c]
            sc = Sin[g, k, tr]
            v = np.zeros(len(tr), bool)
            for jj in range(len(tr)):
                o = np.arange(len(tr)) != jj
                v[jj] = sc[jj] >= S4.fit_thr(sc[o][None], yt[o])[0]
            f1 = S4.clip_f1(v, yt)
            hits = [_hit(Tin[g, loc, k, tr[jj]], tll[tr[jj]]) for jj in range(len(tr)) if yt[jj] and len(tll[tr[jj]])]
            hr = float(np.mean(hits)) if hits else 0.0
            auc = float(S4.auc_rows(np.nan_to_num(sc, neginf=-1e9)[None], yt)[0])
            key.append((f1 + hr, auc, -c))
        c = cfgs[max(range(len(cfgs)), key=lambda i: key[i])]
        g, loc = CFGS[c]
        thr = S4.fit_thr(Sin[g, k, tr][None], yt)[0]
        pred[k], tpred[k], ch[k] = Sout[g, k] >= thr, Tout[g, loc, k], c
        rho[k] = S4.spearman(np.nan_to_num(Sin[g, k, tr], neginf=-1e9), [U[idx[m]]["dur"] for m in tr])
    return pred, tpred, ch, rho


def hit_rate(tpred, tl, y):
    h = [_hit(tpred[i], tl[i]) for i in range(len(y)) if y[i] and len(tl[i])]
    return float(np.mean(h)) if h else float("nan")


def random_support(U, P, idx, y, tl):
    """Expected hit of a time drawn uniformly over the clip's VALID scan steps (S4-4)."""
    ex = []
    for i in idx:
        if y[i] and len(tl[i]):
            tv = U[i]["t"][P[i]["valid"]]
            ex.append(float(np.mean([min(abs(x - a) for a in tl[i]) <= TOL for x in tv])) if len(tv) else 0.0)
    return float(np.mean(ex)) if ex else float("nan")


def permute(y, tl, dur, rng):
    pi = rng.permutation(len(y))
    yp = y[pi]
    tlp = [[t / dur[pi[i]] * dur[i] for t in tl[pi[i]]] if yp[i] else [] for i in range(len(y))]
    return yp, tlp


def null_summary(v, obs):
    v = np.asarray(v, float)
    ge = int((v >= obs - 1e-12).sum())
    return {"p50": float(np.nanpercentile(v, 50)), "p95": float(np.nanpercentile(v, 95)), "n_null_ge_observed": ge,
            "p_perm": float((1 + ge) / (1 + len(v)))}


def isfinite_all(x):
    return all(math.isfinite(v) for v in x)
