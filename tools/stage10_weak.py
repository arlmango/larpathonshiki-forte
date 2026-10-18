"""STAGE 10 (TRAIN only, EXPLORATORY): weakly supervised L2-logistic head, leave-pair-out (docs/PROTOCOL_STAGE10.md).

usage: python -m tools.stage10_weak --print-lock | --candidate C02 --prelock-duration-auc (durations + pins only)
       python -m tools.stage10_weak --candidate C02 --run --expect-lock <lock16>
Weak supervision (fixed): positives = frames inside the confirmed numeric intervals after the strict stage-8/9 mask;
negatives = ALL frames of the provider-explicit no-violation clips (bag label, human-UNCONFIRMED). Never in the loss:
unknown positive clips (not loaded), other frames of interval clips, controls. Units = clips deduplicated by CONTENT
sha16. Frozen point-level clip-median-centred SigLIP2 zone features; scaler on training clips (K grid frames each);
equal clip weights, balanced classes, lambda 10; clip score = max over K = 20 evenly spaced frames; leave-pair-out,
localisation by leave-one-positive-out. No clip file, frame or holdout record is opened; the computation runs in a
child (job via stdin, stderr DEVNULL, killed at 5 min). Outputs aggregates only (summary O_EXCL 0400); no weights saved.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import stage8_audit as S8, stage9_head as S9  # noqa: E402
from tools.local_safe import Refused, run_guarded  # noqa: E402

PROTOCOL = os.path.join(ROOT, "docs", "PROTOCOL_STAGE10.md")
LOCKED = (PROTOCOL, os.path.join(ROOT, "tools", "stage10_weak.py"), os.path.join(ROOT, "tools", "stage10_selftest.py"))
DEPS = tuple(os.path.join(ROOT, "tools", f) for f in ("stage9_head.py", "stage8_audit.py", "local_safe.py",
                                                       "annotate_intervals.py"))
LAMBDA = S9.LAMBDA                  # 10, one value, no grid
MARGIN = S9.MARGIN                  # 5 s inward on an uncertain side (training rows only)
K_GRID = 20                         # evenly spaced frames per clip: clip score + scaler support
TOL_UNCERTAIN, TOL_CERTAIN = 5.0, 2.0
BUDGET_SEC = 300.0
SUMMARY = "stage10_summary.json"
PIN10 = {"C02": "735de57dbc2ee643"}  # combined inputs digest (protocol §2), from --prelock-duration-auc
OBS, NOT_OBS, STOP = "EXPLORATORY_TRANSFER_OBSERVED", "EXPLORATORY_TRANSFER_NOT_OBSERVED", "STOP_NOT_EVALUABLE"
RULE = ("OBSERVED iff for EVERY held-out positive: top frame inside the confirmed numeric interval (exact) AND mean "
        "in-interval logit > mean logit outside the expanded support AND distance(top frame, interval) < distance("
        "frozen position prior, interval) AND expanded-support share of the clip < 0.5; otherwise NOT_OBSERVED")
TEMPLATE = (RULE + ". A hit means only: on this reused train set, a direction fitted on one confirmed interval vs provider-"
            "negative clips ranked another clip's confirmed interval above (i) that clip's own other frames and/or (ii) "
            "the provider-negative clips; without (i) it means scene similarity; with (i) it may still reflect generic "
            "within-clip activity or novelty shared by the positive clips, not violation specificity; a miss shows "
            "nothing. It is not a violation detector, not generalisation, not a validated FP rate or recall, not "
            "significance")
DISCLOSURE = ("stage 10: ONE exploratory weakly supervised fit; train set reused 5+ times (stages 4, 5, 7, human audit, "
              "8b) and the design followed stage 7; negatives are the provider's explicit whole-clip statements, "
              "human-UNCONFIRMED, never 'confirmed normal'; unknown positives excluded, never negatives; outside-"
              "interval frames of positive clips are unknown, never negatives or FP; units = clips; no p-values "
              "(exchangeability broken by scene, duration and reuse); open risk: whether the positive clips share a "
              "camera/site distinct from the negatives is unverified; GO bars of stages 5/7 untouched and not claimed; "
              "holdout closed; production stays fail-closed; no weights saved")


def lp():
    return S8.lp()


def lock_digest():
    h, parts = hashlib.sha256(), {}
    for p in LOCKED + DEPS:
        try:
            d = S8.fsha(p)
        except OSError:
            d = "missing"
        parts[os.path.basename(p)] = d[:16]
        h.update(os.path.basename(p).encode() + b"\0" + d.encode() + b"\n")
    return h.hexdigest()[:16], parts


# ============================================================================ small statistics
def auc(sp, sn):
    sp, sn = np.asarray(sp, float)[:, None], np.asarray(sn, float)[None, :]
    return float(((sp > sn) + 0.5 * (sp == sn)).mean()) if sp.size and sn.size else float("nan")


def binom_cdf(m, n, p):
    """P(X <= m), X ~ Binomial(n, p); exact."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(m + 1))


def clopper_pearson(k, n, alpha=0.05):
    """exact two-sided CI by bisection in p on the binomial CDF (cdf is decreasing in p):
    lower: P(X >= k | p) = alpha/2  <=>  cdf(k-1, p) = 1 - alpha/2;  upper: cdf(k, p) = alpha/2."""
    def solve(m, target):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if binom_cdf(m, n, mid) > target else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(k - 1, 1 - alpha / 2)
    upper = 1.0 if k == n else solve(k, alpha / 2)
    return [round(lower, 4), round(upper, 4)]


def spearman(a, b):
    def rank(v):
        v = np.asarray(v, float)
        r = np.empty(len(v)); r[np.argsort(v, kind="mergesort")] = np.arange(len(v), dtype=float)
        for u in np.unique(v):
            r[v == u] = r[v == u].mean()
        return r
    ra, rb = rank(a), rank(b)
    return float("nan") if ra.std() == 0 or rb.std() == 0 else float(np.corrcoef(ra, rb)[0, 1])


def duration_auc(U):
    """label-free score (duration) vs the provider/confirmed clip labels; AUC with 'longer = positive'."""
    a = auc([u["dur"] for u in U if u["y"]], [u["dur"] for u in U if not u["y"]])
    sep = max(a, 1 - a) if np.isfinite(a) else float("nan")
    return {"duration_auc_longer_is_positive": round(a, 4), "duration_separability": round(sep, 4),
            "clip_level_descriptive_only": bool(sep >= 1.0)}


# ============================================================================ rows, weights, model
def grid(n, k=K_GRID):
    return np.rint(np.linspace(0, n - 1, k)).astype(int)


def clip_rows(u):
    """-> (row mask, within-clip weights summing to 1, label). Positive: strict mask (S9.rows, controls never used)."""
    P = u["P"]
    if not u["y"]:
        n = len(P["t"])
        return np.ones(n, bool), np.full(n, 1.0 / n), 0
    pm, wp, _nm, _b = S9.rows(P, {"ivs": u["lab"]["ivs"], "ctl": [], "all": u["lab"]["all"]})
    w = wp[pm]
    return pm, (w / w.sum() if w.size else w), 1


def train(U, idx):
    """fit on units idx only. -> (model, None) or (None, reason). Equal clip weights, classes balanced, sum w = n."""
    Xs, ys, ws, G = [], [], [], []
    for i in idx:
        m, w, y = clip_rows(U[i])
        G.append(U[i]["P"]["z"][grid(len(U[i]["P"]["t"]))])
        if m.any():
            Xs.append(U[i]["P"]["z"][m]); ys.append(np.full(m.sum(), float(y))); ws.append(w)
    y = np.concatenate(ys) if ys else np.zeros(0)
    if not (y == 1).any():
        return None, "no_positive_rows_under_strict_mask"
    if not (y == 0).any():
        return None, "no_negative_clips"
    med, sd = S9.scaler(G)                                           # training clips only, K frames each
    X, w = (np.vstack(Xs) - med) / sd, np.concatenate(ws)
    w = np.where(y == 1, w / w[y == 1].sum(), w / w[y == 0].sum()) * (len(y) / 2.0)
    beta, b0, conv = S9.fit_logreg(X, y, w, lam=LAMBDA)
    return {"beta": beta, "b0": b0, "med": med, "sd": sd, "converged": conv}, None


def frame_scores(m, P):
    return ((P["z"] - m["med"]) / m["sd"]) @ m["beta"] + m["b0"]


def clip_score(m, P):
    return float(frame_scores(m, P)[grid(len(P["t"]))].max())


# ============================================================================ localisation (held-out positive)
def hit_spans(lab):
    return [(iv["start"] - (TOL_UNCERTAIN if iv["su"] else TOL_CERTAIN),
             iv["end"] + (TOL_UNCERTAIN if iv["eu"] else TOL_CERTAIN)) for iv in lab["ivs"]]


def coverage(sp, dur):
    cov, e = 0.0, -np.inf
    for a, b in sorted((max(0.0, a), min(dur, b)) for a, b in sp):
        a = max(a, e); cov += max(0.0, b - a); e = max(e, b)
    return cov / dur if dur > 0 else float("nan")


def err_to(tp, spans):
    """distance (s) from a time to the nearest span; 0 inside; inf if tp is not finite."""
    return float(min(0.0 if a <= tp <= b else min(abs(tp - a), abs(tp - b)) for a, b in spans)) if np.isfinite(tp) \
        else float("inf")


def localise(s, u, prior_rel):
    """exact = inside the confirmed numeric bounds; tolerance = inside the pre-registered expanded support.
    Each hit is compared with the uniform chance share of the SAME support."""
    t, lab = u["P"]["t"], u["lab"]
    sp, ex = hit_spans(lab), [(iv["start"], iv["end"]) for iv in lab["ivs"]]
    top = float(t[int(np.argmax(s))])                                # earliest argmax over ALL frames
    tpr = prior_rel * u["dur"] if np.isfinite(prior_rel) else float("nan")
    ins, out = S9.in_any(t, ex), ~S9.in_any(t, sp)
    gap = pct = float("nan")
    if ins.any() and out.any():
        mi = s[ins].mean()
        gap = float(mi - s[out].mean())
        pct = float(((s[out] < mi).sum() + 0.5 * (s[out] == mi).sum()) / out.sum())
    return {"exact_hit": err_to(top, ex) == 0.0, "tolerance_hit": err_to(top, sp) == 0.0,
            "prior_exact_hit": err_to(tpr, ex) == 0.0, "prior_tolerance_hit": err_to(tpr, sp) == 0.0,
            "err_model": err_to(top, ex), "err_prior": err_to(tpr, ex),
            "chance_exact": coverage(ex, u["dur"]), "chance_tolerance": coverage(sp, u["dur"]), "gap": gap, "pct": pct}


def observed_rule(r):
    """pre-registered per held-out positive (protocol §5): -> list of failed conditions (empty = this positive passes)."""
    c = (("no_exact_interval_hit", r["exact_hit"]), ("gap_not_positive", np.isfinite(r["gap"]) and r["gap"] > 0),
         ("not_better_than_position_prior", r["err_model"] < r["err_prior"]),
         ("expanded_support_share_ge_0.5", r["chance_tolerance"] < 0.5))
    return [k for k, ok in c if not ok]


def rel_mid(u):
    iv = u["lab"]["ivs"][0]
    return (iv["start"] + iv["end"]) / 2 / u["dur"]


# ============================================================================ analysis (pure, no I/O)
def _r(v, n=4):
    return round(float(v), n) if np.isfinite(v) else None


def analyse(U, deadline=None):
    """U: [{"P": point_prep, "dur", "y": 0/1, "lab": supervision entry or None}] (distinct content-hash units)."""
    pos = [i for i, u in enumerate(U) if u["y"]]
    neg = [i for i, u in enumerate(U) if not u["y"]]
    out = {"n_positive_units": len(pos), "n_negative_units": len(neg), **duration_auc(U)}
    if len(pos) < 2 or len(neg) < 2:
        out.update(outcome=STOP, stop_reason="fewer than 2 positive or 2 negative units with a valid cache")
        return out

    def tick():
        if deadline and time.time() > deadline: raise Refused("time budget exceeded")  # noqa: E701
    # ---- leave-pair-out: one model per (positive, negative) pair, trained on all other units
    sp_, sn_, mp, mn, conv, untrain = {}, {}, {}, {}, [], collections.Counter()
    for i in pos:
        for j in neg:
            tick()
            m, why = train(U, [k for k in range(len(U)) if k not in (i, j)])
            if m is None:
                untrain[why] += 1
                continue
            thr = max(clip_score(m, U[k]["P"]) for k in neg if k != j)      # fixed rule: max training-negative score
            si, sj = clip_score(m, U[i]["P"]), clip_score(m, U[j]["P"])
            sp_[i, j], sn_[i, j], mp[i, j], mn[i, j] = si, sj, si - thr, sj - thr
            conv.append(m["converged"])
            fi = frame_scores(m, U[i]["P"])
            ins = S9.in_any(U[i]["P"]["t"], [(iv["start"], iv["end"]) for iv in U[i]["lab"]["ivs"]])
            mp[i, j, "imax_win"] = float((fi[ins].max() > sj) + 0.5 * (fi[ins].max() == sj)) if ins.any() else np.nan
            mp[i, j, "hit"] = localise(fi, U[i], np.nan)["exact_hit"]
    keys = list(sp_)
    lpo = (float(np.mean([(sp_[k] > sn_[k]) + 0.5 * (sp_[k] == sn_[k]) for k in keys])) if keys else float("nan"))
    # ---- clip-level verdicts: median margin over the pair models in which the clip was held out
    med_pos = {i: np.median([mp[i, j] for j in neg if (i, j) in sp_]) for i in pos if any((i, j) in sp_ for j in neg)}
    med_neg = {j: np.median([mn[i, j] for i in pos if (i, j) in sp_]) for j in neg if any((i, j) in sp_ for i in pos)}
    tp = sum(v > 0 for v in med_pos.values()); fn = len(med_pos) - tp
    fp = sum(v > 0 for v in med_neg.values()); tn = len(med_neg) - fp
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    # median held-out clip score per unit vs duration (descriptive)
    msc = {i: np.median([sp_[i, j] for j in neg if (i, j) in sp_]) for i in med_pos}
    msc.update({j: np.median([sn_[i, j] for i in pos if (i, j) in sp_]) for j in med_neg})
    ks = sorted(msc)
    rho = spearman([msc[k] for k in ks], [U[k]["dur"] for k in ks]) if len(ks) > 2 else float("nan")
    # ---- localisation: leave-one-positive-out model per held-out positive
    loc, why_l = [], collections.Counter()
    for i in pos:
        tick()
        m, why = train(U, [k for k in range(len(U)) if k != i])
        if m is None:
            why_l[why] += 1
            continue
        others = [rel_mid(U[k]) for k in pos if k != i]
        r = localise(frame_scores(m, U[i]["P"]), U[i], float(np.median(others)))
        r["ii_share"] = float(np.nanmean([mp[i, j, "imax_win"] for j in neg if (i, j) in sp_])) \
            if any((i, j) in sp_ for j in neg) else float("nan")
        r["pair_hits"] = [int(sum(mp[i, j, "hit"] for j in neg if (i, j) in sp_)), sum((i, j) in sp_ for j in neg)]
        conv.append(m["converged"])
        loc.append(r)
    fails = collections.Counter(x for r in loc for x in observed_rule(r))
    n_pass = sum(not observed_rule(r) for r in loc)
    cnt = lambda key: [int(sum(r[key] for r in loc)), len(loc)]  # noqa: E731
    out.update({
        "n_pairs": len(pos) * len(neg), "n_pairs_trainable": len(keys), "pairs_not_trainable": dict(untrain),
        "lpo_auc_clip_grid_max": _r(lpo), "lpo_auc_minus_duration_separability": _r(lpo - out["duration_separability"]),
        "pair_slots_flagged_positive_heldout": [int(sum(mp[k] > 0 for k in keys)), len(keys)],
        "pair_slots_flagged_negative_heldout": [int(sum(mn[k] > 0 for k in keys)), len(keys)],
        "clip_level": {"rule": "median over the clip's pair models of (held-out clip score - max training-negative "
                                "clip score) > 0", "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
                       "recall": [int(tp), len(med_pos)], "fp_of_negatives": [int(fp), len(med_neg)],
                       "fp_rate_cp95": clopper_pearson(int(fp), len(med_neg)) if med_neg else None,
                       "f1": round(f1, 4), "always_positive_f1": round(2 * len(pos) / (2 * len(pos) + len(neg)), 4)},
        "spearman_heldout_clip_score_vs_duration": _r(rho),
        "localisation": {"n_evaluable": len(loc), "not_evaluable": dict(why_l),
                         "exact_interval_hits": cnt("exact_hit"), "tolerance_hits": cnt("tolerance_hit"),
                         "prior_exact_hits": cnt("prior_exact_hit"), "prior_tolerance_hits": cnt("prior_tolerance_hit"),
                         "model_error_lt_prior_error": [int(sum(r["err_model"] < r["err_prior"] for r in loc)), len(loc)],
                         "chance_exact_share_sorted": sorted(_r(r["chance_exact"], 3) for r in loc),
                         "chance_tolerance_share_sorted": sorted(_r(r["chance_tolerance"], 3) for r in loc),
                         "uniform_all_exact_hit_reference_product":
                             _r(float(np.prod([r["chance_exact"] for r in loc])), 4) if loc else None,
                         "uniform_all_tolerance_hit_reference_product":
                             _r(float(np.prod([r["chance_tolerance"] for r in loc])), 4) if loc else None,
                         "observed_rule_positives_passing": [int(n_pass), len(loc)],
                         "observed_rule_failed_conditions": dict(fails),
                         "in_minus_out_logit_gap_sorted": sorted(_r(r["gap"], 3) for r in loc if np.isfinite(r["gap"])),
                         "in_interval_mean_percentile_sorted": sorted(_r(r["pct"], 3) for r in loc
                                                                      if np.isfinite(r["pct"])),
                         "ii_interval_max_beats_heldout_negative_share_sorted":
                             sorted(_r(r["ii_share"], 3) for r in loc if np.isfinite(r["ii_share"])),
                         "pair_model_exact_hits": [r["pair_hits"] for r in loc]},
        "irls_converged_all": bool(all(conv))})
    if len(loc) < len(pos):
        out.update(outcome=STOP, stop_reason="a leave-one-positive-out model is not trainable")
    else:
        out["outcome"] = OBS if n_pass == len(pos) else NOT_OBS
        if out["outcome"] == NOT_OBS:
            out["not_observed_sub_reasons"] = sorted(fails)
    return out


# ============================================================================ I/O (train caches of selected units only)
def units(cid, rel, sha):
    """-> (U, coverage counts, supervision counts). Positives without a numeric interval are never loaded."""
    path, sha = S8.snapshot_spec(cid, rel, sha)
    latest = S8.read_snapshot(path, sha)
    pos, neg, cnt = S8.train_sets(cid)                        # sets keyed by content sha16 -> deduplicated
    if pos & neg:
        raise Refused("a content hash is both positive and negative")
    lab, g = S9.supervision(latest, pos, neg)
    U, c = [], collections.Counter()
    for s in sorted(lab):
        r = S9.load_emb(s)
        c["positive_cache_missing_or_invalid"] += r is None
        if r:
            U.append({"P": r[0], "dur": r[1], "y": 1, "lab": lab[s]})
    for s in sorted(neg):
        r = S9.load_emb(s)
        c["negative_cache_missing_or_invalid"] += r is None
        if r:
            U.append({"P": r[0], "dur": r[1], "y": 0, "lab": None})
    c.update(n_train_positive_units=len(pos), n_train_negative_units_distinct_content=len(neg),
             n_positive_units_excluded_no_numeric_interval=len(pos) - len(lab), selector_skipped=int(cnt["skipped"]))
    g.pop("n_controls_used", None)
    return U, {k: int(v) for k, v in sorted(c.items())}, {k: int(v) for k, v in sorted(g.items())}


def input_pins(cid, rel, sha):
    """-> (components sha256-16, combined sha256-16): snapshot, split file, imported-label file and the manifest of
    the SigLIP2 zone caches of the units that are used (lab positives + distinct negatives). Unknown-task caches are
    not part of the manifest and are not opened."""
    path, sha = S8.snapshot_spec(cid, rel, sha)
    latest = S8.read_snapshot(path, sha)
    pos, neg, _c = S8.train_sets(cid)
    lab, _g = S9.supervision(latest, pos, neg)
    man = hashlib.sha256()
    used = sorted(set(lab) | set(neg))
    for s in used:
        cp = os.path.join(lp(), "train_probe", "cache_emb_siglip2", f"{s}.npz")
        man.update(f"{s} {S8.fsha(cp) if os.path.isfile(cp) else 'absent'}\n".encode())
    full = {"snapshot": sha, "split": S8.fsha(os.path.join(lp(), "splits", f"{cid}.json")),
            "imported_labels": S8.fsha(os.path.join(lp(), "labels", f"{cid}.imported.jsonl")),
            "cache_manifest": man.hexdigest()}
    comb = hashlib.sha256(json.dumps(full, sort_keys=True).encode()).hexdigest()[:16]
    return {**{k: v[:16] for k, v in full.items()}, "n_caches_in_manifest": len(used)}, comb


def compute(cid, rel, sha, deadline=None):
    U, cov, sup = units(cid, rel, sha)
    res = analyse(U, deadline)
    res.update(coverage=cov, supervision_counts=sup, template=TEMPLATE, model_artefact_saved=False)
    return res


def child_main():
    job = json.loads(sys.stdin.readline())
    S8.TP.LP = job["lp"]
    try:
        r = compute(job["cid"], job["rel"], job["sha"], float(job["deadline"]))
    except Refused:
        r = {"err": "refused"}
    except (OSError, ValueError, KeyError, RuntimeError, np.linalg.LinAlgError):
        r = {"err": "failed"}
    r["child_maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1)
    sys.stdout.write(json.dumps(r, default=float) + "\n")
    return 0


def run_child(job, budget):
    p = subprocess.Popen([sys.executable, "-m", "tools.stage10_weak", "--child"], cwd=ROOT, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    try:
        out, _ = p.communicate((json.dumps(job) + "\n").encode(), timeout=max(1.0, budget))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        p.wait()
        return {}, "timeout"
    try:
        r = json.loads(out.decode().strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {}, "failed"
    return r, ("ok" if p.returncode == 0 and "err" not in r else r.get("err", "failed"))


# ============================================================================ CLI
def _prelock(a):
    """pre-lock duration-only AUC (protocol §6): durations of the evaluable units vs their labels; 4 aggregates."""
    S9._stage8_closed()
    h0 = S8.immutables(a.candidate)
    pins, comb = input_pins(a.candidate, a.snapshot, a.snapshot_sha256)
    U, cov, _sup = units(a.candidate, a.snapshot, a.snapshot_sha256)
    d = duration_auc(U)
    h1 = S8.immutables(a.candidate)
    print("PINS " + json.dumps({**pins, "inputs_sha256_16": comb}, sort_keys=True))
    print("PRELOCK " + json.dumps({"n_positive_units": sum(u["y"] for u in U),
                                   "n_negative_units": sum(not u["y"] for u in U), **d,
                                   "caches_missing": cov.get("positive_cache_missing_or_invalid", 0)
                                   + cov.get("negative_cache_missing_or_invalid", 0),
                                   "immutables_unchanged": h0[2] == h1[2]}, sort_keys=True))
    return 0


def _main(argv=None):
    ap = argparse.ArgumentParser(description="stage 10 weakly supervised exploratory head (aggregates only)")
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--print-lock", action="store_true")
    m.add_argument("--prelock-duration-auc", action="store_true")
    m.add_argument("--run", action="store_true")
    m.add_argument("--child", action="store_true")
    ap.add_argument("--candidate", default="")
    ap.add_argument("--expect-lock", default="")
    ap.add_argument("--snapshot", default=None, help="relative to the private dir (non-pinned candidates only)")
    ap.add_argument("--snapshot-sha256", default=None)
    ap.add_argument("--expect-inputs", default="", help="combined inputs digest (non-pinned candidates only)")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.child:
        return child_main()
    t0 = time.time()
    lock, parts = lock_digest()
    print(f"lock_sha256_16={lock} files={json.dumps(parts, sort_keys=True)}")
    if a.print_lock:
        return 0
    if a.prelock_duration_auc:
        return _prelock(a)
    if a.expect_lock != lock:
        raise Refused("lock digest mismatch: protocol or code changed since the lock")
    out = os.path.join(lp(), "train_probe", SUMMARY)
    if os.path.lexists(out):
        raise Refused("summary already exists: single pre-registered run")
    spath, ssha = S8.snapshot_spec(a.candidate, a.snapshot, a.snapshot_sha256)
    S8.read_snapshot(spath, ssha)                                    # verify sha before the child starts
    g0 = S9._stage8_closed()
    want = PIN10.get(a.candidate) or a.expect_inputs
    if a.candidate in PIN10 and a.expect_inputs and a.expect_inputs != PIN10[a.candidate]:
        raise Refused("inputs digest differs from the pinned one")
    pins0, comb0 = input_pins(a.candidate, a.snapshot, a.snapshot_sha256)
    if not want or comb0 != want:
        raise Refused("inputs digest mismatch: snapshot, split, labels or caches differ from the pre-registration")
    h0, s0 = S8.immutables(a.candidate), S8.fsha(spath)
    r, status = run_child({"lp": lp(), "cid": a.candidate, "rel": a.snapshot, "sha": a.snapshot_sha256,
                           "deadline": t0 + BUDGET_SEC - 10}, BUDGET_SEC - (time.time() - t0))
    if status != "ok":
        r = {"outcome": f"FAILED_{status.upper()}", "child_maxrss_mb": r.get("child_maxrss_mb")}
    h1, s1, g1 = S8.immutables(a.candidate), S8.fsha(spath), S9._stage8_closed()
    try:
        comb1 = input_pins(a.candidate, a.snapshot, a.snapshot_sha256)[1]
    except Exception:  # noqa: BLE001 - a post-run pin failure must not lose the summary; static marker only
        comb1 = "error"
    r.update(protocol="PROTOCOL_STAGE10", lock_sha256_16=lock, locked_files_sha256_16=parts, disclosure=DISCLOSURE,
             runtime_sec=round(time.time() - t0, 2), model_artefact_saved=False,
             fixed={"lambda": LAMBDA, "margin_sec": MARGIN, "k_grid": K_GRID, "tol_uncertain_certain_sec":
                    [TOL_UNCERTAIN, TOL_CERTAIN], "features": "point_level_clip_median_centred_siglip2_zone",
                    "weights": "equal per clip, classes balanced, sum = n rows", "validation": "leave-pair-out"},
             snapshot={"sha256_verified": s0 == ssha, "unchanged": s0 == s1}, stage8_summary_unchanged=g0 == g1,
             inputs={"pins_sha256_16": pins0, "inputs_sha256_16": comb0, "verified": comb0 == want,
                     "unchanged_after_run": comb0 == comb1, "pins_after": "error" if comb1 == "error" else "ok"},
             immutable_files={"n": h0[0], "match_recorded_sha16": h1[1],
                              "status": "unchanged" if h0[2] == h1[2] else "CHANGED"})
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "w") as f:
        json.dump(r, f, indent=1, sort_keys=True, default=float)
    print("STAGE10 " + json.dumps({k: v for k, v in r.items() if k not in ("locked_files_sha256_16", "disclosure",
                                                                         "template")}, sort_keys=True, default=float))
    print(f"OUTCOME={r['outcome']}; {TEMPLATE}")
    print(f"snapshot verified={int(s0 == ssha)} unchanged={int(s0 == s1)}; stage8 summary unchanged={int(g0 == g1)}; "
          f"immutables n={h0[0]} match_recorded={h1[1]} {r['immutable_files']['status']}; summary written (0400); "
          "no weights saved")
    return 0 if status == "ok" else 2


def main(argv=None):
    return run_guarded(_main, argv)


if __name__ == "__main__":
    sys.exit(main())
