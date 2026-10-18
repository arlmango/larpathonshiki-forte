"""STAGE 5 (TRAIN only): frozen SigLIP2 semantics + temporal features + small logistic model, fully nested clip-LOO.

usage: python -m tools.stage5_probe --candidate C02 --n-perm {1000|200} [--n-perm-sens 200] [--budget-sec 900]
Implements docs/PROTOCOL_STAGE5.md (locked; its sha256-16 is printed). Units only via
local_safe.select_split_units(side "train"): dry run first; only clips whose embedding cache is missing are opened
(side/index checked BEFORE open, sha re-checked after) and embedded by the ORT child (job via stdin, stderr DEVNULL,
RSS watchdog PILOT_MEM_MB, default 8192). Embeddings live only in <LP>/train_probe/cache_emb_siglip2/.
stdout and <LP>/train_probe/stage5_summary.json carry AGGREGATES ONLY (no shas, paths, names, per-clip rows/times).
Immutable files (labels/*, inventory_map.json, train_probe/stage4_summary.json): sha256 before/after, printed as
unchanged/CHANGED only. Localisation = timestamp endpoint (±5 s of the label second), NOT onset/interval/legal.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import stage4_probe as S4, stage5_encoder as ENC, stage5_model as M, train_probe as TP  # noqa: E402
from tools.local_safe import Refused, run_guarded, select_split_units  # noqa: E402

PROTOCOL = os.path.join(ROOT, "docs", "PROTOCOL_STAGE5.md")
ZONE_SCALE = 1.5


def lp():
    return TP.LP


# ============================================================================ immutable files
def immutable_digest():
    files = sorted(glob.glob(os.path.join(lp(), "labels", "*"))) + [
        os.path.join(lp(), "inventory_map.json"), os.path.join(lp(), "train_probe", "stage4_summary.json")]
    h = hashlib.sha256()
    for p in files:
        h.update(os.path.basename(p).encode())
        try:
            with open(p, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
        except OSError:
            h.update(b"absent")
    return h.hexdigest(), len(files)


# ============================================================================ units
def expand(z):
    cx, cy, w, h = (z[0] + z[2]) / 2, (z[1] + z[3]) / 2, (z[2] - z[0]) * ZONE_SCALE, (z[3] - z[1]) * ZONE_SCALE
    return [max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)]


def stage4_zone_dur(sha):
    p = os.path.join(lp(), "train_probe", "cache", f"{sha}_4.npz")
    if not os.path.exists(p):
        return None, None
    try:
        z = np.load(p)
        r = TP.features(z)
        return (None if r is None else [float(v) for v in r[2]["zone"]]), float(z["dur"])
    except (OSError, ValueError, KeyError):
        return None, None


def load_units(cid, budget, mem_mb, deadline=float("inf")):
    units, cnt = select_split_units(lp(), cid, "train", dry_run=True)  # no clip file opened
    lab = [(sha, int(st == "reviewed_violation"), ev) for sha, _p, st, ev in units
           if st in ("reviewed_violation", "reviewed_no_violation")]
    cdir = os.path.join(lp(), "train_probe", "cache_emb_siglip2")
    os.makedirs(cdir, exist_ok=True)
    out = lambda sha: os.path.join(cdir, f"{sha}.npz")  # noqa: E731
    zinfo = {sha: stage4_zone_dur(sha) for sha, _y, _e in lab}
    missing = [sha for sha, _y, _e in lab if not os.path.exists(out(sha))]
    st, walls, dec, emb, peaks, frames_src = collections.Counter(), [], [], [], [], 0
    if missing:
        if not ENC.onnx_ok():
            raise Refused("pinned ONNX vision tower missing or sha256 mismatch")
        full, _c = select_split_units(lp(), cid, "train")  # train side only; sha re-checked after open
        paths = {sha: p for sha, p, _s, _e in full}
        for sha in missing:
            if sha not in paths:
                st["not_selectable"] += 1
                continue
            remaining = deadline - time.time()
            if remaining < 30:  # global wall-clock guard: stop the phase, count the rest as failures
                st["incomplete_budget"] += 1
                continue
            z, d4 = zinfo[sha]
            tmp = out(sha) + ".part.npz"
            r = ENC.run_watched(ENC.MAIN_PY, "embed", {"path": paths[sha], "out": tmp, "dur_expected": d4,
                                                         "zone": expand(z) if z else None},
                                min(budget, remaining), mem_mb)
            st[r["status"]] += 1
            peaks.append(r["peak_seen_mb"])
            if r["status"] == "ok" and os.path.exists(tmp):
                os.replace(tmp, out(sha))
                walls.append(r["wall_s"]); dec.append(r["decode_s"]); emb.append(r["embed_s"])
                frames_src += int(r["n_src_frames"])
            elif os.path.exists(tmp):
                os.remove(tmp)
    U, failed, zfb = [], 0, 0
    for sha, y, ev in lab:
        if not os.path.exists(out(sha)):
            failed += 1
            continue
        z = np.load(out(sha))
        t = z["t"].astype(np.float64)
        if len(t) < 3:
            failed += 1
            continue
        E = {v: z[f"E_{v}"].astype(np.float64) for v in M.VIEWS}
        for v in E:
            E[v] /= np.maximum(np.linalg.norm(E[v], axis=1, keepdims=True), 1e-9)
        zone, d4 = zinfo[sha]
        fb = bool(int(z["zone_fallback"])) or zone is None
        zfb += fb
        U.append({"sha": sha, "y": y, "tl": [float(v) for v in ev], "t": t, "E": E,
                  "dur": d4 if d4 else float(z["dur_dec"]), "zone": zone if zone else [np.nan] * 4})

    def q(v):
        return [float(np.percentile(v, 50)), float(np.percentile(v, 95)), float(np.max(v))] if v else None
    cov = {"label_units_selected": len(lab), "skipped_by_selector": cnt["skipped"] + (len(units) - len(lab)),
           "emb_caches_present_at_start": len(lab) - len(missing), "embed_child_status": dict(st),
           "excluded_failed_decode_or_embedding": failed, "primary_complete": failed == 0, "zone_fallback_full_frame": zfb,
           "embed_wall_s_p50_p95_max": q(walls), "decode_s_p50_p95_max": q(dec), "embed_s_p50_p95_max": q(emb),
           "decode_s_total": float(sum(dec)), "embed_s_total": float(sum(emb)), "source_frames_decoded": frames_src,
           "child_peak_rss_mb_max": float(max(peaks)) if peaks else None, "mem_cap_mb": mem_mb}
    return U, cov


# ============================================================================ evaluation
def evaluate(U, P, cache, idx, n_perm, seed, cfg_ok=None, deadline=float("inf")):
    """n_perm is fixed by the protocol; if the global deadline would be crossed the permutation phase STOPS and is
    reported incomplete (perm criteria then not evaluable -> NO-GO by rule). Never adapted to results."""
    idx = list(idx)
    y = np.array([u["y"] for u in U]); tl = [u["tl"] for u in U]; dur = np.array([u["dur"] for u in U])
    t0 = time.time()
    pred, tpred, ch, rho = M.nested(U, P, cache, idx, y, tl, cfg_ok)
    t_first = time.time() - t0
    yl, tll, dl = y[idx], [tl[i] for i in idx], dur[idx]
    obs_f1, obs_hr = S4.clip_f1(pred, yl), M.hit_rate(tpred, tll, yl)
    rng = np.random.default_rng(seed)
    nf, nh, rt = [], [], [t_first]
    for _ in range(n_perm):
        if time.time() + 2 * max(rt) > deadline:
            break
        t1 = time.time()
        yp, tlp = M.permute(yl, tll, dl, rng)
        yg, tlg = y.copy(), list(tl)
        for m, i in enumerate(idx):
            yg[i], tlg[i] = yp[m], tlp[m]
        pp, tp_, _c, _r = M.nested(U, P, cache, idx, yg, tlg, cfg_ok)
        nf.append(S4.clip_f1(pp, yp)); nh.append(M.hit_rate(tp_, tlp, yp))
        rt.append(time.time() - t1)
    cnt = collections.Counter(M.cfg_name(c) for c in ch)
    parts = [n.split("|") for n in (M.cfg_name(c) for c in ch)]
    return {"n_units": len(idx), "n_pos": int(yl.sum()), "outer": TP.metrics(pred, yl, tpred, tll),
            "outer_clip_f1": obs_f1, "outer_hit_rate_timestamp_endpoint": obs_hr,
            "random_time_uniform_clip": S4.random_time(tll, dl, yl),
            "random_time_valid_support": M.random_support(U, P, idx, y, tl),
            "always_positive": TP.metrics(np.ones(len(idx), bool), yl, np.full(len(idx), np.nan), tll),
            "always_positive_clip_f1": S4.clip_f1(np.ones(len(idx), bool), yl),
            "perm": {"n": int(n_perm), "n_completed": len(nf), "complete": len(nf) == n_perm, "seed": seed, "f1": M.null_summary(nf, obs_f1),
                     "hit_rate": M.null_summary(nh, obs_hr)},
            "chosen_configs": {"distinct": len(cnt), "top": cnt.most_common(6),
                               **{d: dict(collections.Counter(p[k] for p in parts))
                                  for k, d in enumerate(("view", "family", "lambda", "localisation"))}},
            "chosen_heldout_score_vs_duration_abs_spearman_median_max":
                [float(np.nanmedian(np.abs(rho))), float(np.nanmax(np.abs(rho)))] if np.isfinite(rho).any()
                else [float("nan"), float("nan")],
            "nested_run_sec_p50_p95_max": [float(np.percentile(rt, 50)), float(np.percentile(rt, 95)), float(max(rt))]}


def duration_only(U, idx):
    y = np.array([U[i]["y"] for i in idx]); tl = [U[i]["tl"] for i in idx]; d = np.array([U[i]["dur"] for i in idx])
    Sd = np.vstack([d, -d])
    p, _t, ch = S4.nested(Sd, np.full_like(Sd, np.nan), y, np.zeros(Sd.shape, bool))
    m = TP.metrics(p, y, np.full(len(y), np.nan), tl)
    m["clip_f1"] = S4.clip_f1(p, y)
    m["sign_chosen_counts"] = {"longer=violation": int((ch == 0).sum()), "shorter=violation": int((ch == 1).sum())}
    return m


def decide(p, dur_f1, complete=True):
    rt = max(p["random_time_uniform_clip"]["clipped_exact"], p["random_time_valid_support"])
    hr = float(np.nan_to_num(p["outer_hit_rate_timestamp_endpoint"]))
    c = {"f1_gt_always_positive": int(p["outer_clip_f1"] > p["always_positive_clip_f1"]),
         "f1_gt_duration_only": int(p["outer_clip_f1"] > dur_f1),
         "f1_gt_perm_p95": int(p["perm"]["complete"] and p["outer_clip_f1"] > p["perm"]["f1"]["p95"]),
         "hit_gt_random_time_max": int(hr > rt),
         "hit_gt_perm_p95": int(p["perm"]["complete"] and hr > p["perm"]["hit_rate"]["p95"]),
         "duration_abs_rho_median_lt_0.5":
             int(np.nan_to_num(p["chosen_heldout_score_vs_duration_abs_spearman_median_max"][0], nan=1.0) < 0.5)}
    c["primary_complete_all_units"] = int(complete)  # amendment: no GO on an incomplete primary set
    c["decision_GO"] = int(all(c.values()))
    if not p["perm"]["complete"]:
        c["perm_criteria"] = "not_evaluable_incomplete"
    return c


def _main(argv=None):
    ap = argparse.ArgumentParser(description="stage 5 train-only probe (aggregates only)")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--n-perm", type=int, required=True)
    ap.add_argument("--n-perm-sens", type=int, default=200)
    ap.add_argument("--budget-sec", type=float, default=900.0)
    ap.add_argument("--mem-mb", type=int, default=int(os.environ.get("PILOT_MEM_MB", "8192")))
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--budget-total-sec", type=float, default=5400.0, help="ONE overall wall-clock budget (90 min)")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    deadline = t0 + a.budget_total_sec
    incomplete = []
    proto = S4.sha256_or_none(PROTOCOL)
    h0 = immutable_digest()
    try:
        U, cov = load_units(a.candidate, a.budget_sec, a.mem_mb, deadline)
        Ut = ENC.text_probes()
    except (Refused, KeyError, OSError, ValueError, RuntimeError) as e:
        print("refused: " + (str(e) if isinstance(e, Refused) else "inputs missing or malformed"), file=sys.stderr)
        return 2
    if len(U) < 6 or len({u["y"] for u in U}) < 2:
        print(f"refused: too few usable train units ({len(U)})", file=sys.stderr)
        return 2
    t_load = time.time() - t0
    P = [M.prep(u, Ut) for u in U]
    cache = M.FoldCache(P)
    n = [len(u["t"]) for u in U]
    dur = np.array([u["dur"] for u in U]); y = np.array([u["y"] for u in U])
    pos = [u for u in U if u["y"] and u["tl"]]
    cov.update({"units": len(U), "pos": int(y.sum()), "neg": int((1 - y).sum()), "views": list(M.VIEWS),
                "frames_per_clip_min_med_max": [int(min(n)), float(np.median(n)), int(max(n))],
                "frames_total": int(sum(n)), "valid_steps_total": int(sum(p_["valid"].sum() for p_ in P)),
                "duration_sec_min_med_max": [float(dur.min()), float(np.median(dur)), float(dur.max())],
                "share_label_times_le_duration": float(np.mean([x <= u["dur"] for u in pos for x in u["tl"]]))
                if pos else float("nan"), "n_configs": len(M.CFGS)})
    allidx = list(range(len(U)))
    if cov["embed_child_status"].get("incomplete_budget"):
        incomplete.append("embedding")
    prim = evaluate(U, P, cache, allidx, a.n_perm, 0, deadline=deadline)
    if not prim["perm"]["complete"]:
        incomplete.append("primary_permutations")
    dur_only = duration_only(U, allidx)
    conf = {"pointbiserial_r_duration_vs_label": float(np.corrcoef(dur, y)[0, 1]),
            "auc_duration_vs_label": float(S4.auc_rows(dur[None], y)[0])}
    sens = abl = None
    agree, zc = S4.zone_groups(a.candidate, U)
    if not a.no_sensitivity:
        est = 2 * prim["nested_run_sec_p50_p95_max"][1] * (a.n_perm_sens + 1)
        if time.time() + est < deadline:
            sens = evaluate(U, P, cache, allidx, a.n_perm_sens, 2, deadline=deadline,
                            cfg_ok=lambda c: M.GFITS[M.CFGS[c][0]][0] == "full")
        else:
            incomplete.append("sensitivity")
        if agree.sum() >= 6 and len(set(y[agree])) == 2 and time.time() + est < deadline:
            abl = evaluate(U, P, cache, list(np.flatnonzero(agree)), a.n_perm_sens, 1, deadline=deadline)
        elif agree.sum() >= 6:
            incomplete.append("ablation")
    crit = decide(prim, dur_only["clip_f1"], cov["primary_complete"])
    h1 = immutable_digest()
    res = {"protocol": "PROTOCOL_STAGE5 locked", "protocol_sha256_16": (proto or "missing")[:16], "coverage": cov,
           "primary_all_units": prim, "baseline_duration_only_nested": dur_only, "duration_confound": conf,
           "SENSITIVITY_view_full_only": sens, "zone_groups": zc, "ABLATION_zone_agree_only_NOT_DEPLOYABLE": abl,
           "decision": crit, "incomplete_phases": incomplete,
           "primary_label": "complete primary" if cov["primary_complete"] else "INCOMPLETE PRIMARY (metrics on the "
                            "completed subset only; GO not issued)", "immutable_files": {"n": h0[1], "status": "unchanged" if h0 == h1 else "CHANGED"},
           "runtime_sec": {"total": time.time() - t0, "load_and_embed": t_load},
           "parent_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
           "disclosure": "adaptive exploratory research after stage 4 on the same train sample; no independent "
                         "validation; holdout untouched and stays closed even under GO until a separate agreement"}
    with open(os.path.join(lp(), "train_probe", "stage5_summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    _print(res)
    return 0


def _f(v):
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _line(name, m):
    c, e = m["clip"], m["event_5s"]
    return (f"{name:34s} clip TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']} P={_f(c['precision'])} "
            f"R={_f(c['recall'])} F1={_f(c['f1'])} | event±5s TP={e['tp']} FP={e['fp']} FN={e['fn']} F1={_f(e['f1'])}")


def _block(title, r):
    print(f"\n{title}  (n={r['n_units']}, pos={r['n_pos']})")
    print(_line("nested outer folds", r["outer"]))
    print(_line("always-positive", r["always_positive"]))
    print(f"timestamp-endpoint hit rate = {_f(r['outer_hit_rate_timestamp_endpoint'])}; random-time uniform="
          f"{_f(r['random_time_uniform_clip']['clipped_exact'])} valid-support={_f(r['random_time_valid_support'])}")
    for k in ("f1", "hit_rate"):
        q = r["perm"][k]
        print(f"permutation ({r['perm']['n']}x whole nested selection) {k:8s}: p50={_f(q['p50'])} p95={_f(q['p95'])} "
              f"n_null>=obs={q['n_null_ge_observed']} p_perm={_f(q['p_perm'])}")
    cc = r["chosen_configs"]
    print(f"chosen configs: distinct={cc['distinct']} top={cc['top']}")
    print(f"  view={cc['view']} family={cc['family']} lambda={cc['lambda']} loc={cc['localisation']}")
    print("  chosen held-out score vs duration |spearman| median/max = "
          + "/".join(_f(v) for v in r["chosen_heldout_score_vs_duration_abs_spearman_median_max"])
          + "; nested run s p50/p95/max = " + "/".join(f"{v:.2f}" for v in r["nested_run_sec_p50_p95_max"]))


def _print(res):
    print(f"protocol_doc_sha256_16={res['protocol_sha256_16']}")
    print("COVERAGE " + json.dumps(res["coverage"]))
    _block("PRIMARY (all units) fully nested clip-LOO, inner-LOO selection by F1+hit", res["primary_all_units"])
    d = res["baseline_duration_only_nested"]
    print(_line("baseline duration-only (nested)", d), d["sign_chosen_counts"])
    print("DURATION CONFOUND " + json.dumps(res["duration_confound"]))
    if res["SENSITIVITY_view_full_only"]:
        _block("SENSITIVITY view=full only (not used for decision)", res["SENSITIVITY_view_full_only"])
    print("ZONE GROUPS " + json.dumps(res["zone_groups"]))
    if res["ABLATION_zone_agree_only_NOT_DEPLOYABLE"]:
        _block("ABLATION zone-agree units only (NOT deployable)", res["ABLATION_zone_agree_only_NOT_DEPLOYABLE"])
    print(f"\nPRIMARY SET: {res['primary_label']}; incomplete phases: {res['incomplete_phases'] or 'none'}")
    print("DECISION " + json.dumps(res["decision"]) + (" -> exploratory GO (holdout stays closed pending a separate "
                                                         "agreement)" if res["decision"]["decision_GO"] else " -> NO-GO"))
    print(f"IMMUTABLE files n={res['immutable_files']['n']} {res['immutable_files']['status']}")
    print(f"runtime_sec total={res['runtime_sec']['total']:.1f} load_and_embed={res['runtime_sec']['load_and_embed']:.1f}"
          f" parent_maxrss_mb={res['parent_maxrss_mb']:.0f}")
    print("DISCLOSURE: " + res["disclosure"])


def main(argv=None):
    return run_guarded(_main, argv)


if __name__ == "__main__":
    sys.exit(main())
