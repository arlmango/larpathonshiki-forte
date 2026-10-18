"""STAGE 7 (TRAIN only): within-clip contrast against unlabeled background (WCR), two pre-registered families.

usage: python -m tools.stage7_wcr --candidate C02 --n-perm 1000 [--budget-total-sec 1800] [--mem-mb 4096]
Implements docs/PROTOCOL_STAGE7.md (its sha256-16 is printed). Units only via
local_safe.select_split_units(side "train", dry_run=True): NO clip file is opened; only existing numeric caches are
read (stage-4 `<LP>/train_probe/cache/<sha16>_4.npz`, stage-5 `<LP>/train_probe/cache_emb_siglip2/<sha16>.npz`).
A missing/invalid cache = failed unit for that family (counted, never no_violation; no GO for that family).
The computation runs in a child (job via stdin, stderr DEVNULL, RSS watchdog --mem-mb); the parent writes
<LP>/train_probe/stage7_summary.json (mode 0600) and prints AGGREGATES ONLY (no shas, paths, per-clip rows/times).
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import resource
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import stage4_probe as S4, train_probe as TP  # noqa: E402
from tools.local_safe import Refused, run_guarded, select_split_units  # noqa: E402
from tools.stage7_core import (ALPHA, MINV, Family, evaluate_family, mot_timeline,  # noqa: E402,F401
                               nested, prep)

PROTOCOL = os.path.join(ROOT, "docs", "PROTOCOL_STAGE7.md")
SEED = 0
FAMILIES = ("A_sem", "A_mot")
TEXT_SHA16 = "e5c2ba8d958bfc2a"                    # stage-5 text_probes.npy (pre-check only)


def lp():
    return TP.LP


# ============================================================================ loading (no clip file opened)
def load(cid):
    units, cnt = select_split_units(lp(), cid, "train", dry_run=True)
    lab = [(sha, int(st == "reviewed_violation"), [float(v) for v in ev]) for sha, _p, st, ev in units
           if st in ("reviewed_violation", "reviewed_no_violation")]
    U, st = [], collections.Counter()
    for sha, y, ev in lab:
        u = {"y": y, "tl": ev if y else [], "sem": None, "mot": None, "dur": None, "Efull": None}
        try:
            z4 = np.load(os.path.join(lp(), "train_probe", "cache", f"{sha}_4.npz"))
            u["dur"] = float(z4["dur"])
            m = mot_timeline(z4)
            if m is not None and len(m[0]) >= MINV:
                u["mot"] = m
            else:
                st["mot_zone_unavailable"] += 1
        except (OSError, ValueError, KeyError):
            st["stage4_cache_missing_or_invalid"] += 1
        try:
            ze = np.load(os.path.join(lp(), "train_probe", "cache_emb_siglip2", f"{sha}.npz"))
            t = ze["t"].astype(np.float64)
            E = ze["E_zone"].astype(np.float64)
            E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
            if len(t) >= MINV and np.isfinite(E).all():
                u["sem"] = (t, E)
                Ef = ze["E_full"].astype(np.float64)
                u["Efull"] = Ef / np.maximum(np.linalg.norm(Ef, axis=1, keepdims=True), 1e-9)
                st["zone_fallback_full_frame"] += int(ze["zone_fallback"])
                if u["dur"] is None:
                    u["dur"] = float(ze["dur_dec"])
            else:
                st["emb_cache_invalid"] += 1
        except (OSError, ValueError, KeyError):
            st["emb_cache_missing_or_invalid"] += 1
        U.append(u)
    cov = {"label_units_selected": len(lab), "skipped_by_selector": cnt["skipped"] + (len(units) - len(lab)),
           **dict(st)}
    return U, cov


def eta2(groups):
    """share of total variance explained by clip id, per column -> (median %, max %)."""
    X = np.vstack(groups)
    mu = X.mean(0)
    tot = ((X - mu) ** 2).sum(0)
    btw = sum(len(g) * (g.mean(0) - mu) ** 2 for g in groups)
    e = btw / np.maximum(tot, 1e-12)
    return [float(100 * np.median(e)), float(100 * np.max(e))]


def precheck(U):
    """label-free scene-identity diagnostic (protocol §8): clip-id share of variance (aggregate %)."""
    out = {}
    sem = [u for u in U if u["sem"] is not None]
    for name, get in (("E_zone", lambda u: u["sem"][1]), ("E_full", lambda u: u["Efull"])):
        if len(sem) >= 2:
            X = np.vstack([get(u) for u in sem]); mu = X.mean(0)
            V = np.linalg.svd(X - mu, full_matrices=False)[2][:8].T
            out[f"{name}_pca8_eta2_pct_median_max"] = eta2([(get(u) - mu) @ V for u in sem])
            cn = [np.linalg.norm(get(u) - np.median(get(u), 0), axis=1)[:, None] for u in sem]
            out[f"{name}_centred_norm_eta2_pct_median_max"] = eta2(cn)  # heteroskedasticity left after centring
    tp = os.path.join(ROOT, "weights", "siglip2_base_patch16_224", "onnx", "text_probes.npy")
    if sem and os.path.exists(tp) and S4.sha256_or_none(tp)[:16] == TEXT_SHA16:
        T = np.load(tp)
        out["prompt_cos_eta2_pct_median_max"] = eta2([u["sem"][1] @ T.T for u in sem])
    else:
        out["prompt_cos_eta2_pct_median_max"] = "not_computed_text_probes_absent_or_unpinned"
    mot = [u["mot"][1] for u in U if u["mot"] is not None]
    if len(mot) >= 2:
        out["A_mot_raw_eta2_pct_median_max"] = eta2(mot)
    return out


def compute(cid, n_perm, deadline):
    U, cov = load(cid)
    n_lab = cov["label_units_selected"]
    pre = precheck(U)
    res = {"coverage": cov, "precheck_label_free": pre, "families": {}}
    for fi, fam in enumerate(FAMILIES):
        key = "sem" if fam == "A_sem" else "mot"
        V = [u for u in U if u[key] is not None and u["dur"]]
        complete = len(V) == n_lab == 17 and cov["skipped_by_selector"] == 0
        if len(V) < 6 or len({u["y"] for u in V}) < 2:
            res["families"][fam] = {"status": "refused_too_few_units", "units": len(V), "criteria": {"family_GO": 0}}
            continue
        Fm = Family([prep(*u[key]) for u in V])
        y = [u["y"] for u in V]; tl = [u["tl"] for u in V]; dur = [u["dur"] for u in V]
        r = evaluate_family(Fm, y, tl, dur, n_perm, SEED + fi, deadline, complete)
        r.update(units=len(V), pos=int(sum(y)), failed_units=n_lab - len(V), dims=int(Fm.P[0]["z"].shape[1]),
                 valid_steps=int(sum(len(z) for z in Fm.Z)))
        res["families"][fam] = r
    d = [u["dur"] for u in U if u["dur"]]
    res["coverage"].update(units=len(U), pos=int(sum(u["y"] for u in U)),
                           duration_sec_min_med_max=[float(min(d)), float(np.median(d)), float(max(d))] if d else None)
    res["decision_GO"] = int(any(f["criteria"]["family_GO"] for f in res["families"].values()))
    res["child_maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    return res


# ============================================================================ child + watchdog
def child_main():
    job = json.loads(sys.stdin.readline())
    if job.get("mode") == "memtest":
        held = []
        for _ in range(int(job["total_mb"]) // 25):
            b = bytearray(25 << 20); b[::4096] = b"x" * len(b[::4096]); held.append(b); time.sleep(0.2)
        time.sleep(2.0)
        print(json.dumps({"memtest_done": 1}))
        return 0
    try:
        r = compute(job["cid"], int(job["n_perm"]), float(job["deadline"]))
    except Refused:
        r = {"err": "refused"}
    except (OSError, ValueError, KeyError, RuntimeError):
        r = {"err": "failed"}
    print(json.dumps(r, default=float))
    return 0


def _rss_tree_mb(pid):
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-g", str(pid)], capture_output=True, text=True, timeout=5).stdout
        return sum(int(x) for x in out.split()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def run_watched(job, budget, mem_mb):
    """same pattern as stage5_encoder.run_watched: RSS of the child's process group polled every 0.25 s."""
    p = subprocess.Popen([sys.executable, "-m", "tools.stage7_wcr", "--child"], cwd=ROOT, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    buf = []
    th = threading.Thread(target=lambda: buf.append(p.stdout.read()), daemon=True)
    th.start()
    p.stdin.write((json.dumps(job) + "\n").encode()); p.stdin.close()
    t0, status, peak_mb = time.time(), None, 0.0
    while p.poll() is None:
        used = _rss_tree_mb(p.pid); peak_mb = max(peak_mb, used)
        status = "memory_cap" if used > mem_mb else ("timeout" if time.time() - t0 > budget else None)
        if status:
            try:
                os.killpg(p.pid, 9)
            except (ProcessLookupError, PermissionError):
                pass
            p.wait()
            break
        time.sleep(0.25)
    th.join(5)
    r = {}
    if status is None:
        try:
            r = json.loads((buf[0] if buf else b"").decode().strip().splitlines()[-1])
        except (ValueError, IndexError):
            r = {}
        status = "ok" if r and "err" not in r and p.returncode == 0 else r.get("err", "failed")
    return r, status, peak_mb


def immutable_digest():
    files = sorted(glob.glob(os.path.join(lp(), "labels", "*"))) + [
        os.path.join(lp(), "inventory_map.json")] + [
        os.path.join(lp(), "train_probe", f"stage{s}_summary.json") for s in (4, 5)]
    h = hashlib.sha256()
    for p in files:
        h.update(os.path.basename(p).encode())
        try:
            with open(p, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
        except OSError:
            h.update(b"absent")
    return h.hexdigest(), len(files)


DISCLOSURE = ("stage 7 = the single pre-registered attempt; third adaptive look at the same train sample after "
              "stages 4-5; exploratory; audit-informed design; no independent validation; holdout seen once in "
              "stage 1, unopened here and closed even under GO; production stays fail-closed")


def _main(argv=None):
    ap = argparse.ArgumentParser(description="stage 7 WCR train-only probe (aggregates only)")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--n-perm", type=int, required=True)
    ap.add_argument("--budget-total-sec", type=float, default=1800.0)
    ap.add_argument("--mem-mb", type=int, default=int(os.environ.get("PILOT_MEM_MB", "4096")))
    ap.add_argument("--memtest", type=int, default=0, help="bounded allocation MB on the real child entry point")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    if a.memtest:
        r, status, pk = run_watched({"mode": "memtest", "total_mb": min(a.memtest, 400)}, 120, a.mem_mb)
        print(f"MEMTEST cap {a.mem_mb} MB alloc {min(a.memtest, 400)} MB: status={status} peak_seen_mb={pk:.0f}")
        return 0 if status == "memory_cap" else 1
    proto = (S4.sha256_or_none(PROTOCOL) or "missing")[:16]
    h0 = immutable_digest()
    deadline = t0 + a.budget_total_sec - 20
    r, status, pk = run_watched({"cid": a.candidate, "n_perm": a.n_perm, "deadline": deadline},
                                a.budget_total_sec, a.mem_mb)
    if status != "ok":
        print(f"refused/failed: child status {status}", file=sys.stderr)
        return 2
    h1 = immutable_digest()
    r.update(protocol="PROTOCOL_STAGE7", protocol_sha256_16=proto, child_status=status, child_peak_rss_mb=pk,
             runtime_sec=time.time() - t0, alpha_per_family_bonferroni=ALPHA, disclosure=DISCLOSURE,
             immutable_files={"n": h0[1], "status": "unchanged" if h0 == h1 else "CHANGED"})
    out = os.path.join(lp(), "train_probe", "stage7_summary.json")
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(r, f, indent=1, default=float)
    _print(r)
    return 0


def _f(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _print(r):
    print(f"protocol_doc_sha256_16={r['protocol_sha256_16']} alpha/family={r['alpha_per_family_bonferroni']}")
    print("COVERAGE " + json.dumps(r["coverage"]))
    print("PRECHECK (label-free, clip-id variance share %) " + json.dumps(r["precheck_label_free"]))
    for fam, f in r["families"].items():
        if "outer" not in f:
            print(f"\n{fam}: {f.get('status')}")
            continue
        c, e = f["outer"]["clip"], f["outer"]["event_5s"]
        print(f"\n{fam} (units={f['units']}, pos={f['pos']}, failed={f['failed_units']}, dims={f['dims']})")
        print(f"  clip TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']} F1={_f(f['clip_f1'])} | event±5s "
              f"TP={e['tp']} FP={e['fp']} FN={e['fn']} F1={_f(f['event_f1'])} | hit={_f(f['hit_timestamp_endpoint'])} "
              f"MAE={_f(f['mae_sec'])}s")
        print(f"  always-positive F1={_f(f['always_positive_f1'])} duration-only F1="
              f"{_f(f['duration_only_nested']['clip_f1'])} random-time u/s={_f(f['random_time_uniform'])}/"
              f"{_f(f['random_time_support'])} position-prior " + json.dumps(f["position_prior"]))
        for k in ("f1", "hit", "event_f1"):
            q = f["perm_N2"][k]
            print(f"  N2 perm ({f['perm_N2']['n_completed']}/{f['perm_N2']['n']}) {k:8s} p50={_f(q.get('p50'))} "
                  f"p95={_f(q.get('p95'))} p_perm={_f(q['p_perm'])}")
        print("  N1 transplant " + json.dumps(f["transplant_N1"]))
        print(f"  |rho| median/max={f['abs_rho_inner_scores_vs_duration_median_max']} nested_s={f['nested_run_sec']:.3f}"
              f" | DIAGNOSTIC signed rho median all/negatives-only="
              f"{f['DIAGNOSTIC_signed_rho_median_all_and_negatives_only']}")
        print("  CRITERIA " + json.dumps(f["criteria"]))
    print("\nDECISION_GO=" + str(r["decision_GO"]) + (" -> exploratory GO (holdout stays closed)" if r["decision_GO"]
                                                       else " -> NO-GO"))
    print(f"IMMUTABLE n={r['immutable_files']['n']} {r['immutable_files']['status']}; runtime_sec="
          f"{r['runtime_sec']:.1f}; child_peak_rss_mb={r['child_peak_rss_mb']:.0f}")
    print("DISCLOSURE: " + r["disclosure"])


def main(argv=None):
    if (sys.argv[1:] if argv is None else argv) == ["--child"]:
        return child_main()
    return run_guarded(_main, argv)


if __name__ == "__main__":
    sys.exit(main())
