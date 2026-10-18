"""STAGE 11 step A (TRAIN only): coverage check + label-independent prefix features (docs/PROTOCOL_STAGE11.md).

usage: .venv/bin/python -m tools.stage11_points --candidate C02 [--force]
Same inputs as stage 10 (pinned snapshot, train side via local_safe.select_split_units, units = content sha16,
positives = confirmed numeric intervals, negatives = provider-explicit no-violation clips; unknown tasks never loaded;
holdout never opened). Per clip one child (job via stdin, stderr DEVNULL, killed at the budget) reads the clip
sequentially with cv2; a frame's time is its PTS = CAP_PROP_POS_MSEC / 1000. It embeds, with the SAME pinned
SigLIP2 ONNX tower, preprocess and zone as the stage-5 cache (tools.stage5_encoder / stage5_probe.expand):
  - the PREFIX: every native decoded frame with PTS in [0, PREFIX_SEC) -- identical for every clip and chosen
    WITHOUT any label (the candidate/training extension of step B);
  - for positives only, a boolean per prefix frame: PTS inside the recorded [start, end] (never widened); label
    frames outside the prefix (none expected) are embedded separately so they are never lost;
  - the first 3 stage-5 cache-grid frames (drift check: cosine vs the existing cache rows).
Centring = stage 10 (L2-normalised, minus the clip's per-dimension median of its existing cache rows; label-free).
stdout carries aggregates only (counts, booleans, buckets); arrays go to <LP>/train_probe/stage11/ (0600) with a
sha16 manifest. Labels are used AS RECORDED: nothing is widened, shifted or reinterpreted.
"""
import argparse
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import stage5_encoder as ENC, stage8_audit as S8, stage9_head as S9  # noqa: E402
from tools.local_safe import Refused, run_guarded, select_split_units  # noqa: E402

PREFIX_SEC = 1.0          # fixed, label-independent native-frame prefix [0, 1.0 s)
N_DRIFT = 3
CLIP_BUDGET = 150.0
OUTDIR = "stage11"
MANIFEST = "manifest.json"


def lp():
    return S8.lp()


def sha16_bytes(b):
    return hashlib.sha256(b).hexdigest()[:16]


# ============================================================================ pure rules (self-tested)
def inside(t, ivs):
    """t inside any recorded [a, b] (closed); zero-width intervals are excluded."""
    t = np.asarray(t, float)
    m = np.zeros(t.shape, bool)
    for a, b in ivs:
        m |= (t >= a) & (t <= b) & (b > a)
    return m


def prefix_mask(pts, prefix=PREFIX_SEC):
    pts = np.asarray(pts, float)
    return np.isfinite(pts) & (pts >= 0.0) & (pts < prefix)


def fps_bucket(f):
    return "<20" if f < 20 else "20-26" if f < 26 else "26-31" if f < 31 else "31-61" if f < 61 else ">=61"


def dur_agree_bucket(d1, d2, fps):
    d = abs(d1 - d2)
    return "le_1frame" if d <= 1.0 / fps + 1e-9 else "le_0.5s" if d <= 0.5 else "le_1s" if d <= 1.0 else "gt_1s"


# ============================================================================ child: decode + embed ONE clip
def child_main():
    job = json.loads(sys.stdin.readline())
    try:
        r = _child(job)
    except Exception:  # noqa: BLE001 - static code only, no message text
        r = {"err": "failed"}
    sys.stdout.write(json.dumps(r) + "\n")
    return 0


def _child(job):
    import cv2
    import onnxruntime as ort
    if not ENC.onnx_ok():
        return {"err": "onnx_sha"}
    cap = cv2.VideoCapture(job["path"])
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n_prop = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not fps > 0:
        cap.release()
        return {"err": "no_fps"}
    ivs = [(float(a), float(b)) for a, b in job["ivs"]]
    keep_t = max([b for _a, b in ivs] + [PREFIX_SEC]) + N_DRIFT + 1.0
    pts, frames, idx, k, cache_idx, errs = [], {}, 0, 0, [], 0
    while cap.grab():
        tt = idx / fps                                # stage-5 cache time rule, used for the drift check only
        if tt <= keep_t:
            ok, fr = cap.retrieve()
            p = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0 if ok and fr is not None else float("nan")
            if ok and fr is not None:
                frames[idx] = fr
            else:
                errs += 1
            pts.append(p)
        if tt >= k + 0.5:
            if len(cache_idx) < N_DRIFT:
                cache_idx.append(idx)
            k = int(tt - 0.5) + 1
        idx += 1
    cap.release()
    dur_dec, dur_cont = idx / fps, (n_prop / fps if n_prop > 0 else float("nan"))
    pts = np.asarray(pts, float)
    pre = np.flatnonzero(prefix_mask(pts))
    lab_all = np.flatnonzero(inside(pts, ivs)) if ivs else np.zeros(0, int)
    lab_out = [int(i) for i in lab_all if i not in set(pre.tolist())]
    want = sorted(set(pre.tolist()) | set(lab_out) | set(cache_idx))
    if any(i not in frames for i in want):
        return {"err": "decode_missing_frame"}
    so = ort.SessionOptions()
    so.log_severity_level = 4
    sess = ort.InferenceSession(ENC.ONNX_PATH, so, providers=["CPUExecutionProvider"])
    e = np.concatenate([sess.run(None, {"pixel_values": ENC.preprocess([frames[i] for i in want[j:j + 16]],
                                                                        job["zone"])})[0]
                        for j in range(0, len(want), 16)])
    e = (e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)).astype(np.float16)
    E = dict(zip(want, e))
    D = e.shape[1]

    def mat(ix):
        return np.array([E[i] for i in ix], np.float16).reshape(len(ix), D)
    np.savez(job["out"], t_pre=pts[pre], E_pre=mat(pre.tolist()), pre_in_label=inside(pts[pre], ivs),
             t_lab_out=pts[lab_out] if lab_out else np.zeros(0), E_lab_out=mat(lab_out),
             E_drift=mat(cache_idx), fps=np.float64(fps), dur_dec=np.float64(dur_dec))
    inc = bool(np.all(np.diff(pts[np.isfinite(pts)]) > 0))
    return {"fps": fps, "dur_dec": dur_dec, "dur_cont": dur_cont, "n_prefix": int(len(pre)),
            "n_label_frames": int(len(lab_all)), "n_label_frames_outside_prefix": len(lab_out),
            "pts_strictly_increasing": inc, "retrieve_errors": int(errs),
            "idx_time_agrees_pts_all": bool(all(abs(pts[i] - i / fps) <= 0.5 / fps + 1e-6 for i in lab_all)),
            "finite": bool(np.isfinite(e.astype(np.float32)).all())}


def run_child(job, budget):
    p = subprocess.Popen([ENC.MAIN_PY, "-m", "tools.stage11_points", "--child"], cwd=ROOT, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    try:
        out, _ = p.communicate((json.dumps(job) + "\n").encode(), timeout=budget)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        p.wait()
        return {"err": "timeout"}
    try:
        return json.loads(out.decode().strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"err": "failed"}


# ============================================================================ parent
def zone_of(sha):
    from tools import stage5_probe as S5P
    z, _d4 = S5P.stage4_zone_dur(sha)
    return S5P.expand(z) if z else None


def centre_ref(sha):
    """-> (normalised cache rows, clip median): the stage-10 label-free centring reference of that clip."""
    with np.load(os.path.join(lp(), "train_probe", "cache_emb_siglip2", f"{sha}.npz"), allow_pickle=False) as ze:
        Ec = ze["E_zone"].astype(np.float64)
    Ec = Ec / np.maximum(np.linalg.norm(Ec, axis=1, keepdims=True), 1e-9)
    return Ec, np.median(Ec, 0)


def centre(E, med):
    E = np.asarray(E, np.float64).reshape(-1, len(med))
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9) - med


def _write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(path, 0o600)


def compute(cid, force=False):
    t0 = time.time()
    spath, ssha = S8.snapshot_spec(cid, None, None)
    latest = S8.read_snapshot(spath, ssha)
    pos, neg, _cnt = S8.train_sets(cid)
    if pos & neg:
        raise Refused("a content hash is both positive and negative")
    lab, _g = S9.supervision(latest, pos, neg)
    odir = os.path.join(lp(), "train_probe", OUTDIR)
    if os.path.exists(os.path.join(odir, MANIFEST)) and not force:
        raise Refused("stage 11 point features already exist (use --force to rebuild)")
    os.makedirs(odir, mode=0o700, exist_ok=True)
    for f in os.listdir(odir):                                      # only our own derived files
        if f.endswith(".npz") or f == MANIFEST:
            os.remove(os.path.join(odir, f))
    P, N = sorted(lab), sorted(neg)
    clips = [(s, 1) for s in P if S9.load_emb(s) is not None] + [(s, 0) for s in N if S9.load_emb(s) is not None]
    full, _c = select_split_units(lp(), cid, "train")              # train side only; sha re-checked after open
    paths = {s: p for s, p, _st, _e in full}
    rep = {"n_positive_units": len(P), "n_negative_units": len(N), "n_units_valid_cache": len(clips),
           "prefix_sec": PREFIX_SEC, "per_positive": {}}
    files, errs, drift, rows = {}, {}, [], {"pos_prefix": 0, "neg_prefix": 0, "neg_cache": 0}
    allz = []
    for s, y in clips:
        ivs = [(iv["start"], iv["end"]) for iv in lab[s]["ivs"]] if y else []
        if s not in paths:
            errs["not_selectable"] = errs.get("not_selectable", 0) + 1
            continue
        tmp = os.path.join(odir, f"{s}.tmp.npz")
        r = run_child({"path": paths[s], "ivs": ivs, "zone": zone_of(s), "out": tmp}, CLIP_BUDGET)
        if "err" not in r and (not r["finite"] or not r["pts_strictly_increasing"] or
                               r["retrieve_errors"] or not r["idx_time_agrees_pts_all"] or
                               r["n_label_frames_outside_prefix"] or not r["n_prefix"]):
            r = {"err": "feature_sanity_failed"}
        if "err" in r:
            errs[r["err"]] = errs.get(r["err"], 0) + 1
            if y:
                rep["per_positive"][f"P{P.index(s) + 1}"] = {"decode_ok": False}
            continue
        with np.load(tmp) as z:
            Ep, tp, inl, Eo, to, Ed = z["E_pre"], z["t_pre"], z["pre_in_label"], z["E_lab_out"], z["t_lab_out"], \
                z["E_drift"]
        os.remove(tmp)
        Ec, med = centre_ref(s)
        dn = centre(Ed, np.zeros(Ec.shape[1]))
        drift += [float(dn[i] @ Ec[i]) for i in range(min(len(dn), len(Ec)))]
        rec = {"y": np.int8(y), "t_pre": tp, "z_pre": centre(Ep, med), "pre_in_label": inl,
               "t_lab_out": to, "z_lab_out": centre(Eo, med)}
        allz += [rec["z_pre"], rec["z_lab_out"]]
        buf = io.BytesIO()
        np.savez(buf, **rec)
        _write(os.path.join(odir, f"{s}.feat.npz"), buf.getvalue())
        files[f"{s}.feat.npz"] = sha16_bytes(buf.getvalue())
        rows["pos_prefix" if y else "neg_prefix"] += len(tp)
        if not y:
            rows["neg_cache"] += len(Ec)
        else:
            rep["per_positive"][f"P{P.index(s) + 1}"] = {
                "decode_ok": True, "width_positive": [b > a for a, b in ivs],
                "bounds_ok_0_le_start_le_end_le_dur": [0 <= a <= b <= r["dur_dec"] for a, b in ivs],
                "fps_bucket": fps_bucket(r["fps"]), "pts_strictly_increasing": r["pts_strictly_increasing"],
                "container_vs_decoded_duration": dur_agree_bucket(r["dur_dec"], r["dur_cont"], r["fps"])
                if np.isfinite(r["dur_cont"]) else "container_unknown",
                "n_prefix_frames": r["n_prefix"], "n_frames_pts_inside_label": r["n_label_frames"],
                "n_prefix_frames_inside_label": int(inl.sum()),
                "n_label_frames_outside_prefix": r["n_label_frames_outside_prefix"],
                "idx_time_agrees_pts_all": r["idx_time_agrees_pts_all"], "retrieve_errors": r["retrieve_errors"]}
    blocked = [f"P{k + 1}" for k in range(len(P))
               if not rep["per_positive"].get(f"P{k + 1}", {}).get("n_frames_pts_inside_label")]
    man = {"version": 2, "cid": cid, "snapshot_sha16": ssha[:16], "onnx_sha16": ENC.ONNX_SHA256[:16],
           "code_sha16": S8.fsha(os.path.abspath(__file__))[:16], "prefix_sec": PREFIX_SEC, "files": files,
           "rule": "every native frame with PTS in [0, prefix) (label-independent) + label frames outside the prefix; "
                   "PTS = CAP_PROP_POS_MSEC/1000; stage-10 clip-median centring on the existing cache"}
    mb = json.dumps(man, sort_keys=True, indent=1).encode()
    _write(os.path.join(odir, MANIFEST), mb)
    Z = np.vstack([z for z in allz if z.size]) if allz else np.zeros((0, 0))
    rep.update({"outcome": "STOP_FEATURE_SANITY" if (blocked or errs or len(files) != 14 or len(P) != 2 or len(N) != 12 or not np.isfinite(Z).all() or not drift or min(drift) < 0.999) else "OK", "blocked": blocked,
                "rows": rows, "clips_with_features": len(files), "decode_errors": errs,
                "all_finite": bool(np.isfinite(Z).all()), "feature_dim": int(Z.shape[1]) if Z.ndim == 2 else 0,
                "drift_cos_vs_cache_min": round(min(drift), 5) if drift else None, "drift_n": len(drift),
                "manifest_sha16": sha16_bytes(mb), "runtime_s": round(time.time() - t0, 1)})
    return rep


def _main(argv=None):
    ap = argparse.ArgumentParser(description="stage 11 step A: coverage + prefix features (aggregates only)")
    ap.add_argument("--candidate", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.child:
        return child_main()
    if not ENC.onnx_ok():
        raise Refused("pinned ONNX vision tower missing or sha256 mismatch")
    rep = compute(a.candidate, a.force)
    print("STAGE11A " + json.dumps(rep, sort_keys=True, default=float))
    return 0 if rep["outcome"] == "OK" else 3


def main(argv=None):
    return run_guarded(_main, argv)


if __name__ == "__main__":
    sys.exit(main())
