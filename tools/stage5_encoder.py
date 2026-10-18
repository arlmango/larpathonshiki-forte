"""STAGE 5 (research): frozen SigLIP2 image encoder -- export, parity and budget SMOKE on SYNTHETIC frames only.

usage: python -m tools.stage5_encoder smoke [--n-frames 48] | memtest [--mem-mb 256 --alloc-mb 300]  (main .venv)
Model: google/siglip2-base-patch16-224 @ 75de2d55 (card `license: apache-2.0`), loaded natively by transformers
5.20.1 (`model_type: siglip`, no auto_map, trust_remote_code never set, local_files_only=True). Fetched with
SHA256 by `python tools/fetch_weights.py siglip2_base_patch16_224`.
Steps (each in a CHILD process, job via stdin, stderr DEVNULL, numeric-only JSON on stdout):
  1. research venv (torch 2.14.1): load, text-prompt embeddings, ms/frame on CPU and MPS, ONNX export of the
     vision tower (pooled image embedding) to weights/siglip2_base_patch16_224/onnx/vision.onnx, parity probe set;
  2. main venv (onnxruntime 1.31.0): ms/frame on CPUExecutionProvider and CoreMLExecutionProvider, parity
     (cosine to the torch embedding of the same synthetic frames) and cv2-vs-HF preprocessing parity.
Frames come from tools.make_synthetic only (written into a scratch LOCAL_PRIVATE_DIR-independent temp dir).
Nothing real is opened here. Output: aggregates (ms, MB, cosines) only.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESEARCH_PY = os.path.join(ROOT, ".venv-research", "bin", "python")
MAIN_PY = os.path.join(ROOT, ".venv", "bin", "python")
MODEL_DIR = os.path.join(ROOT, "weights", "siglip2_base_patch16_224")
ONNX_PATH = os.path.join(MODEL_DIR, "onnx", "vision.onnx")
# pinned derived artifact (exported once at Part A from the pinned weights; never re-exported over)
ONNX_SHA256 = "4e95b080c23eff44205bd7fc87cc7818f1d613c559d8380de204db82374ffdba"
SIZE = 224
# Pre-registered text probes (PROTOCOL_STAGE5 §2, family T). Fixed BEFORE any real-data computation.
PROMPTS = (
    "a cashier counting banknotes at a counter",
    "a hand holding paper money",
    "a person handing money across a counter",
    "an open cash drawer",
    "a person putting money into a pocket",
    "a person holding a mobile phone",
    "two people behind a cashier counter",
    "an empty cashier workplace with nobody at the desk",
)


def preprocess(bgr_frames, box=None):
    """BGR uint8 frames -> float32 NCHW in [-1, 1] (SigLIP: resize 224x224, rescale 1/255, mean=std=0.5).
    box = normalized (x0, y0, x1, y1) crop applied first (zone view); INTER_AREA ~ antialiased bilinear."""
    import cv2
    out = []
    for fr in bgr_frames:
        if box is not None:
            h, w = fr.shape[:2]
            x0, y0 = int(max(0, box[0]) * w), int(max(0, box[1]) * h)
            x1, y1 = int(min(1, box[2]) * w), int(min(1, box[3]) * h)
            fr = fr[y0:max(y1, y0 + 2), x0:max(x1, x0 + 2)]
        im = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        out.append(im.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0)
    return np.stack(out).astype(np.float32)


def synthetic_frames(n, tmpdir):
    """n frames at 1 fps from a make_synthetic clip (no real data)."""
    import cv2
    from tools.make_synthetic import make
    p = os.path.join(tmpdir, "syn.mp4")
    make(p, dur=max(n, 2) + 1, fps=10, size=(1280, 720), two=(n / 3, 2 * n / 3))
    cap = cv2.VideoCapture(p)
    fr, k = [], 0
    while len(fr) < n:
        ok, f = cap.read()
        if not ok:
            break
        if k % 10 == 0:
            fr.append(f)
        k += 1
    cap.release()
    return fr


def _maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20  # macOS: bytes


def _bench(fn, x, bs, reps=1):
    fn(x[:bs])  # warm-up
    t = time.perf_counter()
    for _ in range(reps):
        for i in range(0, len(x), bs):
            fn(x[i:i + bs])
    return (time.perf_counter() - t) * 1000 / (reps * len(x))


# ============================================================================ child: research venv (torch)
def child_torch(job):
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
    torch.set_num_threads(os.cpu_count() or 4)
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    load_s = time.time() - t0
    res = {"load_s": load_s, "model_class": type(model).__name__, "n_params_vision_M":
           sum(p.numel() for p in model.vision_model.parameters()) / 1e6, "rss_after_load_mb": _maxrss_mb()}

    def img_feats(x):
        o = model.get_image_features(pixel_values=x)
        return o if torch.is_tensor(o) else o.pooler_output

    with tempfile.TemporaryDirectory() as td:
        frames = synthetic_frames(job["n"], td)
    x = torch.from_numpy(preprocess(frames))
    with torch.no_grad():
        t1 = time.time()
        ti = tok(list(PROMPTS), padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        te = model.get_text_features(**ti)
        te = te if torch.is_tensor(te) else te.pooler_output
        res["text_s"] = time.time() - t1
        e_cpu = img_feats(x)
        res["emb_dim"] = int(e_cpu.shape[1])
        for bs in (1, 8):
            res[f"torch_cpu_ms_per_frame_bs{bs}"] = _bench(lambda b: img_feats(b), x, bs)
        # HF processor parity (PIL path) vs our cv2 preprocessing, cosine of embeddings
        try:
            from PIL import Image
            proc = AutoImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
            import cv2
            pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames[:8]]
            e_hf = img_feats(proc(images=pil, return_tensors="pt")["pixel_values"])
            res["cos_cv2_vs_hf_preproc_min"] = float(torch.nn.functional.cosine_similarity(e_hf, e_cpu[:8]).min())
        except Exception as e:  # noqa: BLE001 - class name only
            res["hf_preproc_error"] = type(e).__name__
        # zero-shot sanity on synthetic frames: sigmoid(logit_scale*cos + bias), mean over frames (numbers only)
        ei = torch.nn.functional.normalize(e_cpu, dim=-1)
        tn = torch.nn.functional.normalize(te, dim=-1)
        lg = ei @ tn.T * model.logit_scale.exp() + model.logit_bias
        res["probe_sigmoid_mean_range"] = [float(torch.sigmoid(lg).mean(0).min()), float(torch.sigmoid(lg).mean(0).max())]
        if torch.backends.mps.is_available():
            try:
                mm = model.to("mps")
                xm = x.to("mps")

                def f_mps(b):
                    o = mm.get_image_features(pixel_values=b)
                    o = o if torch.is_tensor(o) else o.pooler_output
                    torch.mps.synchronize()
                    return o
                for bs in (1, 8):
                    res[f"torch_mps_ms_per_frame_bs{bs}"] = _bench(f_mps, xm, bs)
                res["cos_mps_vs_cpu_min"] = float(torch.nn.functional.cosine_similarity(
                    f_mps(xm).cpu(), e_cpu).min())
                model = mm.to("cpu")
            except Exception as e:  # noqa: BLE001
                res["mps_error"] = type(e).__name__
        # ONNX export of the pooled vision embedding
        class V(torch.nn.Module):
            def __init__(s, m):
                super().__init__()
                s.m = m

            def forward(s, pixel_values):
                o = s.m.get_image_features(pixel_values=pixel_values)
                return o if torch.is_tensor(o) else o.pooler_output
        os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
        t2 = time.time()
        res["onnx_export"] = "kept_existing" if os.path.exists(ONNX_PATH) else "failed"
        for dyn in (() if os.path.exists(ONNX_PATH) else (False, True)):
            try:
                kw = {"dynamo": dyn}
                if dyn:
                    kw["dynamic_shapes"] = {"pixel_values": {0: torch.export.Dim("batch", min=1, max=64)}}
                else:
                    kw["dynamic_axes"] = {"pixel_values": {0: "batch"}, "emb": {0: "batch"}}
                torch.onnx.export(V(model).eval(), (x[:2],), ONNX_PATH, input_names=["pixel_values"],
                                  output_names=["emb"], opset_version=18, **kw)
                res["onnx_export"] = "dynamo" if dyn else "torchscript"
                break
            except Exception as e:  # noqa: BLE001
                res[f"onnx_export_error_{'dynamo' if dyn else 'ts'}"] = type(e).__name__
        res["onnx_export_s"] = time.time() - t2
        np.save(job["ref"], e_cpu.numpy().astype(np.float32))
        np.save(job["text"], te.numpy().astype(np.float32))
    res["maxrss_mb"] = _maxrss_mb()
    return res


# ============================================================================ child: main venv (onnxruntime)
def child_ort(job):
    import onnxruntime as ort
    with tempfile.TemporaryDirectory() as td:
        frames = synthetic_frames(job["n"], td)
    x = preprocess(frames)
    ref = np.load(job["ref"])
    res = {}
    for prov in ("CPUExecutionProvider", "CoreMLExecutionProvider"):
        if prov not in ort.get_available_providers():
            continue
        try:
            t0 = time.time()
            so = ort.SessionOptions()
            so.log_severity_level = 4
            s = ort.InferenceSession(ONNX_PATH, so, providers=[prov, "CPUExecutionProvider"])
            k = prov[:-17].lower()
            res[f"ort_{k}_load_s"] = time.time() - t0

            def run(b):
                return s.run(None, {"pixel_values": b})[0]
            for bs in (1, 8):
                res[f"ort_{k}_ms_per_frame_bs{bs}"] = _bench(run, x, bs)
            e = run(x)
            cos = (e * ref).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(ref, axis=1))
            res[f"ort_{k}_cos_vs_torch_min"] = float(cos.min())
        except Exception as e:  # noqa: BLE001
            res[f"ort_{prov[:-17].lower()}_error"] = type(e).__name__
    res["maxrss_mb"] = _maxrss_mb()
    return res


# ============================================================================ child: main venv, REAL embedding job
def onnx_ok():
    """True iff the pinned ONNX vision tower is present with the recorded sha256 (never overwritten)."""
    import hashlib
    if not os.path.exists(ONNX_PATH):
        return False
    h = hashlib.sha256()
    with open(ONNX_PATH, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest() == ONNX_SHA256


def child_embed(job):
    """ONE clip: decode at 1 fps (t_k = k + 0.5 s), embed full + zone views with ORT CPU, write npz.
    mode 'memtest' = bounded synthetic allocation in this SAME entry point (proves the watchdog kill path)."""
    if job.get("mode") == "memtest":
        held = []
        for _ in range(int(job["total_mb"]) // int(job["step_mb"])):
            b = bytearray(int(job["step_mb"]) << 20)
            b[::4096] = b"x" * len(b[::4096])  # touch pages so RSS grows
            held.append(b)
            time.sleep(0.3)
        time.sleep(2.0)
        return {"memtest_done": 1}
    import cv2
    import onnxruntime as ort
    if not onnx_ok():
        return {"err": "onnx_sha"}
    so = ort.SessionOptions()
    so.log_severity_level = 4
    sess = ort.InferenceSession(ONNX_PATH, so, providers=["CPUExecutionProvider"])
    t0 = time.time()
    cap = cv2.VideoCapture(job["path"])
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n_prop = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not fps > 0:
        cap.release()
        return {"err": "decode_incomplete"}
    ts, frames, idx, k, broken = [], [], 0, 0, False
    while cap.grab():
        if idx / fps >= k + 0.5:
            ok, fr = cap.retrieve()
            if not ok or fr is None:
                broken = True
                break
            ts.append(idx / fps)
            frames.append(fr)
            k = int(idx / fps - 0.5) + 1
        idx += 1
    cap.release()
    dec_s = time.time() - t0
    # FAIL-CLOSED completeness: expected samples from the probe/inventory duration (else container frame count)
    exp_dur = float(job.get("dur_expected") or (n_prop / fps))
    exp_n = max(0, int(np.ceil(exp_dur - 0.5)))
    if broken or len(frames) < max(3, exp_n - 1) or abs(idx / fps - exp_dur) > 1.0:
        return {"err": "decode_incomplete", "n_frames": len(frames), "n_expected": exp_n}
    t1 = time.time()
    zone = job.get("zone")

    def emb(box):
        out = []
        for i in range(0, len(frames), 16):
            out.append(sess.run(None, {"pixel_values": preprocess(frames[i:i + 16], box)})[0])
        e = np.concatenate(out)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    ef = emb(None)
    ez = emb(zone) if zone is not None else ef
    np.savez_compressed(job["out"], t=np.array(ts, np.float32), E_full=ef.astype(np.float16),
                        E_zone=ez.astype(np.float16), zone_fallback=np.int8(zone is None),
                        dur_dec=np.float32(idx / fps))
    return {"n_frames": len(frames), "n_src_frames": idx, "decode_s": dec_s, "embed_s": time.time() - t1,
            "maxrss_mb": _maxrss_mb()}


def child_text(job):
    """research venv: the 8 pre-registered prompts -> L2-normalised text embeddings (8, 768)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    with torch.no_grad():
        ti = tok(list(PROMPTS), padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        te = model.get_text_features(**ti)
        te = (te if torch.is_tensor(te) else te.pooler_output).numpy().astype(np.float32)
    np.save(job["out"], te / np.linalg.norm(te, axis=1, keepdims=True))
    return {"ok": 1}


TEXT_PATH = os.path.join(MODEL_DIR, "onnx", "text_probes.npy")


def text_probes():
    """Cached (8, 768) prompt embeddings; computed once in the research venv if missing."""
    if not os.path.exists(TEXT_PATH):
        r = _spawn(RESEARCH_PY, "text", {"out": TEXT_PATH}, 300)
        if r.get("status") != "ok":
            raise RuntimeError("text probes")
    return np.load(TEXT_PATH)


def _rss_tree_mb(pid):
    """RSS of the child's whole process group (ps), MB."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-g", str(pid)], capture_output=True, text=True, timeout=5).stdout
        return sum(int(x) for x in out.split()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def run_watched(py, mode, job, budget, mem_mb):
    """Spawn a child (job via stdin, stderr DEVNULL) and ENFORCE mem_mb by polling the RSS of its process group
    every 0.25 s; > mem_mb -> SIGKILL of the group -> status 'memory_cap'; wall > budget -> 'timeout'.
    -> (result dict incl. status, peak_seen_mb)."""
    import threading
    p = subprocess.Popen([py, "-m", "tools.stage5_encoder", "--child", mode], cwd=ROOT, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    buf = []
    th = threading.Thread(target=lambda: buf.append(p.stdout.read()), daemon=True)
    th.start()
    try:
        p.stdin.write((json.dumps(job) + "\n").encode())
        p.stdin.close()
    except OSError:
        pass
    t0, status, peak = time.time(), None, 0.0
    while p.poll() is None:
        used = _rss_tree_mb(p.pid)
        peak = max(peak, used)
        if used > mem_mb:
            status = "memory_cap"
        elif time.time() - t0 > budget:
            status = "timeout"
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
        status = (r["err"] if r.get("err") == "decode_incomplete" else "failed") if "err" in r or not r \
            or p.returncode != 0 else "ok"
    r.update(status=status, wall_s=time.time() - t0, peak_seen_mb=peak)
    return r


def memtest(mem_mb, total_mb):
    """Lowered-limit kill test on the REAL embedding child entry point (bounded allocation)."""
    r = run_watched(MAIN_PY, "embed", {"mode": "memtest", "total_mb": min(total_mb, 400), "step_mb": 25}, 120, mem_mb)
    left = subprocess.run(["pgrep", "-f", "tools.stage5_encoder --child"], capture_output=True, text=True).stdout.split()
    print(f"MEMTEST cap {mem_mb} MB, bounded allocation {min(total_mb, 400)} MB in 25 MB steps, child entry 'embed': "
          f"status={r['status']} peak_seen_mb={r['peak_seen_mb']:.0f} leftover_children={len(left)}")
    return 0 if r["status"] == "memory_cap" and not left else 1


def _spawn(py, mode, job, budget):
    t0 = time.time()
    p = subprocess.Popen([py, "-m", "tools.stage5_encoder", "--child", mode], cwd=ROOT, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    try:
        out, _ = p.communicate((json.dumps(job) + "\n").encode(), timeout=budget)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        p.wait()
        return {"status": "timeout"}
    try:
        r = json.loads(out.decode().strip().splitlines()[-1])
    except (ValueError, IndexError):
        r = {}
    r["status"] = "ok" if p.returncode == 0 and r else f"failed_rc{p.returncode}"
    r["wall_s"] = time.time() - t0
    return r


def smoke(n, budget):
    if not os.path.exists(RESEARCH_PY) or not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
        raise SystemExit("refused: research venv or SigLIP2 weights missing")
    with tempfile.TemporaryDirectory() as td:
        job = {"n": n, "ref": os.path.join(td, "ref.npy"), "text": os.path.join(td, "text.npy")}
        rt = _spawn(RESEARCH_PY, "torch", job, budget)
        ro = _spawn(MAIN_PY, "ort", job, budget) if rt.get("onnx_export", "failed") != "failed" else {"status": "skipped"}
    print("SMOKE torch (research venv) " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                        for k, v in rt.items()}))
    print("SMOKE onnxruntime (main venv) " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                         for k, v in ro.items()}))
    if os.path.exists(ONNX_PATH):
        print(f"onnx file MB {os.path.getsize(ONNX_PATH) / 2 ** 20:.1f} (+ external data if any: "
              f"{sum(os.path.getsize(os.path.join(os.path.dirname(ONNX_PATH), f)) for f in os.listdir(os.path.dirname(ONNX_PATH))) / 2 ** 20:.1f} MB dir)")
    return 0 if rt.get("status") == "ok" else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--child"]:
        job = json.loads(sys.stdin.readline())
        r = {"torch": child_torch, "ort": child_ort, "embed": child_embed, "text": child_text}[argv[1]](job)
        sys.stdout.write(json.dumps(r) + "\n")
        return 0
    from tools.local_safe import run_guarded

    def _m(av):
        ap = argparse.ArgumentParser(description="stage-5 encoder smoke (synthetic only, aggregates only)")
        ap.add_argument("cmd", choices=("smoke", "memtest"))
        ap.add_argument("--mem-mb", type=int, default=int(os.environ.get("PILOT_MEM_MB", "8192")))
        ap.add_argument("--alloc-mb", type=int, default=300)
        ap.add_argument("--n-frames", type=int, default=48)
        ap.add_argument("--budget-sec", type=float, default=900.0)
        a = ap.parse_args(av)
        if a.cmd == "memtest":
            return memtest(a.mem_mb, a.alloc_mb)
        return smoke(a.n_frames, a.budget_sec)
    return run_guarded(_m, argv)


if __name__ == "__main__":
    sys.exit(main())
