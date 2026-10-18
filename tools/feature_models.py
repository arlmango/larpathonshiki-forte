"""RESEARCH (stage 3): local VLM pilot that answers the 8 human attribute questions as CLOSED ENUMS.

usage: python -m tools.feature_models pilot --candidate C02 [--budget-sec 900]
Model: Qwen2-VL-2B-Instruct (Apache-2.0, LICENSE file in Qwen/Qwen2-VL-2B-Instruct @ 895c3a49), 4-bit MLX
conversion mlx-community/Qwen2-VL-2B-Instruct-4bit @ 01af461c, fetched with SHA256 by
`python tools/fetch_weights.py qwen2_vl_2b_instruct_mlx4`. Runs in the separate research venv
(.venv-research: mlx 0.32.4, mlx-vlm 0.7.8, transformers 5.20.1) so the production .venv is untouched.

Fixed pilot rule (train only, holdout never opened): train units sorted by sha16 ascending; the first 2
reviewed_violation units with a label time and the first reviewed_no_violation unit. Frames at 1 fps:
violation = label time ±15 s; negative = a 30 s window whose start is drawn with seed int(sha16, 16).
At most 31 frames per clip; frames downscaled to 640 px wide.

PRIVACY: frames are decoded and answered inside a CHILD process (job via stdin, stderr DEVNULL). The
model's raw text never leaves the child's memory: it is mapped to an enum code (or -1 = invalid) and
discarded. Only numeric codes are cached (<LP>/train_probe/cache_qwen2vl/) and only aggregate counts
are printed. Pseudo-labels are features, never ground truth.
"""
import argparse
import json
import os
import re
import resource
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
LP = os.environ.get("LOCAL_PRIVATE_DIR") or os.path.join(ROOT, "local_private")
RESEARCH_PY = os.path.join(ROOT, ".venv-research", "bin", "python")
MODEL_DIR = os.path.join(ROOT, "weights", "qwen2_vl_2b_instruct_mlx4")
# (key, question, options); option order = enum code
QUESTIONS = [
    ("persons_in_work_area", "How many people are inside the cashier's work area behind the counter? "
     "Answer with exactly one of: 0, 1, 2+.", ["0", "1", "2+"]),
    ("cashier_hands_visible", "Are the cashier's hands visible? Answer with exactly one word: both, one, none, unclear.",
     ["both", "one", "none", "unclear"]),
    ("banknotes_on_counter", "Are banknotes (paper money) visible on the counter? Answer with exactly one word: "
     "yes, no, or unclear.", ["yes", "no", "unclear"]),
    ("cash_being_counted", "Is someone counting money at this moment? Answer with exactly one word: yes, no, or unclear.",
     ["yes", "no", "unclear"]),
    ("drawer_or_safe_open", "Is a cash drawer, cash box or safe open? Answer with exactly one word: yes, no, or unclear.",
     ["yes", "no", "unclear"]),
    ("object_handed_across_counter", "Is an object being handed across the counter at this moment? Answer with "
     "exactly one word: yes, no, or unclear.", ["yes", "no", "unclear"]),
    ("client_at_counter", "Is a customer standing at the counter in front of the cashier? Answer with exactly one "
     "word: yes, no, or unclear.", ["yes", "no", "unclear"]),
    ("cashier_orientation", "Is the cashier facing the counter, turned away, or not visible? Answer with exactly one "
     "word: facing, away, notvisible.", ["facing", "away", "notvisible"]),
]
SYN = {"two": "2+", "2": "2+", "3": "2+", "4": "2+", "many": "2+", "zero": "0", "none": "0", "one": "1",
       "notvisible": "notvisible", "not": "notvisible"}


def to_code(text, options):
    """Raw model text -> enum code (index in options) or -1. The text itself is never stored."""
    t = re.sub(r"[^a-z0-9+ ]", "", (text or "").strip().lower())
    if not t:
        return -1
    joined = t.replace(" ", "")
    for cand in (joined, t.split()[0]):
        if cand in options:
            return options.index(cand)
        if cand in SYN and SYN[cand] in options:
            # "none" is a valid option for hands; only map it to "0" when "0" is the option set
            return options.index(cand) if cand in options else options.index(SYN[cand])
    return -1


# ============================================================================ child (research venv)
def _heartbeat(mx):
    """Numeric-only JSON heartbeat on stdout every 1 s (RSS + MLX memory) for the parent watchdog."""
    import threading

    def loop():
        while True:
            hb = {"hb": 1, "rss_mb": _rss_mb(os.getpid())}
            if mx is not None:
                hb.update(mlx_active_mb=mx.get_active_memory() / 2 ** 20, mlx_peak_mb=mx.get_peak_memory() / 2 ** 20)
            sys.stdout.write(json.dumps(hb) + "\n")
            sys.stdout.flush()
            time.sleep(1.0)
    threading.Thread(target=loop, daemon=True).start()


def _rss_mb(pid):
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout
        return int(out.strip() or 0) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def child():
    job = json.loads(sys.stdin.readline())
    cap_bytes = int(job["mem_mb"]) << 20
    if job["mode"] == "memtest":  # bounded synthetic allocation, used only to prove the kill path
        mx = None
        if job.get("mlx"):
            import mlx.core as mx
            mx.set_memory_limit(cap_bytes)
            mx.set_cache_limit(cap_bytes)
        _heartbeat(mx)
        held = []
        for _ in range(int(job["total_mb"]) // int(job["step_mb"])):
            if mx is not None:
                a = mx.ones((int(job["step_mb"]) << 18,), dtype=mx.float32)
                mx.eval(a)
                held.append(a)
            else:
                held.append(bytearray(int(job["step_mb"]) << 20))
                held[-1][::4096] = b"x" * len(held[-1][::4096])  # touch pages
            time.sleep(0.4)
        time.sleep(2.5)
        return 0
    import cv2
    import mlx.core as mx
    from PIL import Image
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    mx.set_memory_limit(cap_bytes)
    mx.set_cache_limit(min(cap_bytes, 1 << 30))
    _heartbeat(mx)
    t0 = time.time()
    model, processor = load(job["model_dir"])
    prompts = [apply_chat_template(processor, model.config, q, num_images=1) for _k, q, _o in QUESTIONS]
    load_s = time.time() - t0
    cap = cv2.VideoCapture(job["path"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n / fps if fps > 0 else 0
    if job["mode"] == "label":
        times = [job["t"] + k for k in range(-15, 16) if 0 <= job["t"] + k < dur]
    else:
        import random
        s0 = random.Random(job["seed"]).uniform(0, max(0.0, dur - 30))
        times = [s0 + k for k in range(31) if s0 + k < dur]
    def answer(img):
        row = []
        for (_k, _q, opts), p in zip(QUESTIONS, prompts):
            r = generate(model, processor, p, image=[img], max_tokens=5, temperature=0.0, verbose=False)
            row.append(to_code(getattr(r, "text", ""), opts))
            r = None  # raw text discarded
        return row

    codes, ms, repeat, perturb = [], [], [], []
    for j, t in enumerate(times[:31]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            codes.append([-2] * len(QUESTIONS))
            continue
        h, w = fr.shape[:2]
        fr = cv2.resize(fr, (640, int(640 * h / w))) if w > 640 else fr
        img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        tf = time.time()
        row = answer(img)
        ms.append((time.time() - tf) * 1000)
        codes.append(row)
        if j % 6 == 0:  # reliability: identical re-run, and a negligible perturbation (resize to 608 px)
            repeat.append([row, answer(img)])
            small = cv2.resize(fr, (608, int(608 * fr.shape[0] / fr.shape[1])))
            perturb.append([row, answer(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))])
    cap.release()
    out = {"codes": codes, "ms": ms, "repeat": repeat, "perturb": perturb, "n_frames": len(times[:31]),
           "load_s": load_s, "mlx_peak_gb": mx.get_peak_memory() / 1e9,
           "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9}
    with open(job["out"] + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(job["out"] + ".tmp", job["out"])
    return 0


def run_child(job, budget, mem_mb):
    """Spawn the research child; ENFORCE the memory cap: a watchdog reads the child's numeric heartbeat
    (RSS, MLX active memory) and also polls RSS via ps; any value > mem_mb -> SIGKILL of the whole
    process group -> 'memory_cap'. Also 'timeout' at the wall budget. Non-heartbeat stdout is discarded."""
    import threading
    job = dict(job, mem_mb=mem_mb)
    p = subprocess.Popen([RESEARCH_PY, "-m", "tools.feature_models", "--child"], cwd=ROOT, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    last = {"rss_mb": 0.0, "mlx_active_mb": 0.0, "peak_seen_mb": 0.0}

    def reader():
        for line in p.stdout:
            try:
                hb = json.loads(line)
            except ValueError:
                continue  # never echoed
            if isinstance(hb, dict) and hb.get("hb") == 1:
                for k in ("rss_mb", "mlx_active_mb"):
                    v = hb.get(k)
                    if isinstance(v, (int, float)):
                        last[k] = float(v)
    threading.Thread(target=reader, daemon=True).start()
    try:
        p.stdin.write((json.dumps(job) + "\n").encode())
        p.stdin.close()
    except OSError:
        pass
    t0, status = time.time(), None
    while p.poll() is None:
        used = max(last["rss_mb"], last["mlx_active_mb"], _rss_mb(p.pid))
        last["peak_seen_mb"] = max(last["peak_seen_mb"], used)
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
        time.sleep(0.5)
    if status is None:
        status = "ok" if p.returncode == 0 and (job["mode"] == "memtest" or os.path.exists(job["out"])) else "failed"
    return status, last["peak_seen_mb"]


def memtest(mem_mb, total_mb, step_mb, use_mlx):
    """Prove the kill path with a SMALL cap and a bounded allocation (never > total_mb)."""
    st, peak = run_child({"mode": "memtest", "total_mb": total_mb, "step_mb": step_mb, "mlx": use_mlx,
                          "out": ""}, 120, mem_mb)
    try:
        leftover = subprocess.run(["pgrep", "-f", "tools.feature_models --child"], capture_output=True,
                                  text=True).stdout.split()
    except OSError:
        leftover = ["?"]
    print(f"MEMTEST cap {mem_mb} MB, bounded allocation {total_mb} MB in {step_mb} MB steps, "
          f"{'MLX arrays' if use_mlx else 'host bytes'}: status={st} peak_seen_mb={peak:.0f} "
          f"leftover_children={len(leftover)}")
    return 0 if st in ("memory_cap", "failed") and not leftover else 1


# ============================================================================ parent (main venv)
def pick_units(cid):
    from tools.local_safe import select_split_units
    units, counts = select_split_units(LP, cid, "train")
    units = sorted(units)  # by sha16 ascending
    pos = [u for u in units if u[2] == "reviewed_violation" and u[3]][:2]
    neg = [u for u in units if u[2] == "reviewed_no_violation"][:1]
    return pos, neg, counts


def pilot(cid, budget, mem_mb, show_dist=False):
    import numpy as np
    if not os.path.exists(RESEARCH_PY) or not os.path.isdir(MODEL_DIR):
        raise SystemExit("refused: research venv or model weights missing")
    pos, neg, counts = pick_units(cid)
    cache = os.path.join(LP, "train_probe", "cache_qwen2vl")
    os.makedirs(cache, exist_ok=True)
    jobs = [(sha, path, {"mode": "label", "t": float(ev[0])}) for sha, path, _s, ev in pos] + \
           [(sha, path, {"mode": "random", "seed": int(sha, 16) % (2 ** 32)}) for sha, path, _s, _e in neg]
    res, status, walls, peaks, t_all = [], {}, [], [], time.time()
    for sha, path, spec in jobs:
        out = os.path.join(cache, f"{sha}_pilot.json")
        remaining = budget - (time.time() - t_all)
        if remaining <= 10:
            status["budget_exhausted"] = status.get("budget_exhausted", 0) + 1
            continue
        t0 = time.time()
        st, peak = run_child(dict(spec, path=path, out=out, model_dir=MODEL_DIR), remaining, mem_mb)
        peaks.append(peak)
        walls.append(time.time() - t0)
        status[st] = status.get(st, 0) + 1
        if st == "ok":
            with open(out) as f:
                res.append((spec["mode"], json.load(f)))
    child_rss_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e9
    allc = [np.array(r["codes"]) for _m, r in res]
    ms = np.concatenate([r["ms"] for _m, r in res]) if res else np.zeros(0)
    print(f"PILOT clips: {len(jobs)} (violation {len(pos)}, negative {len(neg)}); child status {json.dumps(status)}; "
          f"train selector counts {json.dumps(counts)}")
    if not res:
        return 2
    frames = sum(len(c) for c in allc)
    print(f"frames answered: {frames}; questions/frame: {len(QUESTIONS)}; model load s (median): "
          f"{np.median([r['load_s'] for _m, r in res]):.1f}")
    print(f"ms/frame (8 answers) p50 {np.percentile(ms, 50):.0f} p95 {np.percentile(ms, 95):.0f} max {ms.max():.0f}; "
          f"wall per clip s: {', '.join(f'{w:.0f}' for w in sorted(walls))} (sorted); total wall s {time.time()-t_all:.0f}")
    print(f"peak memory: child maxrss {child_rss_gb:.2f} GB; watchdog max seen {max(peaks) / 1024:.2f} GB "
          f"(cap {mem_mb / 1024:.1f} GB); MLX peak (max over clips) {max(r['mlx_peak_gb'] for _m, r in res):.2f} GB; "
          f"load avg 1m {os.getloadavg()[0]:.2f}")
    print("\nper question, pooled over all pilot frames: valid rate | enum distribution | frac clips non-constant")
    C = np.concatenate(allc)
    for qi, (k, _q, opts) in enumerate(QUESTIONS):
        col = C[:, qi]
        valid = col >= 0
        dist = {o: int((col == i).sum()) for i, o in enumerate(opts)} if show_dist else "(not shown: diagnostic run)"
        nonconst = np.mean([len(set(c[:, qi][c[:, qi] >= 0])) > 1 for c in allc])
        print(f"{k:30s} {valid.mean():5.2f} | {json.dumps(dist)} | invalid {int((col == -1).sum())} "
              f"unreadable {int((col == -2).sum())} | {nonconst:.2f}")
    print("\nreliability (sampled frames, every 6th): agreement identical re-run | agreement after resize 640->608 px")
    for qi, (k, _q, _o) in enumerate(QUESTIONS):
        rep_ = [a[qi] == b[qi] for _m, r in res for a, b in r["repeat"]]
        per_ = [a[qi] == b[qi] for _m, r in res for a, b in r["perturb"]]
        print(f"{k:30s} {np.mean(rep_) if rep_ else float('nan'):.2f} (n={len(rep_)}) | "
              f"{np.mean(per_) if per_ else float('nan'):.2f} (n={len(per_)})")
    print("question order: each question is a separate prompt with no shared state, so order cannot change answers.")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--child"]:
        return child()
    from tools.local_safe import run_guarded

    def _m(av):
        ap = argparse.ArgumentParser(description="stage-3 VLM pilot (aggregates only)")
        ap.add_argument("cmd", choices=("pilot", "memtest"))
        ap.add_argument("--candidate")
        ap.add_argument("--budget-sec", type=float, default=900.0)
        ap.add_argument("--mem-mb", type=int, default=int(os.environ.get("PILOT_MEM_MB", "8192")))
        ap.add_argument("--alloc-mb", type=int, default=300)
        ap.add_argument("--mlx", action="store_true")
        ap.add_argument("--show-distributions", action="store_true",
                        help="print enum distributions (synthetic false-positive measurement only)")
        a = ap.parse_args(av)
        if a.cmd == "memtest":
            return memtest(a.mem_mb, min(a.alloc_mb, 400), 32, a.mlx)
        if not a.candidate:
            raise SystemExit("refused: --candidate required")
        return pilot(a.candidate, a.budget_sec, a.mem_mb, a.show_distributions)
    return run_guarded(_m, argv)


if __name__ == "__main__":
    sys.exit(main())
