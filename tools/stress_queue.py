"""Queue/isolation self-check. SYNTHETIC DATA ONLY (tools/make_synthetic.py).

Phase A (subprocess isolation, default mode): synthetic clips, corrupt files and
FIFO inputs (cv2 blocks in native open() -> simulates a hung decoder that must be
SIGKILLed). Phase B (in-process analyze_fn): fast/slow/hanging fake analyzers.
Checks: every admitted job ends done/failed within budget + 5 s, no job left in
`processing`, no leftover upload/temp files, durable status readable by a new store.

usage: python tools/stress_queue.py [N] [budget_sec]
"""
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import config  # noqa: E402
from app.jobs import JobStore, QueueFull  # noqa: E402
from tools.make_synthetic import make  # noqa: E402

SLACK = 5.0


def run_phase(name, settings, inputs, analyze_fn=None):
    store = JobStore(settings, analyze_fn)
    submitted, rejected = {}, 0
    for i, (kind, maker) in enumerate(inputs):
        p = os.path.join(settings.data_dir, f"{os.urandom(16).hex()}.upload")
        maker(p)
        try:
            jid = store.submit(p)
            submitted[jid] = (kind, time.time())
        except QueueFull:
            rejected += 1
            assert not os.path.exists(p), "rejected upload not removed"
        time.sleep(0.02)
    done_at = {}
    limit = time.time() + settings.job_budget_sec + SLACK + 2
    while len(done_at) < len(submitted) and time.time() < limit:
        for jid in submitted:
            if jid not in done_at and store.get(jid)["status"] != "processing":
                done_at[jid] = time.time()
        time.sleep(0.05)
    results = {jid: store.get(jid) for jid in submitted}
    lat = [done_at[j] - submitted[j][1] for j in done_at]
    by = {}
    for jid, r in results.items():
        key = (submitted[jid][0], r["status"], r.get("error", r.get("verdict")))
        by[key] = by.get(key, 0) + 1
    leftover_processing = [j for j, r in results.items() if r["status"] == "processing"]
    over = [x for x in lat if x > settings.job_budget_sec + SLACK]
    bad_status = [r for r in results.values() if r["status"] not in ("done", "failed")]
    no_verdict_on_fail = all("verdict" not in r for r in results.values() if r["status"] == "failed")
    time.sleep(1.0)  # let worker threads finish their cleanup
    stray = [f for f in os.listdir(settings.data_dir) if not f.endswith(".json") or ".worker" in f]
    # durability: a fresh store on the same data_dir answers from disk
    store.close()
    fresh = JobStore(settings, analyze_fn or (lambda *a, **k: None))
    durable = sum(1 for j in submitted if (fresh.get(j) or {}).get("status") == results[j]["status"])
    fresh.close()
    print(f"[{name}] offered={len(inputs)} admitted={len(submitted)} rejected_429={rejected}")
    for k in sorted(by, key=str):
        print(f"    {k[0]:<8} -> {k[1]:<7} {str(k[2]):<45} x{by[k]}")
    if lat:
        print(f"    latency submit->final: min={min(lat):.2f}s max={max(lat):.2f}s "
              f"(budget {settings.job_budget_sec}s, limit {settings.job_budget_sec + SLACK:.0f}s)")
    print(f"    leftover processing={len(leftover_processing)} over_limit={len(over)} "
          f"stray_files={len(stray)} durable_on_disk={durable}/{len(submitted)} "
          f"failed_without_verdict={no_verdict_on_fail}")
    ok = not leftover_processing and not over and not stray and not bad_status \
        and no_verdict_on_fail and durable == len(submitted) and len(done_at) == len(submitted)
    return ok


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    work = tempfile.mkdtemp(prefix="stressq_")
    try:
        clip = os.path.join(work, "clip.mp4")
        make(clip, 3)  # 3 s synthetic clip

        def synth(p): shutil.copyfile(clip, p)
        def corrupt(p): open(p, "wb").write(b"not a video" * 100)
        def fifo(p): os.mkfifo(p)

        kinds = [("synth", synth), ("corrupt", corrupt), ("hang", fifo)]
        inputs = [kinds[i % 3] for i in range(n)]
        base = replace(config.Settings(), api_key="x", job_budget_sec=budget, workers=2,
                       queue_limit=n, job_estimate_sec=budget / 4, kill_grace_sec=1.0,
                   admission_margin_sec=0.5)

        da = os.path.join(work, "a"); os.makedirs(da)
        ok_a = run_phase("A subprocess", replace(base, data_dir=da), inputs)

        gate = threading.Event()

        def fake(path, s, deadline=None):
            k = os.path.getsize(path) % 3
            if k == 0:
                time.sleep(0.2); return {"verdict": "no_violation", "violations": []}
            if k == 1:
                raise RuntimeError("synthetic crash")
            gate.wait(budget * 3)  # hangs past the deadline (cannot be killed in-process)
            return {"verdict": "no_violation", "violations": []}

        def sized(k):
            return lambda p: open(p, "wb").write(b"\0" * (300 + k))
        inputs_b = [(("ok", "crash", "hang")[i % 3], sized(i % 3)) for i in range(n)]
        db = os.path.join(work, "b"); os.makedirs(db)
        ok_b = run_phase("B in-process", replace(base, data_dir=db), inputs_b, fake)
        gate.set()
        print("RESULT:", "PASS" if ok_a and ok_b else "FAIL")
        return 0 if ok_a and ok_b else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
