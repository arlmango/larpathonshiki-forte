import base64
import io
import json
import os
import time
from dataclasses import replace

import cv2
import numpy as np

from app import config
from app import pipeline
from tools.make_synthetic import make

KEY = "k"
ALLOWED_CODES = set(pipeline.AnalysisError.CODES) | {
    "time_budget_exceeded", "worker_crashed", "worker_crashed_oom_suspected", "job_lost",
    "invalid_result", "internal_error", "service_shutting_down", "analysis_failed"}


def settings(tmp_path, **kw):
    kw.setdefault("data_dir", str(tmp_path / "data"))
    return replace(config.Settings(), api_key=KEY, **kw)


def clip(tmp_path, name="v.mp4", dur=40, **kw):
    p = tmp_path / name
    make(str(p), dur, fps=10, **kw)
    return str(p)


def rules_file(tmp_path, mutate=None, enable=("operator_zone_unattended_with_counter_activity",)):
    """Copy of the default ruleset with `enable` rules enabled; mutate(rules_dict) for extra edits."""
    with open(pipeline.DEFAULT_RULESET) as f:
        r = json.load(f)
    for k in enable:
        r["rules"][k]["enabled"] = True
    if mutate:
        mutate(r)
    p = tmp_path / "rules.json"
    p.write_text(json.dumps(r))
    return str(p)


def jpeg_b64(w=64, h=48):
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 128, np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def wait_store(store, jid, timeout=30):
    t = time.time()
    while time.time() - t < timeout:
        j = store.get(jid)
        if j and j["status"] != "processing":
            return j
        time.sleep(0.1)
    raise AssertionError("job still processing after timeout")


def wait_http(c, jid, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        j = c.get(f"/api/v1/jobs/{jid}", headers={"X-API-Key": KEY}).json()
        if j["status"] != "processing":
            return j
        time.sleep(0.2)
    raise AssertionError("timeout")


def post(c, data, key=KEY, name="a.mov", ctype=None):
    f = (name, io.BytesIO(data)) if ctype is None else (name, io.BytesIO(data), ctype)
    return c.post("/api/v1/analyze", headers={"X-API-Key": key}, files={"file": f})


def inproc(detector=None):
    """analyze_fn running the real pipeline in-process with an injected detector (None -> env/synthetic)."""
    def fn(path, s, deadline=None):
        return pipeline.analyze(path, s, detector=detector, deadline=deadline)
    return fn


class EmptyDetector:
    info = {"name": "fake-empty"}

    def detect(self, frame):
        return np.zeros((0, 5), np.float32)


def leftovers(data_dir):
    return [n for n in os.listdir(data_dir) if n.endswith((".upload", ".tmp", ".worker.json"))] \
        if os.path.isdir(data_dir) else []


class TwoPersonDetector:
    """Wraps the synthetic figure detector and adds, for every detection, a second box beside it
    (low IoU, inside the padded operator zone) so 'two persons in zone' is deterministic."""
    info = {"name": "fake-two-persons"}

    def __init__(self):
        from app.detector import SyntheticFigureDetector
        self.inner = SyntheticFigureDetector()

    def detect(self, frame):
        b = self.inner.detect(frame)
        if not len(b):
            return b
        extra = b.copy()
        w = extra[:, 2] - extra[:, 0]
        extra[:, 0] += 0.9 * w
        extra[:, 2] += 0.9 * w
        return np.concatenate([b, extra], 0)
