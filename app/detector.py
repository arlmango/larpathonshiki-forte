"""Local person detector: YOLOX (Megvii, Apache-2.0) ONNX via onnxruntime.

Code licence:    https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE (Apache-2.0)
Weights:         release 0.1.1rc0 assets of the same repository (Apache-2.0 repo licence;
                 trained on COCO - see docs for dataset-terms caveat).
No ultralytics code or weights are used anywhere.

The only production path is the ONNX model. Missing / tampered weights raise
AnalysisError (job -> failed), never a silent fallback.  For deterministic tests on
the synthetic videos from tools/make_synthetic.py set SYNTHETIC_DETECTOR=1 (finds the
drawn figure by colour; useless on real video and labelled as such in describe()).

detect(bgr) -> float32 array (N, 5): x0, y0, x1, y1 (pixels of the given frame), score.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import threading

import cv2
import numpy as np


class DetectorError(Exception):
    """Unusable detector; pipeline maps it to AnalysisError with the same .code."""

    def __init__(self, msg: str, code: str = "model_load_failed"):
        super().__init__(msg)
        self.code = code


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(ROOT, "weights")
REPO = "https://github.com/Megvii-BaseDetection/YOLOX"
LICENSE_URL = REPO + "/blob/main/LICENSE"

# sha256 recorded from the official release assets (tools/fetch_weights.py verifies).
MODELS = {
    "yolox_nano": dict(file="yolox_nano.onnx", size=3659407, input=416,
                       sha256="c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"),
    "yolox_tiny": dict(file="yolox_tiny.onnx", size=20219662, input=416,
                       sha256="427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"),
    "yolox_s": dict(file="yolox_s.onnx", size=35858002, input=640,
                    sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"),
}
for _k, _m in MODELS.items():
    _m["url"] = f"{REPO}/releases/download/0.1.1rc0/{_m['file']}"
DEFAULT_MODEL = "yolox_s"
PERSON_CLASS = 0  # COCO index


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    x0, y0, x1, y1 = boxes.T
    areas = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx0 = np.maximum(x0[i], x0[order[1:]]); yy0 = np.maximum(y0[i], y0[order[1:]])
        xx1 = np.minimum(x1[i], x1[order[1:]]); yy1 = np.minimum(y1[i], y1[order[1:]])
        inter = np.maximum(0, xx1 - xx0) * np.maximum(0, yy1 - yy0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


# ---- normalised-box geometry helpers (boxes: (N,5) x0,y0,x1,y1,score in 0..1) ----
def centers(b):
    return (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2


def inside(b, z):
    cx, cy = centers(b)
    return (cx >= z[0]) & (cx <= z[2]) & (cy >= z[1]) & (cy <= z[3])


def pad(z, p):
    w, h = z[2] - z[0], z[3] - z[1]
    return [max(0.0, z[0] - p * w), max(0.0, z[1] - p * h), min(1.0, z[2] + p * w), min(1.0, z[3] + p * h)]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


# ---- adaptive operator-zone estimation over stage-1 samples ----
def estimate_zone(samples, rules):
    """Densest person-box-centre region over the clip (operators stay put, clients move)."""
    G = rules["zone"]["heatmap_grid"]
    allb = np.concatenate([s["boxes"] for s in samples if len(s["boxes"])] or [np.zeros((0, 5))])
    if not len(allb):
        return None, 0
    cx, cy = centers(allb)
    heat = np.zeros((G, G), np.float32)
    np.add.at(heat, (np.clip((cy * G).astype(int), 0, G - 1), np.clip((cx * G).astype(int), 0, G - 1)),
              allb[:, 4])
    heat = cv2.GaussianBlur(heat, (5, 5), 0)
    my, mx = np.unravel_index(int(heat.argmax()), heat.shape)
    mxn, myn = (mx + 0.5) / G, (my + 0.5) / G
    size = np.maximum(allb[:, 2] - allb[:, 0], allb[:, 3] - allb[:, 1])
    near = np.hypot(cx - mxn, cy - myn) < np.maximum(0.5 * size, 1.5 / G)
    if not near.any():
        return None, 0
    z = np.median(allb[near, :4], axis=0).tolist()
    return pad(z, rules["zone"]["pad"]), int(near.sum())


def cell_mask(z, gw, gh):
    m = np.zeros((gh, gw), bool)
    x0, x1 = int(np.floor(z[0] * gw)), int(np.ceil(z[2] * gw))
    y0, y1 = int(np.floor(z[1] * gh)), int(np.ceil(z[3] * gh))
    m[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)] = True
    return m


# ---- temporal helpers over per-sample signals ----
def smooth(x, k):
    if k <= 1 or len(x) < k:
        return x.copy()
    p = np.pad(x.astype(np.float32), (k // 2, k // 2), mode="edge")
    return np.convolve(p, np.ones(k) / k, mode="valid") >= 0.5


def runs(mask, times, dt):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((times[i], times[j + 1] if j + 1 < n else times[j] + dt))
            i = j + 1
        else:
            i += 1
    return out


def merge_intervals(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


class YoloxPersonDetector:
    def __init__(self, path: str, name: str = "custom", providers: list[str] | None = None,
                 conf: float = 0.30, nms_thr: float = 0.45, threads: int = 4,
                 expected_sha256: str | None = None):
        if not os.path.isfile(path):
            raise DetectorError("detector weights missing; run: python tools/fetch_weights.py",
                                code="model_missing")
        digest = sha256_file(path)
        if expected_sha256 and digest != expected_sha256:
            raise DetectorError("detector weights sha256 mismatch", code="model_invalid")
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, threads)
            so.log_severity_level = 3
            # no busy-wait spinning: keeps CPU-seconds close to useful work (RLIMIT_CPU in worker)
            if os.environ.get("DETECTOR_SPIN", "0") != "1":
                so.add_session_config_entry("session.intra_op.allow_spinning", "0")
                so.add_session_config_entry("session.inter_op.allow_spinning", "0")
            avail = ort.get_available_providers()
            prov = [p for p in (providers or ["CPUExecutionProvider"]) if p in avail]
            if "CPUExecutionProvider" not in prov:
                prov.append("CPUExecutionProvider")
            self.sess = ort.InferenceSession(path, so, providers=prov)
        except Exception as e:  # noqa: BLE001
            raise DetectorError("detector model failed to load") from e
        inp = self.sess.get_inputs()[0]
        shp = inp.shape
        if len(shp) != 4 or not all(isinstance(v, int) for v in shp[2:]):
            raise DetectorError("unsupported detector input shape", code="model_invalid")
        out = self.sess.get_outputs()[0].shape
        if len(out) != 3 or out[-1] != 85:
            raise DetectorError("detector output is not a YOLOX COCO head (85 channels)", code="model_invalid")
        self.inp_name, self.size = inp.name, (int(shp[2]), int(shp[3]))
        self.conf, self.nms_thr = conf, nms_thr
        self._grid, self._strides = self._make_grid(self.size)
        self.info = {"name": name, "license": "Apache-2.0 (YOLOX, Megvii)" if name in MODELS else
                     "unknown (custom ONNX_MODEL; operator must verify)",
                     "license_url": LICENSE_URL if name in MODELS else "",
                     "weights_sha256_prefix": digest[:12], "validated": False,
                     "providers": self.sess.get_providers(), "input": list(self.size)}

    @staticmethod
    def _make_grid(size):
        grids, strides = [], []
        for s in (8, 16, 32):
            h, w = size[0] // s, size[1] // s
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((h * w, 1), s))
        return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def _preprocess(self, bgr):
        """YOLOX >=0.1.1 letterbox: top-left aligned, pad 114, BGR, 0..255, no normalisation."""
        ih, iw = self.size
        r = min(ih / bgr.shape[0], iw / bgr.shape[1])
        nh, nw = int(bgr.shape[0] * r), int(bgr.shape[1] * r)
        pad = np.full((ih, iw, 3), 114, np.uint8)
        pad[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(pad.transpose(2, 0, 1)[None], dtype=np.float32), r

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        x, r = self._preprocess(bgr)
        out = self.sess.run(None, {self.inp_name: x})[0][0]
        if out.shape[0] != self._grid.shape[0]:
            raise DetectorError("detector output size does not match grid", code="model_invalid")
        score = out[:, 4] * out[:, 5 + PERSON_CLASS]
        # class filter: keep anchors whose best class is person
        m = (score >= self.conf) & (out[:, 5:].argmax(1) == PERSON_CLASS)
        if not m.any():
            return np.zeros((0, 5), np.float32)
        o, g, st = out[m], self._grid[m], self._strides[m]
        cxy = (o[:, :2] + g) * st
        wh = np.exp(np.clip(o[:, 2:4], -10, 10)) * st
        boxes = np.concatenate([cxy - wh / 2, cxy + wh / 2], 1) / r
        h, w = bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        sc = score[m]
        keep = nms(boxes, sc, self.nms_thr)
        return np.concatenate([boxes[keep], sc[keep, None]], 1).astype(np.float32)

    def detect_classes(self, bgr: np.ndarray, classes, conf: float = 0.25,
                       person_conf: float | None = None) -> np.ndarray:
        """Research helper (tools/train_probe.py; not used by the production pipeline): one
        inference, several COCO classes. -> (N, 6) x0,y0,x1,y1 (px), score, class; class-aware NMS.
        Person rows use the same score/argmax rule and threshold as detect()."""
        x, r = self._preprocess(bgr)
        out = self.sess.run(None, {self.inp_name: x})[0][0]
        cls = out[:, 5:].argmax(1)
        score = out[:, 4] * out[np.arange(len(out)), 5 + cls]
        pc = self.conf if person_conf is None else person_conf
        m = np.isin(cls, list(classes)) & (score >= np.where(cls == PERSON_CLASS, pc, conf))
        if not m.any():
            return np.zeros((0, 6), np.float32)
        o, g, st, c, sc = out[m], self._grid[m], self._strides[m], cls[m], score[m]
        cxy = (o[:, :2] + g) * st
        wh = np.exp(np.clip(o[:, 2:4], -10, 10)) * st
        boxes = np.concatenate([cxy - wh / 2, cxy + wh / 2], 1) / r
        h, w = bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        keep = nms(boxes + (c[:, None] * 8192.0), sc, self.nms_thr)  # offset => per-class NMS
        return np.concatenate([boxes[keep], sc[keep, None], c[keep, None]], 1).astype(np.float32)


class SyntheticFigureDetector:
    """TEST-ONLY: finds the figure drawn by tools/make_synthetic.py (body BGR (40,40,160)).
    Enabled only with SYNTHETIC_DETECTOR=1. Has no meaning on real video."""

    def __init__(self):
        self.info = {"name": "synthetic-figure-test-detector", "license": "project code",
                     "license_url": "", "weights_sha256_prefix": "", "validated": False,
                     "providers": ["numpy"], "input": None,
                     "warning": "TEST ONLY - not a person detector"}

    def detect(self, bgr):
        d = np.abs(bgr.astype(np.int16) - np.array([40, 40, 160], np.int16)).max(2)
        mask = (d < 25).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = []
        min_area = 0.002 * bgr.shape[0] * bgr.shape[1]
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a >= min_area:
                y0 = max(0, y - int(0.6 * h))  # include head drawn above the body
                out.append([x, y0, x + w, y + h, 0.9])
        return np.array(out, np.float32).reshape(-1, 5)


_stderr_lock = threading.Lock()
_stderr_state = {"depth": 0, "saved": None}


@contextlib.contextmanager
def quiet_native_stderr():
    """Redirect OS-level fd 2 to /dev/null (OpenCV/FFmpeg print file paths there natively).
    Re-entrant and thread-safe via a depth counter: first entrant redirects, last restores."""
    with _stderr_lock:
        if _stderr_state["depth"] == 0:
            try:
                saved, null = os.dup(2), os.open(os.devnull, os.O_WRONLY)
                os.dup2(null, 2)
                os.close(null)
                _stderr_state["saved"] = saved
            except OSError:
                _stderr_state["saved"] = None
        _stderr_state["depth"] += 1
    try:
        yield
    finally:
        with _stderr_lock:
            _stderr_state["depth"] -= 1
            if _stderr_state["depth"] == 0 and _stderr_state["saved"] is not None:
                os.dup2(_stderr_state["saved"], 2)
                os.close(_stderr_state["saved"])
                _stderr_state["saved"] = None


_cache: dict = {}
_lock = threading.Lock()


def load(onnx_path: str = "") -> object:
    """Resolve and cache (process-wide; ORT sessions are thread-safe for run()).
    Order: explicit onnx_path (always wins; missing -> model_missing even under
    SYNTHETIC_DETECTOR) -> SYNTHETIC_DETECTOR=1 -> weights/<DETECTOR_MODEL>.onnx."""
    if not onnx_path and os.environ.get("SYNTHETIC_DETECTOR") == "1":
        return SyntheticFigureDetector()
    providers = [p.strip() for p in os.environ.get("DETECTOR_PROVIDERS", "CPUExecutionProvider").split(",")
                 if p.strip()]
    threads = int(os.environ.get("DETECTOR_THREADS", "4"))
    conf = float(os.environ.get("DETECTOR_CONF", "0.30"))
    if onnx_path:
        name, path, sha = "custom", onnx_path, None
        for k, m in MODELS.items():
            if os.path.basename(onnx_path) == m["file"]:
                name, sha = k, m["sha256"]
    else:
        name = os.environ.get("DETECTOR_MODEL", DEFAULT_MODEL)
        if name not in MODELS:
            raise DetectorError(f"unknown DETECTOR_MODEL {name!r}", code="invalid_config")
        path, sha = os.path.join(WEIGHTS_DIR, MODELS[name]["file"]), MODELS[name]["sha256"]
    key = (os.path.abspath(path), tuple(providers), threads, conf)
    with _lock:
        if key not in _cache:
            _cache[key] = YoloxPersonDetector(path, name, providers, conf=conf, threads=threads,
                                              expected_sha256=sha)
        return _cache[key]


def describe(onnx_path: str = "") -> dict:
    try:
        return dict(load(onnx_path).info)
    except DetectorError as e:
        name = "custom" if onnx_path else os.environ.get("DETECTOR_MODEL", DEFAULT_MODEL)
        m = MODELS.get(name, {})
        return {"name": name, "license": "Apache-2.0 (YOLOX, Megvii)" if m else "unknown",
                "license_url": LICENSE_URL if m else "",
                "weights_sha256_prefix": m.get("sha256", "")[:12], "validated": False,
                "providers": [], "error": e.code}
