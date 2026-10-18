"""Isolated analysis child process.

usage: python -m app.worker <input_video> <out.json>

Environment (set by app.jobs):
  JOB_SETTINGS  JSON of config.Settings fields (falls back to config.get())
  JOB_DEADLINE  absolute epoch seconds; passed to pipeline as soft deadline
  JOB_PARENT_PID  exit if this parent goes away (no orphaned analyzers)

Always writes <out.json> atomically as either
  {"status": "done", "verdict": ..., "violations": [...]}  or
  {"status": "failed", "error": "..."}
The parent treats a missing/invalid file or non-zero exit as `failed`.
stderr is discarded by the parent (may contain real-data details).
The parent hard-kills this process (SIGKILL to its process group) at the deadline.
"""
import base64
import binascii
import dataclasses
import json
import math
import os
import sys
import threading
import time


# Data boundary: the API `error` field is a bare snake_case code from a closed
# set. Never str(e) of any exception, never class names, never decoder stderr.
#
# Codes raised by app/pipeline.py / app/detector.py via AnalysisError.code
# (our own copy of that closed set; anything outside it -> analysis_failed).
PIPELINE_CODES = frozenset({
    "model_missing", "model_load_failed", "model_invalid", "invalid_config",
    "corrupt_video", "too_long", "too_few_frames", "decode_error",
    "no_validated_rule", "too_few_detections", "zone_unreliable", "jpeg_failed",
})
# pipeline code -> API code where the queue uses one shared name
_CODE_ALIASES = {"time_budget": "time_budget_exceeded"}
# Codes produced by the queue / worker itself.
QUEUE_CODES = frozenset({
    "time_budget_exceeded", "worker_crashed", "worker_crashed_oom_suspected",
    "job_lost", "invalid_result", "internal_error", "service_shutting_down",
    "analysis_failed",
})
ALLOWED_ERROR_CODES = PIPELINE_CODES | QUEUE_CODES
GENERIC_ERROR = "analysis_failed"

# FALLBACK HOOK: used only when an AnalysisError has no valid `.code`
# (e.g. an older pipeline). Prefix of the message -> code; dynamic tails dropped.
_PREFIX_CODES = (
    ("cannot open video", "corrupt_video"),
    ("corrupt or empty video", "corrupt_video"),
    ("video too long", "too_long"),
    ("time budget exceeded", "time_budget_exceeded"),
    ("decode error", "decode_error"),
    ("stage-2 decode failed", "decode_error"),
    ("too few decodable frames", "too_few_frames"),
    ("evidence frame extraction failed", "jpeg_failed"),
    ("jpeg encode failed", "jpeg_failed"),
    ("invalid operator zone override", "invalid_config"),
    ("operator zone could not be estimated", "zone_unreliable"),
    ("operator zone occupied in", "zone_unreliable"),
    ("detector found persons in", "too_few_detections"),
    ("no validated rule active", "no_validated_rule"),
    ("detector weights missing", "model_missing"),
    ("configured detector model is missing", "model_missing"),
    ("detector weights sha256 mismatch", "model_invalid"),
    ("detector model failed to load", "model_load_failed"),
    ("unsupported detector input shape", "model_invalid"),
    ("detector output is not", "model_invalid"),
    ("detector output size does not match", "model_invalid"),
    ("ONNX decode head not implemented", "model_invalid"),
    ("unknown DETECTOR_MODEL", "invalid_config"),
)


def safe_code(code) -> str:
    """Normalise a candidate code; anything outside the closed set -> analysis_failed."""
    if not isinstance(code, str):
        return GENERIC_ERROR
    code = _CODE_ALIASES.get(code, code)
    return code if code in ALLOWED_ERROR_CODES else GENERIC_ERROR


def safe_error(e: BaseException) -> str:
    """Map an exception to a bare static error code for the API and logs."""
    if isinstance(e, MemoryError):
        return "worker_crashed_oom_suspected"
    if e.__class__.__name__ != "AnalysisError":
        return GENERIC_ERROR
    code = getattr(e, "code", None)  # preferred: AnalysisError.code (closed set)
    if isinstance(code, str) and _CODE_ALIASES.get(code, code) in ALLOWED_ERROR_CODES:
        return _CODE_ALIASES.get(code, code)
    msg = str(e)  # fallback hook: prefix mapping, message itself never surfaced
    for prefix, c in _PREFIX_CODES:
        if msg.startswith(prefix):
            return c
    return GENERIC_ERROR


# ------------------------------------------------------------------
# Public output contract (shared by the worker child and the parent JobStore,
# so in-process and subprocess paths validate identically). Never logs content.
VERDICTS = ("violation", "no_violation")
MAX_VIOLATIONS = 50
MAX_TYPE_LEN = 200
MAX_DESCRIPTION_LEN = 4000
MIN_FRAMES, MAX_FRAMES = 1, 3
MIN_JPEG_BYTES = 101            # "non-trivial": > 100 bytes
MAX_JPEG_BYTES = 2 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 16, 4096   # decoded width/height bounds (px)
_MAX_B64_LEN = 4 * ((MAX_JPEG_BYTES + 2) // 3)


class InvalidResult(ValueError):
    """Analysis output violates the public contract (-> failed / invalid_result).
    The message is a static description of the rule, never content."""


def _num(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise InvalidResult(f"{what} must be a finite number >= 0")
    return v


def _text(v, what, max_len):
    if not isinstance(v, str) or not v.strip() or len(v) > max_len:
        raise InvalidResult(f"{what} must be a non-empty string <= {max_len} chars")
    return v


def _frame(f) -> dict:
    if not isinstance(f, dict):
        raise InvalidResult("evidence frame must be an object")
    ts = _num(f.get("timestamp_sec"), "timestamp_sec")
    b64 = f.get("image_base64")
    if not isinstance(b64, str) or not b64 or len(b64) > _MAX_B64_LEN:
        raise InvalidResult("image_base64 must be a non-empty string within size limit")
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise InvalidResult("image_base64 is not strict base64") from None
    if not (MIN_JPEG_BYTES <= len(raw) <= MAX_JPEG_BYTES) or raw[:2] != b"\xff\xd8" \
            or raw[-2:] != b"\xff\xd9":  # SOI ... EOI; truncated files lack EOI
        raise InvalidResult("image_base64 must be a complete JPEG of sane size")
    _check_jpeg_decodes(raw)
    return {"timestamp_sec": ts, "image_base64": b64}


_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _jpeg_header(raw: bytes):
    """Walk JPEG marker segments up to the first SOS. Returns (w, h) from SOF,
    or raises InvalidResult if the structure is malformed."""
    i, n, dims, dqt = 2, len(raw), None, False
    while i + 4 <= n:
        if raw[i] != 0xFF:
            raise InvalidResult("JPEG marker structure is malformed")
        m = raw[i + 1]
        if m == 0xFF:  # fill byte
            i += 1
            continue
        if m in (0x01,) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m in (0xD8, 0xD9):
            raise InvalidResult("JPEG marker structure is malformed")
        seg = int.from_bytes(raw[i + 2:i + 4], "big")
        if seg < 2 or i + 2 + seg > n:
            raise InvalidResult("JPEG segment length is invalid")
        if m == 0xDB:
            dqt = True
        elif m in _SOF:
            if seg < 8:
                raise InvalidResult("JPEG frame header is invalid")
            h = int.from_bytes(raw[i + 5:i + 7], "big")
            w = int.from_bytes(raw[i + 7:i + 9], "big")
            dims = (w, h)
        elif m == 0xDA:  # start of scan: entropy-coded data follows until EOI
            if dims is None or not dqt or n - (i + 2 + seg) - 2 < 2:
                raise InvalidResult("JPEG is missing frame/quant tables or scan data")
            return dims
        i += 2 + seg
    raise InvalidResult("JPEG has no scan data")


def _check_jpeg_decodes(raw: bytes):
    """Decode only to validate; images are discarded, never stored or logged."""
    import cv2
    import numpy as np
    try:
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error:
        img = None
    if img is None or img.ndim < 2:
        raise InvalidResult("image_base64 is not a decodable JPEG")
    h, w = img.shape[:2]
    if _jpeg_header(raw) != (w, h):
        raise InvalidResult("JPEG header does not match decoded image")
    if not (MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE):
        raise InvalidResult(f"evidence image must be {MIN_SIDE}-{MAX_SIDE} px per side")


def _violation(v) -> dict:
    if not isinstance(v, dict):
        raise InvalidResult("violation must be an object")
    frames = v.get("evidence_frames")
    if not isinstance(frames, list) or not MIN_FRAMES <= len(frames) <= MAX_FRAMES:
        raise InvalidResult(f"evidence_frames must be a list of {MIN_FRAMES}-{MAX_FRAMES} items")
    return {"type": _text(v.get("type"), "type", MAX_TYPE_LEN),
            "timecode_sec": _num(v.get("timecode_sec"), "timecode_sec"),
            "description": _text(v.get("description"), "description", MAX_DESCRIPTION_LEN),
            "evidence_frames": [_frame(f) for f in frames]}


def validate_result(res) -> dict:
    """Validate analysis output and return ONLY the public projection
    {verdict, violations:[{type, timecode_sec, description,
    evidence_frames:[{timestamp_sec, image_base64}]}]}. Raises InvalidResult."""
    if not isinstance(res, dict):
        raise InvalidResult("result must be an object")
    verdict, vs = res.get("verdict"), res.get("violations")
    if not isinstance(verdict, str) or verdict not in VERDICTS:
        raise InvalidResult("verdict must be one of " + "/".join(VERDICTS))
    if not isinstance(vs, list) or len(vs) > MAX_VIOLATIONS:
        raise InvalidResult(f"violations must be a list of <= {MAX_VIOLATIONS} items")
    if verdict == "no_violation" and vs:
        raise InvalidResult("no_violation requires empty violations")
    if verdict == "violation" and not vs:
        raise InvalidResult("violation requires at least one violation")
    return {"verdict": verdict, "violations": [_violation(v) for v in vs]}


def _load_settings():
    from . import config
    raw = os.environ.get("JOB_SETTINGS")
    if not raw:
        return config.get()
    data = json.loads(raw)
    names = {f.name for f in dataclasses.fields(config.Settings)}
    return config.Settings(**{k: v for k, v in data.items() if k in names})


def _limit_memory(mb: int):
    if mb <= 0:
        return
    try:
        import resource
        b = mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (b, b))
    except (ImportError, ValueError, OSError):
        pass  # not supported on this platform (e.g. macOS); parent timeout still applies


def _watch_parent(ppid: int, deadline):
    def loop():
        while True:
            time.sleep(1.0)
            if os.getppid() != ppid:
                os._exit(4)  # parent died
            if deadline and time.time() > deadline + 30:
                os._exit(5)  # parent should have killed us long ago
    threading.Thread(target=loop, daemon=True).start()


def _write(out_path: str, obj: dict):
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, out_path)


def run(in_path: str, out_path: str) -> int:
    deadline = float(os.environ["JOB_DEADLINE"]) if os.environ.get("JOB_DEADLINE") else None
    ppid = os.environ.get("JOB_PARENT_PID")
    if ppid and ppid.isdigit():
        _watch_parent(int(ppid), deadline)
    try:
        settings = _load_settings()
        _limit_memory(settings.worker_mem_mb)
        # No RLIMIT_CPU: CPU time sums over all inference threads, so it would kill
        # legitimate multi-threaded runs early. Wall clock is enforced by the parent
        # (SIGKILL at the deadline + watchdog) and by _watch_parent.
        from . import pipeline
        res = pipeline.analyze(in_path, settings, deadline=deadline)
        try:
            out = {"status": "done", **validate_result(res)}  # extra keys never leave the child
        except InvalidResult:
            out = {"status": "failed", "error": "invalid_result"}
    except Exception as e:  # any error -> failed, never no_violation
        out = {"status": "failed", "error": safe_error(e)}
    _write(out_path, out)
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    return run(argv[0], argv[1])


if __name__ == "__main__":
    sys.exit(main())
