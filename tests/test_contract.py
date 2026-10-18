"""F10: strict output contract of JobStore for malformed analyze_fn results (synthetic JPEGs only)."""
import base64
import copy
import json
import math
import os

import cv2
import numpy as np
import pytest

from app.jobs import JobStore
from helpers import jpeg_b64, settings, wait_store


def _jpeg(w=64, h=48, noise=False):
    img = (np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8) if noise
           else np.full((h, w, 3), 128, np.uint8))
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def b64(b):
    return base64.b64encode(b).decode()


def good(n=1):
    return {"verdict": "violation", "violations": [{
        "type": "t", "timecode_sec": 1.5, "description": "НБРК demo не валидировано",
        "evidence_frames": [{"timestamp_sec": 1.0 + i, "image_base64": jpeg_b64()} for i in range(n)]}]}


def run(tmp_path, res):
    def fn(path, s, deadline=None):
        if isinstance(res, BaseException):
            raise res
        return copy.deepcopy(res)
    store = JobStore(settings(tmp_path), fn)
    try:
        p = tmp_path / (os.urandom(16).hex() + ".upload"); p.write_bytes(b"x")
        jid = store.submit(str(p))
        j = wait_store(store, jid, 15)
        disk = json.load(open(tmp_path / "data" / (jid + ".json")))
        return j, disk
    finally:
        store.close()


def mut(f):
    r = good(); f(r); return r


NEG = {
    "verdict_enum": {"verdict": "maybe", "violations": []},
    "verdict_missing": {"violations": []},
    "verdict_wrong_type": {"verdict": 1, "violations": []},
    "no_violation_with_items": {**good(), "verdict": "no_violation"},
    "violation_with_empty": {"verdict": "violation", "violations": []},
    "violations_not_list": {"verdict": "no_violation", "violations": None},
    "not_a_dict": [1, 2],
    "timecode_nan": mut(lambda r: r["violations"][0].update(timecode_sec=float("nan"))),
    "timecode_inf": mut(lambda r: r["violations"][0].update(timecode_sec=float("inf"))),
    "timecode_negative": mut(lambda r: r["violations"][0].update(timecode_sec=-1)),
    "timecode_str": mut(lambda r: r["violations"][0].update(timecode_sec="1")),
    "timecode_bool": mut(lambda r: r["violations"][0].update(timecode_sec=True)),
    "type_empty": mut(lambda r: r["violations"][0].update(type="")),
    "type_int": mut(lambda r: r["violations"][0].update(type=5)),
    "description_empty": mut(lambda r: r["violations"][0].update(description="  ")),
    "description_none": mut(lambda r: r["violations"][0].update(description=None)),
    "missing_key": mut(lambda r: r["violations"][0].pop("evidence_frames")),
    "violation_not_dict": {"verdict": "violation", "violations": ["x"]},
    "frames_zero": good(0),
    "frames_four": good(4),
    "frames_not_list": mut(lambda r: r["violations"][0].update(evidence_frames="x")),
    "frame_ts_negative": mut(lambda r: r["violations"][0]["evidence_frames"][0].update(timestamp_sec=-0.1)),
    "frame_ts_nan": mut(lambda r: r["violations"][0]["evidence_frames"][0].update(timestamp_sec=float("nan"))),
    "frame_missing_ts": mut(lambda r: r["violations"][0]["evidence_frames"][0].pop("timestamp_sec")),
    "bad_base64": mut(lambda r: r["violations"][0]["evidence_frames"][0].update(image_base64="!!!notb64")),
    "b64_wrong_type": mut(lambda r: r["violations"][0]["evidence_frames"][0].update(image_base64=b"x")),
    "non_jpeg_bytes": mut(lambda r: r["violations"][0]["evidence_frames"][0].update(image_base64=b64(b"PNG" * 100))),
    "tiny_jpeg_magic": mut(lambda r: r["violations"][0]["evidence_frames"][0].update(image_base64=b64(b"\xff\xd8" + b"\0" * 5))),
    # F10 addendum: magic + length pass, but not decodable
    "magic_plus_zeros_not_decodable": mut(lambda r: r["violations"][0]["evidence_frames"][0].update(
        image_base64=b64(b"\xff\xd8" + b"\x00" * 200))),
    "truncated_real_jpeg_60pct": mut(lambda r: r["violations"][0]["evidence_frames"][0].update(
        image_base64=b64(_jpeg(160, 120, noise=True)[: int(len(_jpeg(160, 120, noise=True)) * 0.6)]))),
    "oversized_evidence_gt_2mb": mut(lambda r: r["violations"][0]["evidence_frames"][0].update(
        image_base64=b64(_jpeg(64, 48) + b"\x00" * (2 * 1024 * 1024 + 10)))),
}


@pytest.mark.parametrize("name", sorted(NEG))
def test_malformed_result_is_invalid_result(tmp_path, name):
    j, disk = run(tmp_path, NEG[name])
    for rec in (j, disk):
        assert rec["status"] == "failed" and rec["error"] == "invalid_result", f"{name}: {rec.get('status')}/{rec.get('error')}"
        assert "verdict" not in rec and "violations" not in rec


def test_worker_exception_is_failed(tmp_path):
    j, disk = run(tmp_path, RuntimeError("x /Users/a/b.mov"))
    for rec in (j, disk):
        assert rec["status"] == "failed" and "verdict" not in rec and "/Users" not in json.dumps(rec)


def test_too_many_violations_rejected(tmp_path):
    r = good(); r["violations"] = r["violations"] * 51
    j, _ = run(tmp_path, r)
    assert j["status"] == "failed" and j["error"] == "invalid_result"


def test_valid_result_done_and_public_schema_not_extended(tmp_path):
    r = good(3)
    r["observations"] = [{"secret": 1}]; r["meta"] = {"path": "/x"}
    r["violations"][0].update(extra="no", observed={"a": 1}, confidence=0.9, width=64)
    r["violations"][0]["evidence_frames"][0].update(width=64, height=48)
    j, disk = run(tmp_path, r)
    for rec in (j, disk):
        assert rec["status"] == "done" and rec["verdict"] == "violation"
        assert set(rec) == {"job_id", "status", "verdict", "violations"}
        e = rec["violations"][0]
        assert set(e) == {"type", "timecode_sec", "description", "evidence_frames"}
        assert len(e["evidence_frames"]) == 3
        for f in e["evidence_frames"]:
            assert set(f) == {"timestamp_sec", "image_base64"}
            assert cv2.imdecode(np.frombuffer(base64.b64decode(f["image_base64"]), np.uint8), 1) is not None
        assert "observations" not in json.dumps(rec) and "/x" not in json.dumps(rec)


def test_no_violation_valid_done(tmp_path):
    j, disk = run(tmp_path, {"verdict": "no_violation", "violations": [], "meta": {"k": 1}})
    for rec in (j, disk):
        assert rec == {"job_id": rec["job_id"], "status": "done", "verdict": "no_violation", "violations": []}


def test_valid_noise_jpeg_accepted(tmp_path):
    r = good(); r["violations"][0]["evidence_frames"][0]["image_base64"] = b64(_jpeg(160, 120, noise=True))
    assert run(tmp_path, r)[0]["status"] == "done"


def _sized(w, h):
    ok, buf = cv2.imencode(".jpg", np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _with(img):
    r = good(); r["violations"][0]["evidence_frames"][0]["image_base64"] = b64(img); return r


@pytest.mark.parametrize("w,h", [(8, 8), (5000, 64), (64, 5000), (15, 64)])
def test_dimension_bounds_rejected(tmp_path, w, h):
    j, disk = run(tmp_path, _with(_sized(w, h)))
    for rec in (j, disk):
        assert rec["status"] == "failed" and rec["error"] == "invalid_result"


@pytest.mark.parametrize("w,h", [(64, 64), (16, 16)])
def test_dimension_bounds_accepted(tmp_path, w, h):
    assert run(tmp_path, _with(_sized(w, h)))[0]["status"] == "done"
