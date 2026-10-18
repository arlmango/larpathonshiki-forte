"""Full HTTP/child/decoder smoke tests. Only generated synthetic clips are used.

The opt-in rules are temporary test copies; production rules remain disabled.
The optional pinned ONNX case verifies execution, never detection accuracy.
"""
import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import detector
from app.main import create_app
from helpers import KEY, clip, leftovers, post, rules_file, settings, wait_http


def _repeated_clip(tmp_path):
    source = clip(tmp_path, absent=(12, 34))
    out = tmp_path / "two_events.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    assert writer.isOpened()
    try:
        for _ in range(2):
            cap = cv2.VideoCapture(source)
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    writer.write(frame)
            finally:
                cap.release()
    finally:
        writer.release()
    return out


def test_http_real_child_multiple_events_and_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_UNVALIDATED_RULES", "1")
    video = _repeated_clip(tmp_path)
    s = settings(tmp_path, ruleset=rules_file(tmp_path), workers=1)
    with TestClient(create_app(s)) as client:
        response = post(client, video.read_bytes(), name="synthetic.mp4")
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        assert client.get(f"/api/v1/jobs/{job_id}").status_code == 401
        result = wait_http(client, job_id, timeout=90)
        assert result.get("status") == "done", result.get("error")
        assert result.get("verdict") == "violation"
        assert set(result) == {"job_id", "status", "verdict", "violations"}
        events = result["violations"]
        assert len(events) == 2
        times = [e["timecode_sec"] for e in events]
        assert times == sorted(times) and times[1] - times[0] > 20
        for event in events:
            assert set(event) == {"type", "timecode_sec", "description", "evidence_frames"}
            assert "не валидировано" in event["description"].lower()
            assert 1 <= len(event["evidence_frames"]) <= 3
            for evidence in event["evidence_frames"]:
                assert set(evidence) == {"timestamp_sec", "image_base64"}
                assert 0 <= evidence["timestamp_sec"] <= 80
                raw = base64.b64decode(evidence["image_base64"], validate=True)
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                assert image is not None and min(image.shape[:2]) >= 16
    assert leftovers(s.data_dir) == []


def test_http_real_child_pinned_onnx_synthetic_input(tmp_path, monkeypatch):
    """No injected analyzer/detector: the child must load and run pinned YOLOX-S.

    Blank synthetic content has no person detections. Reaching that explicit
    error verifies decoding and inference; it must never become no_violation.
    """
    model = Path(detector.WEIGHTS_DIR) / detector.MODELS["yolox_s"]["file"]
    if not model.exists():
        pytest.skip("optional local pinned YOLOX-S weights not installed")
    assert detector.sha256_file(str(model)) == detector.MODELS["yolox_s"]["sha256"]
    monkeypatch.delenv("SYNTHETIC_DETECTOR", raising=False)
    monkeypatch.setenv("ALLOW_UNVALIDATED_RULES", "1")
    video = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    assert writer.isOpened()
    try:
        for _ in range(30):
            writer.write(np.full((360, 640, 3), 128, np.uint8))
    finally:
        writer.release()
    s = settings(tmp_path, ruleset=rules_file(tmp_path), onnx_model=str(model), workers=1)
    with TestClient(create_app(s)) as client:
        response = post(client, video.read_bytes(), name="blank.mp4")
        assert response.status_code == 200
        result = wait_http(client, response.json()["job_id"], timeout=90)
        assert result.get("status") == "failed"
        assert result.get("error") == "too_few_detections"
        assert set(result) == {"job_id", "status", "error"}
    assert leftovers(s.data_dir) == []
