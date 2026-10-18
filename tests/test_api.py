"""HTTP API incl. F8: ASGI streaming slow-upload, early 413, key checked before body."""
import asyncio
import os

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from helpers import KEY, clip, inproc, post, settings, wait_http

MP = {"content-type": "multipart/form-data; boundary=BND"}


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _noresidue(d):
    return not os.path.isdir(d) or [n for n in os.listdir(d)] == []


def test_slow_upload_408_no_orphans(tmp_path, monkeypatch):
    app = create_app(settings(tmp_path, receive_timeout_sec=0.3, upload_timeout_sec=5))

    async def body():
        yield b"--BND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp4\"\r\n"
        for _ in range(5):
            await asyncio.sleep(1.0)  # longer than the receive timeout
            yield b"x"

    async def go():
        async with _client(app) as c:
            return await c.post("/api/v1/analyze", headers={**MP, "X-API-Key": KEY}, content=body())
    r = asyncio.run(go())
    assert r.status_code == 408 and r.json() == {"detail": "upload timeout"}
    assert not [n for n in os.listdir(tmp_path / "data") if n.endswith((".upload", ".json"))]


def test_total_upload_timeout_with_fast_trickle(tmp_path, monkeypatch):
    app = create_app(settings(tmp_path, receive_timeout_sec=5, upload_timeout_sec=1))

    async def body():
        yield b"--BND\r\n"
        for _ in range(50):
            await asyncio.sleep(0.2)
            yield b"x" * 10
    async def go():
        async with _client(app) as c:
            return await c.post("/api/v1/analyze", headers={**MP, "X-API-Key": KEY}, content=body())
    assert asyncio.run(go()).status_code == 408
    assert not [n for n in os.listdir(tmp_path / "data") if n.endswith((".upload", ".json"))]


def test_oversized_stream_no_content_length_413_early(tmp_path):
    app = create_app(settings(tmp_path, max_upload_mb=1))
    pulled = {"n": 0}

    async def body():
        yield b"--BND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp4\"\r\n\r\n"
        for _ in range(200):
            pulled["n"] += 1
            yield b"0" * (256 * 1024)
    async def go():
        async with _client(app) as c:
            return await c.post("/api/v1/analyze", headers={**MP, "X-API-Key": KEY}, content=body())
    r = asyncio.run(go())
    assert r.status_code == 413
    assert pulled["n"] < 20, f"generator consumed {pulled['n']}/200 chunks: cap not enforced before parsing"
    assert not [n for n in os.listdir(tmp_path / "data") if n.endswith((".upload", ".json"))]


def test_bad_key_large_body_401_body_not_read(tmp_path):
    app = create_app(settings(tmp_path))
    pulled = {"n": 0}

    async def body():
        for _ in range(100):
            pulled["n"] += 1
            yield b"0" * (256 * 1024)
    async def go():
        async with _client(app) as c:
            return await c.post("/api/v1/analyze", headers={**MP, "X-API-Key": "bad"}, content=body())
    r = asyncio.run(go())
    assert r.status_code == 401
    assert pulled["n"] <= 1, f"{pulled['n']} chunks pulled before key check"


def test_bad_key_and_job_ids(tmp_path):
    c = TestClient(create_app(settings(tmp_path)))
    assert post(c, b"x", key="bad").status_code == 401
    assert c.get("/api/v1/jobs/abc").status_code == 401
    for bad in ("abc", "../../etc/passwd", "Z" * 32, "0" * 31, "0" * 33):
        assert c.get(f"/api/v1/jobs/{bad}", headers={"X-API-Key": KEY}).status_code == 404, bad
    assert c.get("/api/v1/jobs/" + "0" * 32, headers={"X-API-Key": KEY}).status_code == 404


def test_text_plain_415(tmp_path):
    c = TestClient(create_app(settings(tmp_path)))
    r = post(c, b"hello", name="a.txt", ctype="text/plain")
    assert r.status_code == 415
    assert not [n for n in os.listdir(tmp_path / "data") if n.endswith(".upload")]


def test_upload_too_large_413_and_empty_400(tmp_path):
    c = TestClient(create_app(settings(tmp_path, max_upload_mb=1)))
    assert post(c, b"0" * (3 << 20)).status_code == 413
    assert post(c, b"").status_code == 400
    assert not [n for n in os.listdir(tmp_path / "data") if n.endswith(".upload")]


def test_model_endpoint_has_rules(tmp_path):
    c = TestClient(create_app(settings(tmp_path)))
    assert c.get("/api/v1/model").status_code == 401
    r = c.get("/api/v1/model", headers={"X-API-Key": KEY})
    assert r.status_code == 200
    m = r.json()["model"]
    assert "rules" in m and isinstance(m["rules"], list) and m["rules"]
    assert all(not x["active"] for x in m["rules"])  # default: nothing active


def test_corrupt_upload_end_to_end(tmp_path):
    c = TestClient(create_app(settings(tmp_path)))
    j = wait_http(c, post(c, b"not a video at all" * 100).json()["job_id"])
    assert j["status"] == "failed" and j["error"] == "corrupt_video" and "verdict" not in j


def test_default_rules_clip_is_failed_no_validated_rule_e2e(tmp_path):
    v = open(clip(tmp_path, dur=6), "rb").read()
    c = TestClient(create_app(settings(tmp_path)))
    j = wait_http(c, post(c, v).json()["job_id"])
    assert j["status"] == "failed" and j["error"] == "no_validated_rule" and "verdict" not in j


def test_analyzer_crash_failed(tmp_path):
    def boom(path, s, deadline=None):
        raise RuntimeError("boom")
    c = TestClient(create_app(settings(tmp_path), boom))
    j = wait_http(c, post(c, b"x").json()["job_id"])
    assert j["status"] == "failed" and "verdict" not in j and "boom" not in str(j)
