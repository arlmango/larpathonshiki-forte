"""Stage 6 API hardening: headers, hidden docs, static isolation, limits, clean-start.
Fake values only (key 'k' from helpers == test-fake, tmp DATA_DIR, synthetic bytes)."""
import asyncio
import os
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from helpers import KEY, post, settings

HDRS = {"cache-control": "no-store", "pragma": "no-cache", "x-content-type-options": "nosniff"}
MP = {"content-type": "multipart/form-data; boundary=BND"}
VIDEO = b"\x00\x00\x00\x18ftypmp42" + b"\x01" * 2048  # synthetic bytes, not a real video
H = {"X-API-Key": KEY}


def _fast(path, s, deadline=None):
    return {"status": "done", "result": {}}


def _mk(tmp_path, fn=_fast, **kw):
    return create_app(settings(tmp_path, **kw), analyze_fn=fn)


def _check(r, where):
    got = {k.lower(): v.lower() for k, v in r.headers.items()}
    for k, v in HDRS.items():
        assert got.get(k) == v, f"{where}: header {k} -> {got.get(k)!r} (status {r.status_code})"


def _no_paths(r):
    t = r.text
    assert "/Users/" not in t and "local_private" not in t, "path leak in body"


def _async(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


# (1) security headers on every /api/* response -------------------------------
def test_headers_ok_200_model_and_job_404_and_health(tmp_path):
    c = TestClient(_mk(tmp_path))
    for r, w in ((c.get("/api/v1/model", headers=H), "200 model"),
                 (c.get("/api/v1/jobs/" + "0" * 32, headers=H), "404 job"),
                 (c.get("/api/v1/jobs/notahexid", headers=H), "404 badid")):
        _check(r, w)
        _no_paths(r)
    assert c.get("/api/v1/model", headers=H).status_code == 200
    assert c.get("/api/v1/jobs/" + "0" * 32, headers=H).status_code == 404


def test_headers_401_bad_key_get_and_post(tmp_path):
    c = TestClient(_mk(tmp_path))
    r = c.get("/api/v1/model", headers={"X-API-Key": "wrong-fake"})
    assert r.status_code == 401
    _check(r, "401 get")
    r = post(c, VIDEO, key="wrong-fake")
    assert r.status_code == 401
    _check(r, "401 post (guard)")
    r = c.get("/api/v1/jobs/" + "0" * 32)  # missing key
    assert r.status_code == 401
    _check(r, "401 missing")


def test_headers_400_empty_415_422(tmp_path):
    c = TestClient(_mk(tmp_path))
    r = post(c, b"")
    assert r.status_code == 400
    _check(r, "400 empty")
    r = post(c, VIDEO, name="a.txt", ctype="text/plain")
    assert r.status_code == 415
    _check(r, "415")
    r = c.post("/api/v1/analyze", headers=H, data={"x": "y"})
    assert r.status_code == 422
    _check(r, "422 missing field")
    _no_paths(r)


def test_headers_413_content_length(tmp_path):
    c = TestClient(_mk(tmp_path, max_upload_mb=1))
    r = c.post("/api/v1/analyze", headers={**H, **MP, "Content-Length": str(50 * 1024 * 1024)},
               content=b"--BND--")
    assert r.status_code == 413
    _check(r, "413 content-length")
    r = post(c, b"\x01" * (3 * 1024 * 1024))
    assert r.status_code == 413
    _check(r, "413 body")


def test_headers_408_slow(tmp_path):
    app = _mk(tmp_path, receive_timeout_sec=0.3, upload_timeout_sec=5)

    async def body():
        yield b"--BND\r\n"
        for _ in range(4):
            await asyncio.sleep(1.0)
            yield b"x"

    async def go():
        async with _async(app) as c:
            return await c.post("/api/v1/analyze", headers={**MP, **H}, content=body())
    r = asyncio.run(go())
    assert r.status_code == 408
    _check(r, "408")


def test_headers_on_accepted_job(tmp_path):
    c = TestClient(_mk(tmp_path))
    r = post(c, VIDEO, name="a.mp4", ctype="video/mp4")
    assert r.status_code == 200
    _check(r, "accept")
    _no_paths(r)


# (2) docs hidden --------------------------------------------------------------
@pytest.mark.parametrize("p", ["/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"])
def test_docs_disabled(tmp_path, p):
    r = TestClient(_mk(tmp_path)).get(p)
    assert r.status_code == 404, p
    assert "openapi" not in r.text.lower() and "swagger" not in r.text.lower()


# (3) traversal / private paths ---------------------------------------------
PATHS = ["/static/../local_private/inventory_map.json", "/static/%2e%2e/%2e%2e/app/main.py",
         "/static//etc/passwd", "/static/..%2fapp/main.py", "/static/%2e%2e%2flocal_private/x",
         "/local_private/x", "/local_private/inventory_map.json", "/data/x", "/data/jobs/x.json",
         "/weights/x", "/weights/model.onnx", "/docs/QUALITY.md", "/tools/x", "/app/main.py",
         "/tools/make_synthetic.py", "/tests/test_hardening.py", "/.git/config", "/.env"]


@pytest.mark.parametrize("p", PATHS)
def test_traversal_and_private_paths_404(tmp_path, p):
    c = TestClient(_mk(tmp_path))
    r = c.get(p)
    assert r.status_code == 404, f"{p} -> {r.status_code}"
    assert "def create_app" not in r.text and "root:" not in r.text
    _no_paths(r)
    r2 = c.get(p, headers=H)
    assert r2.status_code == 404


def test_traversal_raw_target_not_normalised(tmp_path):
    async def go():
        async with _async(_mk(tmp_path)) as c:
            out = []
            for t in ("/static/../app/main.py", "/static/%2e%2e/app/main.py", "/static//etc/passwd"):
                req = c.build_request("GET", "http://t/x")
                req.url = httpx.URL("http://t" + t)
                out.append((t, (await c.send(req)).status_code))
            return out
    for t, code in asyncio.run(go()):
        assert code != 200 and code in (400, 404), (t, code)


# (4) job-key boundary --------------------------------------------------------
def test_job_key_boundary_and_forged_ids(tmp_path):
    gate = threading.Event()

    def blocking(path, s, deadline=None):
        gate.wait(20)
        return {"status": "done", "result": {}}
    app = _mk(tmp_path, blocking)
    try:
        with TestClient(app) as c:
            r = post(c, VIDEO, name="a.mp4", ctype="video/mp4")
            assert r.status_code == 200
            jid = r.json()["job_id"]
            g = c.get(f"/api/v1/jobs/{jid}", headers=H)
            assert g.status_code == 200 and g.json()["status"] == "processing"
            assert c.get(f"/api/v1/jobs/{jid}", headers={"X-API-Key": "wrong-fake"}).status_code == 401
            assert c.get(f"/api/v1/jobs/{jid}").status_code == 401
            for forged in ("../" + jid, jid.upper(), jid + "0", jid[:-1], "%2e%2e%2f" + jid,
                           jid + ".json", "f" * 32):
                assert c.get(f"/api/v1/jobs/{forged}", headers=H).status_code == 404, forged
            # durable record / upload never reachable via static or data paths
            for p in (f"/static/{jid}.json", f"/data/{jid}.json", f"/data/jobs/{jid}.json",
                      f"/static/../data/{jid}.json", f"/{jid}.json"):
                assert c.get(p, headers=H).status_code == 404, p
    finally:
        gate.set()


# (5) limits -------------------------------------------------------------------
def test_inflight_uploads_limit_429_static(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_INFLIGHT_UPLOADS", "1")
    app = _mk(tmp_path)

    async def go():
        ev = asyncio.Event()

        async def slow():
            yield b"--BND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp4\"\r\n\r\n"
            await ev.wait()
            yield b"x\r\n--BND--\r\n"
        async with _async(app) as c:
            t1 = asyncio.create_task(c.post("/api/v1/analyze", headers={**MP, **H}, content=slow()))
            await asyncio.sleep(0.5)
            r2 = await c.post("/api/v1/analyze", headers=H,
                              files={"file": ("a.mp4", VIDEO, "video/mp4")})
            ev.set()
            await asyncio.wait_for(t1, 20)
            return r2
    r2 = asyncio.run(go())
    assert r2.status_code == 429, r2.status_code
    assert r2.json() == {"detail": r2.json()["detail"]} and len(r2.text) < 120
    _check(r2, "429 inflight")


def test_rate_limit_per_min_third_post_429(tmp_path, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    c = TestClient(_mk(tmp_path))
    codes = [post(c, VIDEO, name="a.mp4", ctype="video/mp4").status_code for _ in range(3)]
    assert codes[2] == 429 and codes[0] != 429 and codes[1] != 429, codes
    r = post(c, VIDEO, name="a.mp4", ctype="video/mp4")
    assert r.status_code == 429
    _check(r, "429 rate")


def test_poll_limit_per_min_fourth_get_429(tmp_path, monkeypatch):
    monkeypatch.setenv("POLL_LIMIT_PER_MIN", "3")
    c = TestClient(_mk(tmp_path))
    codes = [c.get("/api/v1/jobs/" + "0" * 32, headers=H).status_code for _ in range(4)]
    assert codes[:3] == [404] * 3 and codes[3] == 429, codes
    _check(c.get("/api/v1/jobs/" + "0" * 32, headers=H), "429 poll")


# (6) REQUIRE_EMPTY_DATA_DIR (no deletion mechanism) ---------------------------
def test_require_empty_data_dir_refuses_nonempty(tmp_path, monkeypatch):
    monkeypatch.setenv("REQUIRE_EMPTY_DATA_DIR", "1")
    d = tmp_path / "data"
    d.mkdir()
    f = d / "unrelated.txt"
    f.write_bytes(b"keep")
    with pytest.raises(RuntimeError):
        _mk(tmp_path)
    assert f.read_bytes() == b"keep" and os.listdir(d) == ["unrelated.txt"]


def test_require_empty_data_dir_empty_and_missing_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("REQUIRE_EMPTY_DATA_DIR", "1")
    (tmp_path / "data").mkdir()
    with TestClient(_mk(tmp_path)) as c:
        assert c.get("/healthz").status_code == 200


def test_default_unset_starts_and_deletes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("REQUIRE_EMPTY_DATA_DIR", raising=False)
    d = tmp_path / "data"
    d.mkdir()
    names = {"notes.txt": b"k", "a" * 32 + ".json": b"{}", "b" * 32 + ".upload": b"u", "x.tmp": b"t"}
    for n, b in names.items():
        (d / n).write_bytes(b)
    with TestClient(_mk(tmp_path)) as c:
        assert c.get("/healthz").status_code == 200
    for n, b in names.items():
        assert (d / n).read_bytes() == b, n


# (8) rate-limit identity: spoofed forwarding headers are not trusted -------
def test_rate_limit_ignores_spoofed_forwarding_headers(tmp_path, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    c = TestClient(_mk(tmp_path))
    codes = []
    for i in range(3):
        h = {**H, "X-Forwarded-For": f"10.9.8.{i}", "X-Real-IP": f"10.7.7.{i}"}
        r = c.post("/api/v1/analyze", headers=h, files={"file": ("a.mp4", VIDEO, "video/mp4")})
        codes.append(r.status_code)
    assert codes[2] == 429 and 429 not in codes[:2], codes


# (7) no real paths in error bodies -------------------------------------------
def test_no_paths_in_any_error_body(tmp_path):
    c = TestClient(_mk(tmp_path, max_upload_mb=1))
    rs = [c.get("/api/v1/jobs/zz", headers=H), c.get("/api/v1/model", headers={"X-API-Key": "bad-fake"}),
          post(c, b""), post(c, VIDEO, name="a.txt", ctype="text/plain"),
          c.post("/api/v1/analyze", headers=H), c.get("/nonexistent"), c.get("/static/nope"),
          post(c, b"\x01" * (3 * 1024 * 1024))]
    for r in rs:
        _no_paths(r)
