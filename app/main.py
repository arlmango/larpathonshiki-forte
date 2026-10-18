import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .auth import key_matches
from .jobs import JobStore, QueueFull

log = logging.getLogger("app.api")
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_STATIC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))


_SEC_HEADERS = [(b"cache-control", b"no-store"), (b"pragma", b"no-cache"),
                (b"x-content-type-options", b"nosniff")]


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _is_api(path: str) -> bool:
    return path == "/api" or path.startswith("/api/")


def _api_headers(msg_headers) -> list:
    have = {k.lower() for k, _ in msg_headers}
    return list(msg_headers) + [h for h in _SEC_HEADERS if h[0] not in have]


def _require_empty(data_dir: str) -> None:
    """Fail closed (never deletes): refuse to start if data_dir holds anything."""
    try:
        names = os.listdir(data_dir)
    except OSError:
        return  # missing dir is empty
    if names:
        raise RuntimeError("DATA_DIR is not empty; use a fresh empty directory per launch")


class _Bucket:
    """Per-key token bucket (capacity = per-minute rate). rate 0 disables."""

    def __init__(self, per_min: int):
        self.cap, self.rate = float(per_min), per_min / 60.0
        self.b: dict = {}

    def allow(self, key: str) -> bool:
        if self.cap <= 0:
            return True
        now = time.monotonic()
        tok, last = self.b.get(key, (self.cap, now))
        tok = min(self.cap, tok + (now - last) * self.rate)
        ok = tok >= 1.0
        self.b[key] = (tok - 1.0 if ok else tok, now)
        if len(self.b) > 10000:  # bound memory: drop full (idle) buckets
            for k in [k for k, (t, _) in self.b.items() if t >= self.cap - 1e-9]:
                self.b.pop(k, None)
            if len(self.b) > 10000:
                self.b.clear()
        return ok


class _Abort(Exception):
    def __init__(self, code: int, detail: str):
        self.code, self.detail = code, detail


class _Guard:
    """Pure ASGI middleware. Records request start (upload time counts toward the job
    budget), logs method/status/elapsed only, and for POST /api/v1/analyze: checks
    X-API-Key before reading any body byte, caps Content-Length and cumulative body
    bytes, and bounds per-chunk and total receive time (408/413 static)."""

    def __init__(self, app, api_key: str, max_upload_mb: int,
                 receive_timeout: float, upload_timeout: float):
        self.app = app
        self.max_inflight = _env_int("MAX_INFLIGHT_UPLOADS", 4)
        self.inflight = 0
        self.trust_proxy = os.environ.get("TRUST_PROXY") == "1"
        self.post_rl = _Bucket(_env_int("RATE_LIMIT_PER_MIN", 30))
        self.poll_rl = _Bucket(_env_int("POLL_LIMIT_PER_MIN", 600))
        self.key = api_key.encode()
        # multipart framing overhead slack; the route still enforces the exact file cap
        self.cap = max_upload_mb * 1024 * 1024 + (1 << 20)
        self.chunk_to, self.total_to = receive_timeout, upload_timeout

    async def _reply(self, send, code: int, detail: str):
        body = json.dumps({"detail": detail}).encode()
        await send({"type": "http.response.start", "status": code,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"connection", b"close")] + _SEC_HEADERS})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        t0 = time.time()
        scope.setdefault("state", {})["t0"] = t0
        code = {"v": 500}

        abort = []  # first _Abort raised from receive; FastAPI may rewrap it as 400

        async def _send(msg):
            if abort:
                return  # swallow the app's rewrapped error response
            if msg["type"] == "http.response.start":
                code["v"] = msg["status"]
                if api:
                    msg = {**msg, "headers": _api_headers(msg.get("headers", []))}
            await send(msg)

        guarded = scope["method"] == "POST" and scope["path"] == "/api/v1/analyze"
        api = _is_api(scope["path"])
        ip = (scope.get("client") or ("?", 0))[0]  # socket peer only; XFF/X-Real-IP ignored
        if self.trust_proxy:  # explicit opt-in: first X-Forwarded-For hop
            xff = dict(scope["headers"]).get(b"x-forwarded-for", b"").split(b",")[0].strip()
            ip = xff.decode("latin-1")[:64] or ip
        counted = False
        try:
            if scope["method"] == "GET" and scope["path"].startswith("/api/v1/jobs/"):
                if not self.poll_rl.allow(ip):
                    return await self._reply(_send, 429, "too many requests")
            if not guarded:
                return await self.app(scope, receive, _send)
            hdr = {k.lower(): v for k, v in scope["headers"]}
            if not key_matches(hdr.get(b"x-api-key", b""), self.key):
                return await self._reply(_send, 401, "invalid api key")
            try:
                clen = int(hdr.get(b"content-length", b"0") or 0)
            except ValueError:
                return await self._reply(_send, 400, "bad request")
            if clen > self.cap:
                return await self._reply(_send, 413, "file too large")
            if not self.post_rl.allow(ip):
                return await self._reply(_send, 429, "too many requests")
            if self.max_inflight and self.inflight >= self.max_inflight:
                return await self._reply(_send, 429, "too many concurrent uploads")
            self.inflight += 1
            counted = True
            seen = 0

            async def _receive():
                nonlocal seen
                remaining = self.total_to - (time.time() - t0)
                if remaining <= 0:
                    abort.append(_Abort(408, "upload timeout"))
                    raise abort[0]
                try:
                    msg = await asyncio.wait_for(receive(), min(self.chunk_to, remaining))
                except asyncio.TimeoutError:
                    abort.append(_Abort(408, "upload timeout"))
                    raise abort[0] from None
                if msg["type"] == "http.request":
                    seen += len(msg.get("body", b""))
                    if seen > self.cap:
                        abort.append(_Abort(413, "file too large"))
                        raise abort[0]
                return msg

            try:
                await self.app(scope, _receive, _send)
            except _Abort:
                pass
            if abort:
                e = abort[0]
                code["v"] = e.code
                await self._reply(send, e.code, e.detail)
        finally:
            if counted:
                self.inflight -= 1
            log.info("%s status=%s elapsed=%.2fs", scope.get("method"), code["v"], time.time() - t0)


def create_app(settings=None, analyze_fn=None) -> FastAPI:
    s = settings or config.get()
    if not s.api_key:
        raise RuntimeError("API_KEY env var must be set")
    if os.environ.get("REQUIRE_EMPTY_DATA_DIR") == "1":
        _require_empty(s.data_dir)
    store = JobStore(s, analyze_fn)
    bind = os.environ.get("BIND_JOBS_TO_KEY", "1") != "0"
    owners: "OrderedDict[str, str]" = OrderedDict()  # job_id -> sha256(key); in memory only

    def _kh(req: Request) -> str:
        return hashlib.sha256(req.headers.get("x-api-key", "").encode()).hexdigest()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            store.close()

    app = FastAPI(title="ForteBank Datathon26 cashier video analyzer", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store
    app.add_middleware(_Guard, api_key=s.api_key, max_upload_mb=s.max_upload_mb,
                       receive_timeout=s.receive_timeout_sec,
                       upload_timeout=s.upload_timeout_sec)

    def _api_h(req):
        if _is_api(req.url.path):
            return {"Cache-Control": "no-store", "Pragma": "no-cache",
                    "X-Content-Type-Options": "nosniff"}
        return None

    @app.exception_handler(RequestValidationError)
    async def _invalid(_req, _exc):  # static body: never echo input
        return JSONResponse({"detail": "invalid request"}, status_code=422,
                            headers=_api_h(_req))

    @app.exception_handler(Exception)
    async def _crash(_req, _exc):  # static body: never echo exception text
        return JSONResponse({"detail": "internal error"}, status_code=500,
                            headers=_api_h(_req))

    def auth(x_api_key: str = Header(default="")):
        if not key_matches(x_api_key.encode(), s.api_key.encode()):
            raise HTTPException(401, "invalid api key")

    @app.post("/api/v1/analyze", dependencies=[Depends(auth)])
    async def analyze(request: Request, file: UploadFile = File(...)):
        t0 = getattr(request.state, "t0", None) or time.time()
        ctype = (file.content_type or "").split(";")[0].strip().lower()
        if ctype and not (ctype.startswith("video/") or ctype == "application/octet-stream"):
            raise HTTPException(415, "unsupported media type")
        os.makedirs(s.data_dir, exist_ok=True)
        path = os.path.join(s.data_dir, uuid.uuid4().hex + ".upload")
        size, limit = 0, s.max_upload_mb * 1024 * 1024
        submitted = False
        try:
            with open(path, "wb") as f:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, "file too large")
                    f.write(chunk)
            if size == 0:
                raise HTTPException(400, "empty file")
            try:
                jid = store.submit(path, submitted_at=t0)  # store removes file on rejection
                submitted = True
            except QueueFull:
                submitted = True
                raise HTTPException(429, "queue full, retry later")
        finally:
            if not submitted:
                try:
                    os.remove(path)
                except OSError:
                    pass
        if bind:
            owners[jid] = _kh(request)
            while len(owners) > 10000:
                owners.popitem(last=False)
        log.info("job_id=%s accepted", jid)
        return {"job_id": jid, "status": "processing"}

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(auth)])
    def job(job_id: str, request: Request):
        j = store.get(job_id) if JOB_ID_RE.match(job_id) else None
        if j is not None and bind and owners.get(job_id, _kh(request)) != _kh(request):
            j = None  # job belongs to another key: indistinguishable from not found
        if j is None:
            return JSONResponse({"detail": "job not found"}, status_code=404)
        log.info("job_id=%s status=%s", job_id, j.get("status"))
        return j

    @app.get("/api/v1/model", dependencies=[Depends(auth)])
    def model():
        try:
            from . import pipeline
            info = {**pipeline.describe_model(s), "available": True}
            if "rules_error" in info:  # defence in depth: never pass through free-form text
                info["rules_error"] = "rules_unavailable"
        except Exception as e:
            log.warning("model info failed: %s", type(e).__name__)
            info = {"available": False, "error": "model_unavailable"}
        return {
            "model": info,
            "queue": store.stats(),
            "limits": {"max_upload_mb": s.max_upload_mb, "max_duration_sec": s.max_duration_sec,
                       "job_budget_sec": s.job_budget_sec},
        }

    @app.get("/healthz")
    def health():
        return {"ok": True}

    if os.path.isdir(_STATIC):
        app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    @app.get("/")
    def index():
        p = os.path.join(_STATIC, "index.html")
        if not os.path.isfile(p):
            return JSONResponse({"detail": "not found"}, status_code=404)
        return FileResponse(p)

    return app
