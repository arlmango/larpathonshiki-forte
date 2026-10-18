"""Synthetic-only auth, hard-worker bounds and private-job isolation regressions."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import pytest
from fastapi.testclient import TestClient

from research_fixtures import Probe
from tools.research_runtime import JobBusy, JobFailure, JobTimeout, ResearchRunner
from tools.stage15_api import create_research_app

KEY = "fixture-only"
ROUTE = "/research/stage15/predict"


@pytest.fixture
def private_root(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir(mode=0o700)
    return root


def test_missing_key_rejected_at_startup(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API_KEY"):
        create_research_app()


@pytest.mark.parametrize("headers", [[], [(b"x-api-key", b"invalid")],
    [(b"x-api-key", KEY.encode()), (b"x-api-key", KEY.encode())]])
def test_unauthorized_before_body_or_provider(headers, private_root, tmp_path):
    marker = tmp_path / "called"
    app = create_research_app(Probe(marker=str(marker)), api_key=KEY, work_root=private_root)
    sent = []
    async def receive():
        pytest.fail("unauthorized request body was consumed")
    async def send(message):
        sent.append(message)
    asyncio.run(app({"type":"http", "asgi":{"version":"3.0"}, "http_version":"1.1",
        "method":"POST", "scheme":"http", "path":ROUTE, "raw_path":ROUTE.encode(),
        "query_string":b"", "headers":headers, "client":("127.0.0.1", 1),
        "server":("testserver", 80)}, receive, send))
    assert sent[0]["status"] == 401
    assert not marker.exists()
    assert not list(private_root.iterdir())
    app.state.runner.close()


def _assert_no_live_group(pgid):
    result = subprocess.run(["/bin/ps", "-axo", "pgid=,stat="], capture_output=True,
                            text=True, check=True, timeout=2)
    assert not any(int(parts[0]) == pgid and not parts[1].startswith("Z")
                   for line in result.stdout.splitlines() if (parts := line.split()))


def test_timeout_kills_worker_and_cleans_job(private_root, tmp_path):
    marker = tmp_path / "started"
    app = create_research_app(Probe("sleep", str(marker)), api_key=KEY,
                              timeout=.8, memory_mb=64, work_root=private_root)
    started = time.monotonic()
    with TestClient(app) as client:
        response = client.post(ROUTE, headers={"X-API-Key":KEY}, json={"sample_id":"synthetic"})
    assert response.status_code == 504
    assert response.json() == {"detail":"research timeout"}
    assert time.monotonic() - started < 4
    record = json.loads(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(record["pid"], 0)
    assert not Path(record["cwd"]).exists()
    assert not list(private_root.iterdir())


def test_descendants_killed_after_successful_leader_exit(private_root, tmp_path):
    marker = tmp_path / "started"
    runner = ResearchRunner(Probe("descendant", str(marker)), timeout=3, memory_mb=128,
                            work_root=private_root)
    try:
        assert runner.run("synthetic")["clip_logit"] == 1
        _assert_no_live_group(json.loads(marker.read_text())["pid"])
        assert not list(private_root.iterdir())
    finally:
        runner.close()


def test_child_caps_private_modes_and_sanitized_environment(private_root, monkeypatch):
    monkeypatch.setenv("API_KEY", "fixture-secret-never-forward")
    monkeypatch.setenv("UNRELATED_TEST_SECRET", "fixture-secret-never-forward")
    runner = ResearchRunner(Probe("caps"), timeout=3, memory_mb=64, work_root=private_root)
    try:
        assert runner.run("synthetic") == {"clip_logit":1., "candidate_time_sec":2.}
        assert not list(private_root.iterdir())
    finally:
        runner.close()


@pytest.mark.parametrize("mode", ["filecap", "memory"])
def test_resource_limit_stops_worker_and_cleans(private_root, tmp_path, mode):
    marker = tmp_path / "started"
    runner = ResearchRunner(Probe(mode, str(marker)), timeout=3, memory_mb=64, work_root=private_root)
    started = time.monotonic()
    try:
        with pytest.raises(JobFailure) as error:
            runner.run("synthetic")
        assert not isinstance(error.value, JobTimeout), "resource cap should act before wall timeout"
        assert time.monotonic() - started < 3
        record = json.loads(marker.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(record["pid"], 0)
        assert not list(private_root.iterdir())
    finally:
        runner.close()


def test_symlink_output_rejected_without_modifying_target(private_root, tmp_path):
    marker = tmp_path / "synthetic-target"
    runner = ResearchRunner(Probe("symlink", str(marker)), timeout=3, memory_mb=64,
                            work_root=private_root)
    try:
        with pytest.raises(JobFailure):
            runner.run("synthetic")
        assert set(json.loads(marker.read_text())) == {"pid", "cwd"}
        assert not list(private_root.iterdir())
    finally:
        runner.close()


def test_untrusted_work_root_permissions_and_symlink_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    with pytest.raises(ValueError):
        ResearchRunner(Probe(), work_root=root)
    root.chmod(0o700)
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        ResearchRunner(Probe(), work_root=link)


@pytest.mark.parametrize("payload", [{"sample_id":"../private"},
    {"sample_id":"synthetic", "path":"/private/fixture"}])
def test_paths_rejected_without_provider_or_echo(private_root, tmp_path, payload):
    marker = tmp_path / "called"
    with TestClient(create_research_app(Probe(marker=str(marker)), api_key=KEY,
                                       work_root=private_root)) as client:
        response = client.post(ROUTE, headers={"X-API-Key":KEY}, json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail":"invalid research request"}
    assert not marker.exists()
    assert not list(private_root.iterdir())


def test_body_limit_prevents_provider(private_root, tmp_path):
    marker = tmp_path / "called"
    with TestClient(create_research_app(Probe(marker=str(marker)), api_key=KEY,
                                       work_root=private_root)) as client:
        response = client.post(ROUTE, headers={"X-API-Key":KEY, "Content-Type":"application/json"},
                               content=b"x" * 1025)
    assert response.status_code == 413
    assert not marker.exists()
    assert not list(private_root.iterdir())


def test_concurrency_rejected_and_close_kills_active_job(private_root, tmp_path):
    marker = tmp_path / "started"
    runner = ResearchRunner(Probe("sleep", str(marker)), timeout=5, memory_mb=64,
                            work_root=private_root)
    failures = []
    def run():
        try:
            runner.run("synthetic")
        except JobFailure as error:
            failures.append(error)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.exists()
        with pytest.raises(JobBusy):
            runner.run("second")
        runner.close()
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert len(failures) == 1
        assert not list(private_root.iterdir())
        with pytest.raises(JobFailure):
            runner.run("after-close")
        assert not list(private_root.iterdir())
    finally:
        runner.close()
        thread.join(timeout=3)
