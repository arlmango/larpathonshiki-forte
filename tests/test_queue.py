"""Deadline / queue / admission / cleanup."""
import os
import threading
import time

from fastapi.testclient import TestClient

from app.jobs import JobStore
from app.main import create_app
from helpers import KEY, clip, inproc, leftovers, post, settings, wait_http, wait_store


def _up(tmp_path, data=b"x"):
    d = tmp_path / "up"; d.mkdir(exist_ok=True)
    p = d / (os.urandom(16).hex() + ".upload"); p.write_bytes(data)
    return str(p)


def test_inprocess_sleep_hits_budget(tmp_path):
    gate = threading.Event()

    def slow(path, s, deadline=None):
        gate.wait(10)
        return {"verdict": "no_violation", "violations": []}
    s = settings(tmp_path, job_budget_sec=3, job_estimate_sec=1, kill_grace_sec=0.5, orphan_max_age_min=1)
    store = JobStore(s, slow)
    try:
        t = time.time()
        jid = store.submit(_up(tmp_path))
        j = wait_store(store, jid, 12)
        assert time.time() - t <= 3 + 3
        assert j["status"] == "failed" and j["error"] == "time_budget_exceeded" and "verdict" not in j
    finally:
        gate.set(); store.close()


def test_hung_child_fifo_killed_within_budget(tmp_path):
    fifo = str(tmp_path / (os.urandom(16).hex() + ".upload"))
    os.mkfifo(fifo)  # opening it for read blocks forever -> hung decoder stand-in
    s = settings(tmp_path, job_budget_sec=3, job_estimate_sec=1, kill_grace_sec=0.5, orphan_max_age_min=1)
    store = JobStore(s, None)
    try:
        t = time.time()
        jid = store.submit(fifo)
        j = wait_store(store, jid, 12)
        assert time.time() - t <= 3 + 3
        assert j["status"] == "failed" and j["error"] == "time_budget_exceeded"
    finally:
        store.close()


def test_queue_full_429_and_recovery(tmp_path):
    gate = threading.Event()

    def slow(path, s, deadline=None):
        gate.wait(15)
        return {"verdict": "no_violation", "violations": []}
    c = TestClient(create_app(settings(tmp_path, workers=1, queue_limit=1), slow))
    codes = [post(c, b"x").status_code for _ in range(5)]
    gate.set()
    assert 429 in codes and codes[0] == 200
    t = time.time()
    while time.time() - t < 10 and leftovers(str(tmp_path / "data")):
        time.sleep(0.2)
    assert leftovers(str(tmp_path / "data")) == []


def test_parallel_jobs_all_final_no_leftovers(tmp_path):
    v = open(clip(tmp_path, dur=3), "rb").read()
    d = tmp_path / "data"
    c = TestClient(create_app(settings(tmp_path, workers=2, queue_limit=10, job_estimate_sec=1), inproc()))
    ids = []
    for i in range(6):
        r = post(c, v)
        assert r.status_code == 200, r.text
        ids.append(r.json()["job_id"])
    ids.append(post(c, b"garbage" * 100).json()["job_id"])
    out = [wait_http(c, i, 90) for i in ids]
    assert all(j["status"] in ("done", "failed") for j in out)
    assert not any(j["status"] == "processing" for j in out)
    assert all("verdict" not in j for j in out if j["status"] == "failed")
    time.sleep(0.5)
    assert leftovers(str(d)) == []
