"""F1/F7 strict negative regressions + failure semantics + done schema (JobStore, in-process pipeline)."""
import base64
import os

import pytest

from app.jobs import JobStore
from helpers import (ALLOWED_CODES, EmptyDetector, TwoPersonDetector, clip, inproc, jpeg_b64, leftovers, rules_file,
                     settings, wait_store)

UNATT = "operator_zone_unattended_with_counter_activity"
SECOND = "second_person_in_operator_zone"


def run(tmp_path, path, analyze_fn, **kw):
    s = settings(tmp_path, **kw)
    store = JobStore(s, analyze_fn)
    try:
        jid = store.submit(_copy(path, tmp_path))
        return wait_store(store, jid, 120)
    finally:
        store.close()


def _copy(path, tmp_path):
    import shutil
    d = tmp_path / "up"; d.mkdir(exist_ok=True)
    dst = str(d / (os.urandom(16).hex() + ".upload"))
    shutil.copy(path, dst)
    return dst


def assert_failed(j, code):
    assert j["status"] == "failed", j
    assert j["error"] == code, j
    assert "verdict" not in j and "violations" not in j


# ---------------------------------------------------------------- 1a/1b default rules
def test_default_rules_zero_persons_with_counter_motion_is_failed(tmp_path):
    v = clip(tmp_path, absent=(0, 40))  # no operator, banknote motion on counter
    j = run(tmp_path, v, inproc(EmptyDetector()))
    assert_failed(j, "no_validated_rule")  # F1/F7: never no_violation / violation


def test_default_rules_two_persons_is_failed_not_violation(tmp_path):
    v = clip(tmp_path, two=(5, 35))
    j = run(tmp_path, v, inproc())
    assert j.get("verdict") != "violation"
    assert_failed(j, "no_validated_rule")


def test_default_ruleset_has_all_rules_disabled_candidate():
    import json
    from app import pipeline
    r = json.load(open(pipeline.DEFAULT_RULESET))["rules"]
    for k, v in r.items():
        assert v["enabled"] is False and v["status"] == "candidate" and not v["validated"], k


def test_empty_detector_with_optin_is_failed_not_no_violation(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_UNVALIDATED_RULES", "1")
    v = clip(tmp_path, absent=(0, 40))
    j = run(tmp_path, v, inproc(EmptyDetector()), ruleset=rules_file(tmp_path))
    assert j["status"] == "failed" and "verdict" not in j
    assert j["error"] in ALLOWED_CODES and j["error"] != "no_validated_rule"


# ---------------------------------------------------------------- 1c opt-in marker, second person never violates
def _analyze(path, s, det=None):
    from app import pipeline
    return pipeline.analyze(path, s, detector=det)


def test_optin_second_person_rule_cannot_produce_violation(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_UNVALIDATED_RULES", "1")
    rf = rules_file(tmp_path, enable=(UNATT, SECOND))
    res = _analyze(clip(tmp_path), settings(tmp_path, ruleset=rf), TwoPersonDetector())
    obs = [o for o in res["observations"] if o["type"] == SECOND]
    assert obs, "precondition: the second-person observation must be produced (else this test is vacuous)"
    assert all(o["drives_verdict"] is False for o in obs)
    assert all(e["type"] != SECOND for e in res["violations"])
    assert SECOND not in res["meta"]["active_rules"]
    for e in res["violations"]:
        assert "п.42" not in e["description"] and "п.75" not in e["description"]


def test_mutation_guard_removal_would_yield_second_person_violation(tmp_path, monkeypatch):
    """Proves the test above is effective: without the guard (NEVER_VERDICT empty and no
    observation_only flag) the same input DOES produce a SECOND violation."""
    from app import pipeline
    monkeypatch.setenv("ALLOW_UNVALIDATED_RULES", "1")
    monkeypatch.setattr(pipeline, "NEVER_VERDICT", set())

    def mut(r):
        r["rules"][SECOND].pop("observation_only", None)
    rf = rules_file(tmp_path, mut, enable=(UNATT, SECOND))
    res = _analyze(clip(tmp_path), settings(tmp_path, ruleset=rf), TwoPersonDetector())
    assert any(e["type"] == SECOND for e in res["violations"]), \
        "mutated pipeline should emit a SECOND violation; the guard is what prevents it"


def test_zero_person_pattern_is_actually_observed_before_default_failure(tmp_path, monkeypatch):
    v = clip(tmp_path, absent=(12, 34))
    with monkeypatch.context() as m:  # precondition run: pattern really present in this clip
        m.setenv("ALLOW_UNVALIDATED_RULES", "1")
        res = _analyze(v, settings(tmp_path, ruleset=rules_file(tmp_path)))
        assert any(o["type"] == UNATT for o in res["observations"]), "clip does not exercise the pattern"
    j = run(tmp_path, v, inproc())  # default rules: same clip must fail, not no_violation/violation
    assert_failed(j, "no_validated_rule")


def test_optin_violation_is_marked_unvalidated(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_UNVALIDATED_RULES", "1")
    v = clip(tmp_path, absent=(12, 34))
    j = run(tmp_path, v, inproc(), ruleset=rules_file(tmp_path))
    assert j["status"] == "done", j
    assert j["verdict"] == "violation", "synthetic absence window should yield a (demo, unvalidated) violation"
    for e in j["violations"]:
        d = e["description"].lower()
        assert "не валидировано" in d or "unvalidated" in d, e["description"]
        assert e["type"] != SECOND


def test_without_optin_enabled_candidate_rule_is_still_failed(tmp_path):
    v = clip(tmp_path, absent=(12, 34))
    j = run(tmp_path, v, inproc(), ruleset=rules_file(tmp_path))
    assert_failed(j, "no_validated_rule")


# ---------------------------------------------------------------- 1d validated without measured metrics
@pytest.mark.parametrize("holdout", [None, {}, {"value": None}, {"value": "0.9"}, {"value": True}])
def test_validated_flag_without_measured_metric_refused(tmp_path, holdout):
    def mut(r):
        r["rules"][UNATT].update(status="validated", validated=True, holdout=holdout)
    v = clip(tmp_path, absent=(12, 34))
    j = run(tmp_path, v, inproc(), ruleset=rules_file(tmp_path, mut))
    assert_failed(j, "invalid_config")  # claims validation with incomplete/invalid record


# ---------------------------------------------------------------- 2 failure semantics
def test_corrupt_video(tmp_path):
    p = tmp_path / "c.bin"; p.write_bytes(b"not a video" * 500)
    assert_failed(run(tmp_path, str(p), inproc()), "corrupt_video")


def test_too_long(tmp_path):
    assert_failed(run(tmp_path, clip(tmp_path, dur=12), inproc(), max_duration_sec=5), "too_long")


def test_model_missing_even_with_synthetic_detector(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_UNVALIDATED_RULES", "1")
    j = run(tmp_path, clip(tmp_path), inproc(), onnx_model=str(tmp_path / "nope.onnx"),
            ruleset=rules_file(tmp_path))
    assert_failed(j, "model_missing")


def test_model_missing_subprocess_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_UNVALIDATED_RULES", "1")
    s = settings(tmp_path, onnx_model=str(tmp_path / "nope.onnx"), ruleset=rules_file(tmp_path))
    store = JobStore(s, None)
    try:
        jid = store.submit(_copy(clip(tmp_path), tmp_path))
        assert_failed(wait_store(store, jid, 60), "model_missing")
    finally:
        store.close()


def test_analyzer_crash_has_allowed_code_no_verdict(tmp_path):
    def boom(path, s, deadline=None):
        raise RuntimeError("secret path /Users/x/y.mov")
    j = run(tmp_path, clip(tmp_path, dur=2), boom)
    assert j["status"] == "failed" and "verdict" not in j
    assert j["error"] in ALLOWED_CODES and "secret" not in str(j) and "/Users" not in str(j)


def test_result_never_has_third_verdict(tmp_path):
    def odd(path, s, deadline=None):
        return {"verdict": "unknown", "violations": []}
    j = run(tmp_path, clip(tmp_path, dur=2), odd)
    assert j["status"] == "failed" and "verdict" not in j


# ---------------------------------------------------------------- 5 schema of done
def _check_event(e):
    assert {"type", "timecode_sec", "description", "evidence_frames"} <= e.keys()
    assert 1 <= len(e["evidence_frames"]) <= 3
    for fr in e["evidence_frames"]:
        assert base64.b64decode(fr["image_base64"])[:2] == b"\xff\xd8"
        assert fr["timestamp_sec"] >= 0
    assert "НБРК" in e["description"]
    d = e["description"].lower()
    assert "не валидировано" in d or "unvalidated" in d


def test_schema_done_operator_present_throughout(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_UNVALIDATED_RULES", "1")
    j = run(tmp_path, clip(tmp_path), inproc(), ruleset=rules_file(tmp_path))
    assert j["status"] == "done", j
    assert j["verdict"] in ("violation", "no_violation")
    assert (j["verdict"] == "violation") == bool(j["violations"])
    for e in j["violations"]:
        _check_event(e)


def test_schema_done_violation_events(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_UNVALIDATED_RULES", "1")
    j = run(tmp_path, clip(tmp_path, absent=(12, 34)), inproc(), ruleset=rules_file(tmp_path))
    assert j["status"] == "done" and j["verdict"] == "violation", j
    for e in j["violations"]:
        _check_event(e)


def test_no_leftover_files_after_jobs(tmp_path):
    run(tmp_path, clip(tmp_path, dur=2), inproc())
    assert leftovers(str(tmp_path / "data")) == []
