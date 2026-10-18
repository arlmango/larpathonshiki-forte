"""H2/H3/M1/M5/M6 regressions for ruleset integrity (app/detector_rules.py). Synthetic only."""
import math

import pytest

from app import pipeline
from helpers import EmptyDetector, clip, inproc, rules_file, settings
from test_pipeline_rules import UNATT, SECOND, assert_failed, run

GOOD = {"n_clips": 5, "precision": 0.5, "recall": 0.5, "evaluated_at": "2026-10-17",
        "split_id": "x", "tool": "tools/evaluate.py"}


def _claim(h, **claim):
    def mut(r):
        r["rules"][UNATT].update(status="validated", validated=True, holdout=h, **claim)
    return mut


def _bad(tmp_path, mut, det=None, **kw):
    return run(tmp_path, clip(tmp_path, absent=(12, 34)), inproc(det), ruleset=rules_file(tmp_path, mut), **kw)


# ------------------------------------------------------------------ H2
@pytest.mark.parametrize("h", [
    {"value": 0}, {"value": -1}, {"value": float("nan")}, {"value": float("inf")}, {"value": 1e9},
    {"value": 0.99, "n_clips": 0},
    {**GOOD, "n_clips": 4}, {**GOOD, "precision": 1.5}, {**GOOD, "recall": float("nan")},
    {**GOOD, "tool": "other.py"}, {**GOOD, "evaluated_at": "yesterday"}, {**GOOD, "split_id": ""},
], ids=lambda h: str(h)[:40])
def test_h2_invalid_holdout_record_is_invalid_config(tmp_path, h):
    assert_failed(_bad(tmp_path, _claim(h)), "invalid_config")


def test_h2_status_validated_but_flag_false(tmp_path):
    def mut(r):
        r["rules"][UNATT].update(status="validated", validated=False, holdout=GOOD)
    assert_failed(_bad(tmp_path, mut), "invalid_config")


def test_h2_complete_record_is_validated_without_marker(tmp_path):
    j = _bad(tmp_path, _claim(GOOD))
    assert j["status"] == "done", j
    assert j["verdict"] == "violation"
    for e in j["violations"]:
        d = e["description"].lower()
        assert "не валидировано" not in d and "unvalidated" not in d


# ------------------------------------------------------------------ H3
def _top(**kv):
    return lambda r: r.update(kv)


@pytest.mark.parametrize("mut", [
    _top(min_person_sample_ratio=0), _top(min_person_sample_ratio=-1), _top(min_person_sample_ratio=0.05),
    lambda r: r["zone"].update(min_occupancy_to_analyse=0),
    lambda r: r["zone"].update(min_occupancy_to_analyse=0, override=[0.1, 0.1, 0.9, 0.9]),
    lambda r: r["zone"].update(min_occupancy_to_analyse=-1),
    _top(min_confidence_to_emit=2), _top(max_events=0),
    lambda r: r["rules"][UNATT].update(min_confidence_to_emit=2),
], ids=range(9))
def test_h3_gate_tampering_is_invalid_config(tmp_path, monkeypatch, mut):
    monkeypatch.setenv("ALLOW_UNVALIDATED_RULES", "1")
    assert_failed(_bad(tmp_path, mut), "invalid_config")


def test_h3_empty_detector_with_gates_intact_is_too_few_detections(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_UNVALIDATED_RULES", "1")
    assert_failed(_bad(tmp_path, None, EmptyDetector()), "too_few_detections")


# ------------------------------------------------------------------ M1
@pytest.mark.parametrize("fake_holdout", [False, True])
def test_m1_unknown_rule_enabled_is_invalid_config(tmp_path, monkeypatch, fake_holdout):
    monkeypatch.setenv("ALLOW_UNVALIDATED_RULES", "1")

    def mut(r):
        z = {"enabled": True, "status": "candidate", "validated": False, "holdout": None}
        if fake_holdout:
            z.update(status="validated", validated=True, holdout=GOOD)
        r["rules"]["zz"] = z
    assert_failed(_bad(tmp_path, mut), "invalid_config")


def test_m1_missing_known_rule_is_invalid_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_UNVALIDATED_RULES", "1")
    assert_failed(_bad(tmp_path, lambda r: r["rules"].pop(SECOND)), "invalid_config")


# ------------------------------------------------------------------ M6
def test_m6_describe_model_leaks_no_paths(tmp_path):
    s = settings(tmp_path, ruleset=str(tmp_path / "SECRET_rules.json"),
                 onnx_model=str(tmp_path / "SECRET_w.onnx"))
    d = pipeline.describe_model(s)
    assert "SECRET" not in repr(d)
    assert d.get("rules_error") == "ruleset_unreadable"
    assert d.get("error") in (None, "model_missing") or "SECRET" not in str(d.get("error"))
    assert d["rules"] == []


def test_m6_missing_weights_code_only(tmp_path):
    d = pipeline.describe_model(settings(tmp_path, onnx_model=str(tmp_path / "SECRET_w.onnx")))
    assert "SECRET" not in repr(d)
    assert "model_missing" in repr(d)


# ------------------------------------------------------------------ M5
def test_m5_stderr_does_not_contain_filename(tmp_path, capfd):
    p = tmp_path / "SENTINEL_name.mp4"
    p.write_bytes(bytes(range(256)) * 400)
    with pytest.raises(pipeline.AnalysisError):
        pipeline.analyze(str(p), settings(tmp_path))
    cap = capfd.readouterr()
    assert "SENTINEL" not in cap.err and "SENTINEL" not in cap.out
