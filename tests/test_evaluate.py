"""tools/evaluate.py self-test on synthetic labels/predictions only."""
import json
import random

import pytest

from tools import evaluate as ev


def sha(i):
    return f"{i:016x}"


def write(tmp_path, clips, split, preds):
    """clips: {i: (status|None, [event times])}; split: {i: side}; preds: {i: (verdict, [times]) | 'failed'}"""
    lab = tmp_path / "labels.jsonl"
    with open(lab, "w") as f:
        for i, (st, evs) in clips.items():
            if st:
                f.write(json.dumps({"type": "clip_status", "kind": "clip_status", "clip": i, "sha16": sha(i),
                                    "sha": sha(i), "status": st}) + "\n")
            for t in evs:
                f.write(json.dumps({"kind": "event", "clip": i, "sha": sha(i), "t": t, "note": "SECRET NOTE", "norm": None}) + "\n")
            f.write(json.dumps({"kind": "obs_review", "clip": i, "sha": sha(i), "obs_index": 0, "type": "x",
                                "start_sec": 0, "end_sec": 1, "decision": "accept"}) + "\n")
    sp = tmp_path / "split.json"
    sp.write_text(json.dumps({"clips": {sha(i): s for i, s in split.items()}}))
    pd = tmp_path / "preds"; pd.mkdir(exist_ok=True)
    for i, p in preds.items():
        if p == "failed":
            body = {"job_id": "0" * 32, "status": "failed", "error": "no_validated_rule"}
        else:
            body = {"job_id": "0" * 32, "status": "done", "verdict": p[0],
                    "violations": [{"type": "t", "timecode_sec": t} for t in p[1]]}
        (pd / (sha(i) + ".json")).write_text(json.dumps(body))
    return str(lab), str(sp), str(pd)


def base(tmp_path, n=6):
    """n holdout clips: 3 violation, 3 clean; predictions perfect except where tests tweak."""
    clips, split, preds = {}, {}, {}
    for i in range(n):
        pos = i % 2 == 0
        clips[i] = ("reviewed_violation", [10.0 + i]) if pos else ("reviewed_no_violation", [])
        split[i] = "holdout"
        preds[i] = ("violation", [12.0 + i]) if pos else ("no_violation", [])
    return clips, split, preds


def test_perfect_and_mae(tmp_path):
    r = ev.evaluate(*write(tmp_path, *base(tmp_path)), tolerance_sec=5, min_holdout=5)
    assert r["clip_level"]["f1"] == 1.0 and r["clip_level"]["tn"] == 3
    assert r["event_level"]["tp"] == 3 and r["event_level"]["fp"] == 0 and r["event_level"]["fn"] == 0
    assert r["event_level"]["timestamp_mae_sec"] == 2.0


def test_tolerance_and_fp_fn(tmp_path):
    clips, split, preds = base(tmp_path)
    preds[0] = ("violation", [40.0])        # outside tolerance: FP event + FN event, clip still TP
    preds[1] = ("violation", [3.0])         # clip FP
    r = ev.evaluate(*write(tmp_path, clips, split, preds), tolerance_sec=5, min_holdout=5)
    assert r["clip_level"]["tp"] == 3 and r["clip_level"]["fp"] == 1
    assert r["event_level"]["fp"] == 2 and r["event_level"]["fn"] == 1 and r["event_level"]["tp"] == 2


def test_unreviewed_excluded_from_denominators(tmp_path):
    clips, split, preds = base(tmp_path, 6)
    for i in (6, 7, 8):   # unreviewed: no status record, with and without events, bad predictions
        clips[i] = (None, [5.0] if i == 6 else [])
        split[i] = "holdout"; preds[i] = ("violation", [1.0])
    r = ev.evaluate(*write(tmp_path, clips, split, preds), min_holdout=5)
    assert r["holdout_unreviewed_excluded"] == 3 and r["holdout_reviewed"] == 6
    assert r["clip_level"]["fp"] == 0 and r["clip_level"]["tn"] == 3 and r["event_level"]["fp"] == 0


def test_explicit_unreviewed_status_excluded(tmp_path):
    clips, split, preds = base(tmp_path, 6)
    clips[6] = ("unreviewed", []); split[6] = "holdout"; preds[6] = ("no_violation", [])
    r = ev.evaluate(*write(tmp_path, clips, split, preds), min_holdout=5)
    assert r["holdout_unreviewed_excluded"] == 1 and r["clip_level"]["tn"] == 3


def test_empty_events_on_unreviewed_is_not_tn(tmp_path):
    clips, split, preds = base(tmp_path, 6)
    clips[6] = (None, []); split[6] = "holdout"; preds[6] = ("no_violation", [])
    r = ev.evaluate(*write(tmp_path, clips, split, preds), min_holdout=5)
    assert r["clip_level"]["tn"] == 3  # not 4


def test_obs_review_accept_is_not_ground_truth(tmp_path):
    clips, split, preds = base(tmp_path, 6)
    clips[6] = (None, []); split[6] = "holdout"; preds[6] = ("violation", [1.0])  # has obs_review accept, no status
    r = ev.evaluate(*write(tmp_path, clips, split, preds), min_holdout=5)
    assert r["holdout_reviewed"] == 6 and r["clip_level"]["fp"] == 0


def test_train_clips_never_scored(tmp_path):
    clips, split, preds = base(tmp_path, 6)
    for i in (10, 11):
        clips[i] = ("reviewed_violation", [1.0]); split[i] = "train"; preds[i] = ("no_violation", [])
    r = ev.evaluate(*write(tmp_path, clips, split, preds), min_holdout=5)
    assert r["train_clips"] == 2 and r["clip_level"]["fn"] == 0


def test_min_holdout_refusal(tmp_path):
    clips, split, preds = base(tmp_path, 4)
    with pytest.raises(ev.EvalRefused, match="minimum 5"):
        ev.evaluate(*write(tmp_path, clips, split, preds))
    # unreviewed do not count toward the minimum
    clips, split, preds = base(tmp_path, 4)
    for i in (4, 5, 6):
        clips[i] = (None, []); split[i] = "holdout"
    with pytest.raises(ev.EvalRefused):
        ev.evaluate(*write(tmp_path, clips, split, preds))


def test_leakage_detected_list_format(tmp_path):
    lab, _, pd = write(tmp_path, *base(tmp_path))
    sp = tmp_path / "leak.json"
    sp.write_text(json.dumps({"train": [sha(0), sha(99)], "holdout": [sha(i) for i in range(6)]}))
    with pytest.raises(ev.EvalRefused, match="LEAKAGE"):
        ev.evaluate(lab, str(sp), pd)


def test_leakage_helper():
    with pytest.raises(ev.EvalRefused, match="LEAKAGE"):
        ev.assert_no_leakage({"a": "holdout"}, train_shas=["a"])


def test_failed_and_missing_predictions_counted_separately(tmp_path):
    clips, split, preds = base(tmp_path, 6)
    preds[0] = "failed"; del preds[2]
    r = ev.evaluate(*write(tmp_path, clips, split, preds), min_holdout=5)
    assert r["pipeline_failed"] == 1 and r["prediction_missing"] == 1
    assert r["clip_level"]["tp"] == 1 and r["clip_level"]["recall_counting_failed_as_miss"] == 0.3333


def test_inconsistent_labels_excluded(tmp_path):
    clips, split, preds = base(tmp_path, 6)
    clips[1] = ("reviewed_no_violation", [3.0])
    r = ev.evaluate(*write(tmp_path, clips, split, preds), min_holdout=5)
    assert r["excluded_inconsistent_labels"] == 1


def test_cli_prints_aggregates_only(tmp_path, capsys):
    clips, split, preds = base(tmp_path)
    lab, sp, pd = write(tmp_path, clips, split, preds)
    assert ev.main(["--labels", lab, "--split", sp, "--predictions", pd]) == 0
    out = capsys.readouterr().out
    json.loads(out)
    for bad in ("SECRET", str(tmp_path), sha(0), "labels.jsonl"):
        assert bad not in out


def test_cli_refusal_exit_code_and_no_paths(tmp_path, capsys):
    lab, sp, pd = write(tmp_path, *base(tmp_path, 3))
    assert ev.main(["--labels", lab, "--split", sp, "--predictions", pd]) == 2
    out = capsys.readouterr().out
    assert "REFUSED" in out and str(tmp_path) not in out
    assert ev.main(["--labels", str(tmp_path / "nope"), "--split", sp, "--predictions", pd]) == 2
    assert str(tmp_path) not in capsys.readouterr().out


def test_random_property_perfect_predictions_give_f1_one(tmp_path):
    rnd = random.Random(1)
    clips, split, preds = {}, {}, {}
    for i in range(30):
        pos = rnd.random() < 0.5
        t = rnd.uniform(0, 100)
        clips[i] = ("reviewed_violation", [t]) if pos else ("reviewed_no_violation", [])
        split[i] = "holdout"
        preds[i] = ("violation", [t + rnd.uniform(-4, 4)]) if pos else ("no_violation", [])
    r = ev.evaluate(*write(tmp_path, clips, split, preds))
    assert r["clip_level"]["f1"] == 1.0 and r["event_level"]["f1"] == 1.0


# ------------------------------------------------------------------ --source imported_dataset
def write_imported(tmp_path, clips, split, preds, human=()):
    """clips: list of (i, status, [times]) records (repeat i to simulate duplicate clips)."""
    lab = tmp_path / "C01.imported.jsonl"
    with open(lab, "w") as f:
        for k, (i, st, evs) in enumerate(clips):
            f.write(json.dumps({"kind": "imported_clip_status", "clip": k, "sha": sha(i), "sha16": sha(i),
                                "status": st}) + "\n")
            for t in evs:
                f.write(json.dumps({"kind": "imported_event", "clip": k, "sha": sha(i), "timestamp_sec": t,
                                    "status": "imported_unconfirmed"}) + "\n")
        for i, st in human:  # human records must be ignored in this mode
            f.write(json.dumps({"kind": "clip_status", "sha": sha(i), "status": st}) + "\n")
    _, sp, pd = write(tmp_path, {}, split, preds)
    return str(lab), sp, pd


def imp_base(n=6):
    clips = [(i, "reviewed_violation", [10.0 + i]) if i % 2 == 0 else (i, "reviewed_no_violation", [])
             for i in range(n)]
    split = {i: "holdout" for i in range(n)}
    preds = {i: ("violation", [12.0 + i]) if i % 2 == 0 else ("no_violation", []) for i in range(n)}
    return clips, split, preds


def test_imported_basic_and_human_records_ignored(tmp_path):
    clips, split, preds = imp_base()
    lab, sp, pd = write_imported(tmp_path, clips, split, preds, human=[(1, "reviewed_violation")])
    r = ev.evaluate(lab, sp, pd, source="imported_dataset")
    assert r["source"] == "imported_dataset" and r["clip_level"]["f1"] == 1.0 and r["clip_level"]["tn"] == 3
    # same file in human mode sees no imported records -> everything unreviewed -> refusal
    with pytest.raises(ev.EvalRefused):
        ev.evaluate(lab, sp, pd, source="human")


def test_imported_unknown_excluded_and_counted(tmp_path):
    clips, split, preds = imp_base()
    for i in (6, 7):
        clips.append((i, "unknown", [])); split[i] = "holdout"; preds[i] = ("violation", [1.0])
    r = ev.evaluate(*write_imported(tmp_path, clips, split, preds), source="imported_dataset")
    assert r["imported_unknown_excluded"] == 2 and r["holdout_reviewed"] == 6 and r["clip_level"]["fp"] == 0


def test_imported_duplicates_counted_once(tmp_path):
    clips, split, preds = imp_base(5)
    clips += [clips[0], clips[1]]  # byte-identical pairs: same sha16 imported twice
    r = ev.evaluate(*write_imported(tmp_path, clips, split, preds), source="imported_dataset")
    assert r["holdout_reviewed"] == 5 and r["clip_level"]["tp"] == 3 and r["clip_level"]["tn"] == 2
    assert r["event_level"]["tp"] == 3 and r["event_level"]["fn"] == 0


def test_imported_conflicting_duplicates_excluded(tmp_path):
    clips, split, preds = imp_base(6)
    clips.append((0, "reviewed_no_violation", []))
    r = ev.evaluate(*write_imported(tmp_path, clips, split, preds), source="imported_dataset", min_holdout=5)
    assert r["excluded_inconsistent_labels"] == 1


def test_imported_min_holdout_distinct_refusal(tmp_path):
    clips, split, preds = imp_base(4)
    clips += clips  # 8 records, 4 distinct clips
    with pytest.raises(ev.EvalRefused, match="minimum 5"):
        ev.evaluate(*write_imported(tmp_path, clips, split, preds), source="imported_dataset")


def test_imported_cli_provenance_header_and_defaults(tmp_path, capsys):
    clips, split, preds = imp_base()
    lab, sp, pd = write_imported(tmp_path, clips, split, preds)
    priv = tmp_path / "priv"
    (priv / "labels").mkdir(parents=True); (priv / "splits").mkdir()
    import os, shutil
    shutil.copy(lab, priv / "labels" / "C01.imported.jsonl")
    shutil.copy(sp, priv / "splits" / "C01.json")
    shutil.copytree(pd, priv / "predictions")
    before = (priv / "labels" / "C01.imported.jsonl").read_bytes()
    assert ev.main(["--source", "imported_dataset", "--cid", "C01", "--private", str(priv)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "source: imported_dataset — по поставленной XLSX, без ручной проверки"
    json.loads(out.split("\n", 1)[1])
    assert str(tmp_path) not in out and sha(0) not in out
    assert (priv / "labels" / "C01.imported.jsonl").read_bytes() == before  # never modified
    assert not os.path.exists(priv / "labels" / "C01.jsonl")


def test_prf_f1_zero_when_p_and_r_zero():
    assert ev._prf(0, 10, 5) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_prf_all_none_when_no_counts():
    assert ev._prf(0, 0, 0) == {"precision": None, "recall": None, "f1": None}


def test_prf_f1_unchanged_nonzero():
    r = ev._prf(3, 7, 2)
    assert r["precision"] == 0.3 and r["recall"] == 0.6 and r["f1"] == 0.4
