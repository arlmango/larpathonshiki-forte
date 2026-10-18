"""Whole-clip holdout evaluation. Prints AGGREGATES ONLY (counts, P/R/F1, timestamp error).
Never prints clip names, paths, shas or note text. It never enables or validates a rule: it
only prints numbers for a human to read.

usage: python -m tools.evaluate --labels L.jsonl --split S.json --predictions DIR
                                [--tolerance-sec 5] [--min-holdout 5]

labels (tools/annotate.py): one JSON per line. Accepted record shapes
  {"kind":"event","sha":<16hex>,"t":<sec>,...}                (human event label; also sha16/timestamp_sec)
  {"kind":"clip_status","sha":<16hex>,"status":unreviewed|reviewed_no_violation|reviewed_violation}
  --source imported_dataset: reads ONLY imported_clip_status (reviewed_*|unknown; unknown excluded) and
  imported_event (timestamp_sec) records; human records ignored; clips deduplicated by sha16.
  {"kind":"obs_review",...}   ignored: accept/reject of pipeline observations is NOT ground truth
A clip without a clip_status record is `unreviewed`; an empty event list never means "negative".
split: {"clips": {sha: "train"|"holdout"}} or {"train":[sha..],"holdout":[sha..]}.
predictions: DIR/<sha16>.json with the API job body {status, verdict, violations:[{timecode_sec}]}.
"""
import argparse
import json
import os
import re
import sys

SHA_RE = re.compile(r"^[0-9a-f]{16}$")
STATUSES = ("unreviewed", "reviewed_no_violation", "reviewed_violation")


class EvalRefused(Exception):
    """Evaluation refused. Message is static/aggregate-only (no names or paths)."""


def _sha(r):
    s = r.get("sha") if r.get("sha") is not None else r.get("sha16")
    return s if isinstance(s, str) and SHA_RE.match(s) else None


def load_labels(path):
    """-> (events: {sha: [t..]}, status: {sha: str}); malformed lines are counted, not shown."""
    events, status, bad = {}, {}, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if not isinstance(r, dict):
                bad += 1
                continue
            kind, sha = r.get("kind") or r.get("type"), _sha(r)
            if sha is None:
                bad += 1
                continue
            if kind == "event":
                t = r.get("t", r.get("timestamp_sec"))
                if isinstance(t, (int, float)) and not isinstance(t, bool) and t >= 0:
                    events.setdefault(sha, []).append(float(t))
                else:
                    bad += 1
            elif kind == "clip_status":
                if r.get("status") in STATUSES:
                    status[sha] = r["status"]  # last record wins
                else:
                    bad += 1
    return events, status, bad


def load_imported(path):
    """imported_dataset labels: imported_clip_status / imported_event only; human records are ignored.
    Clips are deduplicated by sha16 (byte-identical clips count once). -> (events, status, bad, dup)"""
    events, seen, bad, dup = {}, {}, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if not isinstance(r, dict):
                bad += 1
                continue
            kind = r.get("kind")
            if kind not in ("imported_clip_status", "imported_event"):
                continue
            sha = _sha(r)
            if sha is None:
                bad += 1
                continue
            if kind == "imported_clip_status":
                st = r.get("status")
                if st in STATUSES[1:] or st == "unknown":
                    seen.setdefault(sha, set()).add(st)
                else:
                    bad += 1
            else:
                t = r.get("timestamp_sec")
                if isinstance(t, (int, float)) and not isinstance(t, bool) and t >= 0:
                    events.setdefault(sha, set()).add(round(float(t), 3))
                else:
                    bad += 1
    status = {}
    for sha, st in seen.items():
        real = st - {"unknown"}
        status[sha] = "conflict" if len(real) > 1 else (next(iter(real)) if real else "unknown")
    return {k: sorted(v) for k, v in events.items()}, status, bad


def load_split(path):
    with open(path, encoding="utf-8") as f:
        sp = json.load(f)
    if not isinstance(sp, dict):
        raise EvalRefused("split file malformed")
    side = {}
    if isinstance(sp.get("clips"), dict):
        side = {k: v for k, v in sp["clips"].items() if v in ("train", "holdout")}
    elif isinstance(sp.get("train"), list) and isinstance(sp.get("holdout"), list):
        tr, ho = set(sp["train"]), set(sp["holdout"])
        if tr & ho:
            raise EvalRefused(f"LEAKAGE: {len(tr & ho)} clip(s) are in both train and holdout")
        side = {**{k: "train" for k in tr}, **{k: "holdout" for k in ho}}
    else:
        raise EvalRefused("split file malformed")
    return side


def assert_no_leakage(side, events=None, train_shas=()):
    """A sha must have exactly one side; extra train shas must not appear in holdout."""
    overlap = {s for s in train_shas if side.get(s) == "holdout"}
    if overlap:
        raise EvalRefused(f"LEAKAGE: {len(overlap)} clip(s) are in both train and holdout")


def load_prediction(pred_dir, sha):
    """-> ('missing'|'failed'|'done', sorted event times, verdict)."""
    if not SHA_RE.match(sha):
        return "missing", [], None
    p = os.path.join(pred_dir, sha + ".json")
    try:
        with open(p, encoding="utf-8") as f:
            j = json.load(f)
    except (OSError, ValueError):
        return "missing", [], None
    if not isinstance(j, dict) or j.get("status") != "done" or j.get("verdict") not in ("violation", "no_violation"):
        return "failed", [], None
    ts = [float(v["timecode_sec"]) for v in j.get("violations") or []
          if isinstance(v, dict) and isinstance(v.get("timecode_sec"), (int, float))]
    if j["verdict"] == "violation" and not ts:
        ts = []  # positive clip without usable timestamps
    return "done", sorted(ts), j["verdict"]


def match_events(gt, pred, tol):
    """Greedy one-to-one nearest matching within tol. -> list of abs errors of matched pairs."""
    pairs = sorted(((abs(g - p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)
                    if abs(g - p) <= tol))
    ug, up, errs = set(), set(), []
    for e, i, j in pairs:
        if i in ug or j in up:
            continue
        ug.add(i); up.add(j); errs.append(e)
    return errs


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = (2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else None  # == 2pr/(p+r); 0.0 when P=R=0
    rd = lambda v: None if v is None else round(v, 4)
    return {"precision": rd(p), "recall": rd(r), "f1": rd(f)}


def evaluate(labels_path, split_path, pred_dir, tolerance_sec=5.0, min_holdout=5, source="human"):
    if source == "imported_dataset":
        events, status, bad = load_imported(labels_path)
    elif source == "human":
        events, status, bad = load_labels(labels_path)
    else:
        raise EvalRefused("unknown --source")
    side = load_split(split_path)
    holdout = sorted(s for s, v in side.items() if v == "holdout")
    train = [s for s, v in side.items() if v == "train"]
    assert_no_leakage(side, train_shas=train)
    imported_unknown = sum(status.get(s) == "unknown" for s in holdout)
    reviewed = [s for s in holdout if status.get(s, "unreviewed") not in ("unreviewed", "unknown")]
    unreviewed = len(holdout) - len(reviewed)
    if len(reviewed) < min_holdout:
        raise EvalRefused(f"REFUSED: reviewed holdout clips = {len(reviewed)} < required minimum {min_holdout} "
                          f"(holdout total {len(holdout)}, unreviewed {unreviewed}); metrics would be meaningless")
    c = dict(tp=0, fp=0, fn=0, tn=0)
    e = dict(tp=0, fp=0, fn=0)
    errs, failed, missing, inconsistent, unlocalised, pos_nopred = [], 0, 0, 0, 0, 0
    for s in reviewed:
        st, gt = status[s], events.get(s, [])
        if st == "conflict":
            inconsistent += 1
            continue
        if (st == "reviewed_no_violation" and gt):
            inconsistent += 1  # contradictory labels: excluded rather than guessed
            continue
        if st == "reviewed_violation" and not gt:
            unlocalised += 1  # clip-level positive without a human event time
        kind, pt, verdict = load_prediction(pred_dir, s)
        if kind != "done":
            failed += kind == "failed"
            missing += kind == "missing"
            pos_nopred += st == "reviewed_violation"
            continue
        pos_true, pos_pred = st == "reviewed_violation", verdict == "violation"
        c["tp" if pos_true and pos_pred else "fp" if pos_pred else "fn" if pos_true else "tn"] += 1
        m = match_events(gt, pt, tolerance_sec)
        errs += m
        e["tp"] += len(m); e["fp"] += len(pt) - len(m); e["fn"] += len(gt) - len(m)
    rec_all = c["tp"] / (c["tp"] + c["fn"] + pos_nopred) if c["tp"] + c["fn"] + pos_nopred else None
    errs.sort()
    return {
        "tolerance_sec": tolerance_sec,
        "holdout_clips_total": len(holdout), "train_clips": len(train),
        "holdout_reviewed": len(reviewed), "holdout_unreviewed_excluded": unreviewed,
        "source": source, "imported_unknown_excluded": imported_unknown,
        "reviewed_violation": sum(status[s] == "reviewed_violation" for s in reviewed),
        "reviewed_no_violation": sum(status[s] == "reviewed_no_violation" for s in reviewed),
        "excluded_inconsistent_labels": inconsistent, "violation_clips_without_event_time": unlocalised,
        "pipeline_failed": failed, "prediction_missing": missing, "malformed_label_lines": bad,
        "clip_level": {**c, **_prf(c["tp"], c["fp"], c["fn"]),
                       "recall_counting_failed_as_miss": None if rec_all is None else round(rec_all, 4)},
        "event_level": {**e, **_prf(e["tp"], e["fp"], e["fn"]),
                        "timestamp_mae_sec": round(sum(errs) / len(errs), 3) if errs else None,
                        "timestamp_median_abs_err_sec": round(errs[len(errs) // 2], 3) if errs else None,
                        "timestamp_max_abs_err_sec": round(errs[-1], 3) if errs else None},
        "note": "numbers only; no rule is enabled or validated by this tool",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=("human", "imported_dataset"), default="human")
    ap.add_argument("--cid", help="candidate id (imported_dataset defaults for labels/split)")
    ap.add_argument("--private", default="local_private")
    ap.add_argument("--labels")
    ap.add_argument("--split")
    ap.add_argument("--predictions")
    ap.add_argument("--tolerance-sec", type=float, default=5.0)
    ap.add_argument("--min-holdout", type=int, default=5)
    a = ap.parse_args(argv)
    if a.source == "imported_dataset" and a.cid and re.fullmatch(r"C\d{2,}", a.cid):
        a.labels = a.labels or os.path.join(a.private, "labels", a.cid + ".imported.jsonl")
        a.split = a.split or os.path.join(a.private, "splits", a.cid + ".json")
    a.predictions = a.predictions or os.path.join(a.private, "predictions")
    if not (a.labels and a.split):
        print("REFUSED: --labels and --split required (or --source imported_dataset --cid Cnn)")
        return 2
    try:
        out = evaluate(a.labels, a.split, a.predictions, a.tolerance_sec, a.min_holdout, a.source)
    except EvalRefused as ex:
        print(str(ex))
        return 2
    except (OSError, ValueError):
        print("cannot read inputs (static message; details withheld)")
        return 2
    if a.source == "imported_dataset":
        print("source: imported_dataset — по поставленной XLSX, без ручной проверки")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
