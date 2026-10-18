"""STAGE 8b (TRAIN only): feasibility / sufficiency gate on the interval snapshot. COUNTS ONLY; nothing is scored.

usage: python -m tools.stage8_audit --print-lock
       python -m tools.stage8_audit --candidate C02 --gate-only --expect-lock <lock16>
Implements docs/PROTOCOL_STAGE8.md exactly. No cache, clip file or frame is opened; holdout videos/features/scores
are never opened or computed (the shared split/imported-label files are read locally and non-train records are
discarded); nothing is trained, fitted or selected.
- Labels: ONLY the read-only interval snapshot (sha256 pinned for C02; verified BEFORE parsing; mismatch = refused).
- Units: local_safe.select_split_units(side "train", dry_run=True). Snapshot records whose sha16 is not a train
  unit of the right class are excluded and counted (never opened, never named).
- --print-lock: prints the lock digest (protocol + this file + the self-test); reads no data.
- --gate-only: refused unless the lock digest equals --expect-lock and no summary exists (single attempt);
  8 immutables + the snapshot hashed before/after; writes <LP>/train_probe/stage8_summary.json (O_EXCL, 0400);
  outcome STOP_INSUFFICIENT_DATA (with the unmet numbers) or GATE_PASS_SEPARATE_PROTOCOL_REQUIRED.
stdout and the summary carry counts/aggregates only: no shas, paths, names, per-clip rows, interval or label times.
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import train_probe as TP  # noqa: E402
from tools.annotate_intervals import IMMUTABLE, classify  # noqa: E402
from tools.local_safe import Refused, run_guarded, select_split_units  # noqa: E402

PROTOCOL = os.path.join(ROOT, "docs", "PROTOCOL_STAGE8.md")
LOCKED = (PROTOCOL, os.path.join(ROOT, "tools", "stage8_audit.py"), os.path.join(ROOT, "tools", "stage8_selftest.py"))
PIN = {"C02": ("labels/snapshots/C02.intervals.v1.20261018T0333Z.jsonl",
               "31c0a7548a928fd2d6e45ac0b6cab8d56fef64cd3e5970266a60d6ab57a9ca1b")}
GATE_MIN_POS, GATE_MIN_CTL = 4, 3          # clips (protocol §2)
SUMMARY = "stage8_summary.json"
DISCLOSURE = ("stage 8b: sufficiency gate only, counts only, no scores and no inference; units = clips; unknown tasks "
              "are a separate denominator, never negatives; imported negatives are not normal controls; train set "
              "already reused 4+ times (stages 4, 5, 7, human audit); holdout closed; production fail-closed")


def lp():
    return TP.LP


def fsha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def lock_digest():
    h, parts = hashlib.sha256(), {}
    for p in LOCKED:
        try:
            d = fsha(p)
        except OSError:
            d = "missing"
        parts[os.path.basename(p)] = d[:16]
        h.update(os.path.basename(p).encode() + b"\0" + d.encode() + b"\n")
    return h.hexdigest()[:16], parts


def immutables(cid):
    """-> (n, n matching the recorded sha16, tuple of full digests for a before/after comparison)."""
    dig, match = [], 0
    for _name, rel, exp in IMMUTABLE:
        try:
            d = fsha(os.path.join(lp(), rel.format(cid=cid)))
        except OSError:
            d = "absent"
        dig.append(d)
        match += d[:16] == exp
    return len(IMMUTABLE), match, tuple(dig)


def snapshot_spec(cid, rel, sha):
    if cid in PIN and (rel or PIN[cid][0], sha or PIN[cid][1]) != PIN[cid]:
        raise Refused("snapshot differs from the pinned C02 snapshot")
    if cid not in PIN and not (rel and sha):
        raise Refused("non-pinned candidate needs --snapshot and --snapshot-sha256")
    rel, sha = (rel or PIN[cid][0]), (sha or PIN[cid][1])
    if os.path.isabs(rel) or ".." in rel.split("/"):
        raise Refused("snapshot must be relative to the private dir")
    return os.path.join(lp(), rel), sha


def read_snapshot(path, sha):
    """verify the full sha256 BEFORE parsing; -> {task_id: latest-rev record} (same rule as annotate_intervals)."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        raise Refused("snapshot unreadable") from None
    if hashlib.sha256(raw).hexdigest() != sha:
        raise Refused("snapshot sha256 mismatch")
    latest = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and isinstance(r.get("task_id"), str) and isinstance(r.get("rev"), int):
            if r["rev"] >= latest.get(r["task_id"], {"rev": -1})["rev"]:
                latest[r["task_id"]] = r
    return latest


def train_sets(cid):
    units, cnt = select_split_units(lp(), cid, "train", dry_run=True)
    pos = {s for s, _p, st, _e in units if st == "reviewed_violation"}
    neg = {s for s, _p, st, _e in units if st == "reviewed_no_violation"}
    return pos, neg, cnt


# ============================================================================ gate (counts only)
def numeric(iv):
    return iv.get("start") is not None and iv.get("end") is not None


def overlap(a, b):
    s1 = a["start"] if a.get("start") is not None else 0.0
    e1 = a["end"] if a.get("end") is not None else float("inf")
    return s1 < b["end"] and b["start"] < e1


def confident_ctl(c):
    return bool(c.get("comparable_visible") and c.get("no_violation") and c.get("visibility") == "good"
                and c.get("confidence") == "sure" and numeric(c))


def gate(latest, pos, neg):
    """-> aggregates (protocol §2-§3). Units = clips. Records outside the train class sets are excluded, counted."""
    g = collections.Counter()
    vis, conf, flags = collections.Counter(), collections.Counter(), collections.Counter()
    ctl_clips = 0
    for r in latest.values():
        role, p, sha = r.get("role"), r.get("payload") or {}, r.get("sha16")
        if role not in ("positive", "control"):
            continue
        if (role == "positive" and sha not in pos) or (role == "control" and sha not in neg):
            g["n_records_excluded_not_train"] += 1
            continue
        if role == "control":
            g["n_external_control_tasks_generated"] += 1
            k = classify("control", p)
            g["n_external_controls" if k == "normal_confident" else
              {"normal_low": "n_external_controls_low_confidence", "unclear": "n_external_controls_unclear",
               None: "n_external_control_tasks_not_done"}[k]] += 1
            continue
        g["n_positive_tasks"] += 1
        g["n_positive_tasks_shown"] += bool(p.get("viewed"))
        g["n_positive_tasks_media_error"] += bool(p.get("media_error"))
        k = classify("positive", p)
        if k is None:
            g["n_positive_tasks_not_done"] += 1
            continue
        g["n_positive_tasks_done"] += 1
        g["n_control_not_found_set"] += bool(p.get("control_not_found"))
        ivs = p.get("intervals") or [] if k == "intervals" else []
        if k == "unknown":
            g["n_unknown_tasks"] += 1
            flags.update(f for f, v in (p.get("clip_flags") or {}).items() if v)
        for iv in ivs:
            g["n_intervals"] += 1
            g["n_intervals_numeric"] += numeric(iv)
            g["n_boundary_uncertain"] += bool(iv.get("start_uncertain")) + bool(iv.get("end_uncertain"))
            g["n_boundary_unknown"] += bool(iv.get("start_unknown")) + bool(iv.get("end_unknown"))
            g["n_good_sure"] += iv.get("visibility") == "good" and iv.get("confidence") == "sure"
            vis[str(iv.get("visibility"))] += 1
            conf[str(iv.get("confidence"))] += 1
        ctls, counts_p = [], any(numeric(iv) for iv in ivs)
        for c in p.get("same_clip_controls") or []:
            g["n_same_clip_control_entries"] += 1
            if not counts_p:   # S8b-1: only tasks of class intervals with >= 1 numeric interval feed C
                g["n_same_clip_controls_in_unknown_tasks"] += 1
            elif numeric(c) and any(overlap(e, c) for e in ivs if e.get("start") is not None or e.get("end") is not None):
                g["n_event_control_overlap"] += 1
            elif confident_ctl(c):
                ctls.append(c)
            else:
                g["n_same_clip_controls_low_or_unconfirmed"] += 1
        if ctls:
            g["n_same_clip_controls_confirmed_confident"] += 1
            ctl_clips += 1
        if counts_p:
            g["n_with_numeric_intervals"] += 1
    ctl_clips += g["n_external_controls"]
    keys = ("n_positive_tasks", "n_positive_tasks_shown", "n_positive_tasks_media_error", "n_positive_tasks_done",
            "n_positive_tasks_not_done", "n_with_numeric_intervals", "n_intervals", "n_intervals_numeric",
            "n_boundary_uncertain", "n_boundary_unknown", "n_good_sure", "n_unknown_tasks",
            "n_same_clip_control_entries", "n_same_clip_controls_confirmed_confident",
            "n_same_clip_controls_low_or_unconfirmed", "n_same_clip_controls_in_unknown_tasks",
            "n_event_control_overlap", "n_control_not_found_set",
            "n_external_control_tasks_generated", "n_external_controls", "n_external_controls_low_confidence",
            "n_external_controls_unclear", "n_external_control_tasks_not_done", "n_records_excluded_not_train")
    out = {k: int(g[k]) for k in keys}
    out.update(interval_visibility=dict(vis), interval_confidence=dict(conf), unknown_task_flags=dict(flags),
               n_train_positive_units=len(pos), n_train_negative_units=len(neg))
    unmet = []
    if out["n_with_numeric_intervals"] < GATE_MIN_POS:
        unmet.append(f"positives_with_numeric_intervals {out['n_with_numeric_intervals']} < {GATE_MIN_POS}")
    if ctl_clips < GATE_MIN_CTL:
        unmet.append(f"confirmed_confident_control_clips {ctl_clips} < {GATE_MIN_CTL}")
    out["gate"] = {"min_positives_numeric": GATE_MIN_POS, "min_control_clips_confirmed_confident": GATE_MIN_CTL,
                   "control_clips_confirmed_confident": ctl_clips, "pass": not unmet, "unmet": unmet,
                   "training_allowed_in_this_protocol": False}
    return out


def compute(cid, rel, sha):
    """gate only (protocol §2-§4): no cache or clip file is opened; non-train records are discarded."""
    path, sha = snapshot_spec(cid, rel, sha)
    latest = read_snapshot(path, sha)
    pos, neg, cnt = train_sets(cid)
    g = gate(latest, pos, neg)
    res = {"gate_aggregates": g, "selector_skipped": int(cnt["skipped"])}
    if g["gate"]["pass"]:
        res["outcome"] = "GATE_PASS_SEPARATE_PROTOCOL_REQUIRED"
        res["note"] = "this protocol never trains, fits thresholds or searches representations"
    else:
        res["outcome"] = "STOP_INSUFFICIENT_DATA"
        res["stop_reason"] = g["gate"]["unmet"]
    return res


# ============================================================================ CLI
def _print(r):
    print("GATE " + json.dumps(r["gate_aggregates"], sort_keys=True))
    print(f"OUTCOME={r['outcome']}" + (" unmet: " + "; ".join(r["stop_reason"]) if r.get("stop_reason") else ""))


def _main(argv=None):
    ap = argparse.ArgumentParser(description="stage 8b sufficiency gate (aggregates only)")
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--print-lock", action="store_true")
    m.add_argument("--gate-only", action="store_true")
    ap.add_argument("--candidate", default="")
    ap.add_argument("--expect-lock", default="")
    ap.add_argument("--snapshot", default=None, help="relative to the private dir (non-pinned candidates only)")
    ap.add_argument("--snapshot-sha256", default=None)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    lock, parts = lock_digest()
    print(f"lock_sha256_16={lock} files={json.dumps(parts, sort_keys=True)}")
    if a.print_lock:
        return 0
    if a.expect_lock != lock:
        raise Refused("lock digest mismatch: protocol or code changed since the lock")
    out = os.path.join(lp(), "train_probe", SUMMARY)
    if os.path.lexists(out):
        raise Refused("summary already exists: single pre-registered run")
    spath, ssha = snapshot_spec(a.candidate, a.snapshot, a.snapshot_sha256)
    h0, s0 = immutables(a.candidate), (fsha(spath) if os.path.exists(spath) else "absent")
    r = compute(a.candidate, a.snapshot, a.snapshot_sha256)
    h1, s1 = immutables(a.candidate), (fsha(spath) if os.path.exists(spath) else "absent")
    r.update(protocol="PROTOCOL_STAGE8", lock_sha256_16=lock, locked_files_sha256_16=parts,
             runtime_sec=round(time.time() - t0, 2), disclosure=DISCLOSURE,
             snapshot={"sha256_verified": s0 == ssha, "unchanged": s0 == s1},
             immutable_files={"n": h0[0], "match_recorded_sha16": h1[1],
                              "status": "unchanged" if h0[2] == h1[2] else "CHANGED"})
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "w") as f:
        json.dump(r, f, indent=1, sort_keys=True)
    _print(r)
    print(f"snapshot verified={int(s0 == ssha)} unchanged={int(s0 == s1)}; immutables n={h0[0]} "
          f"match_recorded={h1[1]} {r['immutable_files']['status']}; summary written (0400)")
    return 0


def main(argv=None):
    return run_guarded(_main, argv)


if __name__ == "__main__":
    sys.exit(main())
