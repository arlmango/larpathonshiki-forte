"""Shared fail-closed helpers for local tools that touch real clips (run_local, train_probe).

- run_guarded(): no str(e) / traceback ever reaches the terminal (they can contain real paths);
  only the exception class name is printed, exit code 2.
- select_split_units(): picks files of ONE split side BEFORE any file is opened, using the split
  file's sha list and the clip-index -> sha map of the imported label records. Records of the other
  side are decoded only far enough to see their side and then discarded (only their clip index is
  kept, to refuse an index that is claimed by both sides). Each selected file is re-hashed after
  opening; unreadable / mismatching files are counted, never named.
"""
import hashlib
import json
import os
import re
import sys

CID_RE = re.compile(r"^C\d{2,4}$")
LABEL_STATUSES = ("reviewed_violation", "reviewed_no_violation", "unknown")


class Refused(Exception):
    """Static, content-free reason (safe to print)."""


def _excepthook(exc_type, exc, tb):
    try:
        sys.stderr.write(f"error: {exc_type.__name__}\n")
        sys.stderr.flush()
    finally:
        os._exit(2)


def run_guarded(fn, argv=None):
    sys.excepthook = _excepthook
    try:
        return fn(argv)
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 - static output only
        print(f"error: {e.__class__.__name__}", file=sys.stderr)
        return 2


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def select_split_units(lp, cid, want, dry_run=False):
    """-> (units, counts). units: [(sha, path, status, [event times])] for side `want` only.
    counts: {"skipped": n (no single status/index, index conflict or out of range),
             "unreadable": n, "index_mismatch": n}. No file of another side is ever opened."""
    from tools.annotate import list_clips
    from tools.evaluate import load_split
    if not CID_RE.match(cid or ""):
        raise Refused("candidate id must look like C01")
    if want not in ("train", "holdout"):
        raise Refused("split side must be train or holdout")
    try:
        side = load_split(os.path.join(lp, "splits", f"{cid}.json"))
        lab = open(os.path.join(lp, "labels", f"{cid}.imported.jsonl"), encoding="utf-8")
    except (OSError, ValueError, Exception):  # noqa: BLE001 - EvalRefused etc.: static message
        raise Refused("split or imported label file missing or malformed") from None
    wanted = {s for s, v in side.items() if v == want}
    st_seen, events, idx_of, foreign_idx = {}, {}, {}, set()
    with lab:
        for line in lab:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not isinstance(r, dict):
                continue
            sha = r.get("sha16") or r.get("sha")
            if sha not in wanted:  # other side / unknown: discard immediately, keep index only
                if isinstance(r.get("clip"), int):
                    foreign_idx.add(r["clip"])
                continue
            if r.get("kind") == "imported_clip_status" and r.get("status") in LABEL_STATUSES:
                st_seen.setdefault(sha, set()).add(r["status"])
                if isinstance(r.get("clip"), int):
                    idx_of.setdefault(sha, set()).add(r["clip"])
            elif r.get("kind") == "imported_event":
                t = r.get("timestamp_sec")
                if isinstance(t, (int, float)) and not isinstance(t, bool) and t >= 0:
                    events.setdefault(sha, set()).add(round(float(t), 3))
    try:
        with open(os.path.join(lp, "inventory_map.json"), encoding="utf-8") as f:
            clips = list_clips(json.load(f)[cid])
    except (OSError, ValueError, KeyError, TypeError):
        raise Refused("inventory map missing or candidate directory unreadable") from None
    units, counts = [], {"skipped": 0, "unreadable": 0, "index_mismatch": 0}
    for sha in sorted(wanted):
        real = st_seen.get(sha, set()) - {"unknown"}
        status = next(iter(real)) if len(real) == 1 else ("unknown" if not real and sha in st_seen else None)
        idxs = idx_of.get(sha, set())
        i = min(idxs) if idxs else -1
        # explicit pre-open checks (not assert: must hold under python -O)
        if status is None or side.get(sha) != want or i in foreign_idx or not 0 <= i < len(clips):
            counts["skipped"] += 1
            continue
        if dry_run:
            units.append((sha, None, status, sorted(events.get(sha, set()))))
            continue
        try:
            got = sha16(clips[i])
        except OSError:
            counts["unreadable"] += 1
            continue
        if got != sha:  # post-open re-check
            counts["index_mismatch"] += 1
            continue
        units.append((sha, clips[i], status, sorted(events.get(sha, set()))))
    return units, counts
