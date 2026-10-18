"""Local, metadata-only inventory of a directory tree (usage: python -m tools.inventory ROOT).

PRIVACY CONTRACT: stdout carries only aggregates: per-extension counts, anonymous candidate
IDs (C01..) with counts/bytes, probe statistics, label-readiness booleans/counts. Never file
or directory names, never row values. The ID->path map goes ONLY to local_private/ (gitignored).
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import Counter

MEDIA_EXT = (".mov", ".mp4", ".avi", ".mkv")
LABEL_EXT = (".csv", ".json", ".xlsx")
PRINT_EXT = (".mov", ".mp4", ".avi", ".mkv", ".csv", ".json", ".xlsx", ".xls", ".txt", ".md", ".zip",
             ".pdf", ".jpg", ".png")
# column NAMES may be printed only if they are in this generic allow-list
NAME_ALLOW = {"id", "video", "file", "filename", "clip", "timestamp", "time", "start", "end",
              "sec", "label", "violation", "class", "type"}
TS_HINTS = ("timestamp", "time", "start", "end", "sec", "second", "seconds", "offset", "frame", "ts", "timecode")
VID_HINTS = ("video", "file", "clip", "filename", "path", "name", "id", "uuid")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIVATE = os.environ.get("LOCAL_PRIVATE_DIR") or os.path.join(ROOT, "local_private")


def _ext(name):
    return os.path.splitext(name)[1].lower()


def scan(root):
    """Metadata-only walk (os.walk + stat). Never opens a file. Returns (ext_counter, dirs).
    dirs: {abs_dir: {"media":[abs paths], "bytes":int, "labels":Counter}}"""
    exts, dirs = Counter(), {}
    for dp, _dn, fns in os.walk(root, followlinks=False):
        for fn in fns:
            p = os.path.join(dp, fn)
            if os.path.islink(p):
                continue
            e = _ext(fn)
            exts[e or "(none)"] += 1
            if e in MEDIA_EXT or e in LABEL_EXT:
                d = dirs.setdefault(dp, {"media": [], "bytes": 0, "labels": Counter()})
                if e in MEDIA_EXT:
                    try:
                        d["bytes"] += os.stat(p).st_size
                    except OSError:
                        continue
                    d["media"].append(p)
                else:
                    d["labels"][e] += 1
    return exts, dirs


def candidates(dirs):
    """Directories with >=1 media file, largest media count first; anonymous IDs."""
    c = sorted(((d, v) for d, v in dirs.items() if v["media"]),
               key=lambda kv: (-len(kv[1]["media"]), -kv[1]["bytes"], kv[0]))
    width = max(2, len(str(len(c))))
    return [(f"C{i + 1:0{width}d}", d, v) for i, (d, v) in enumerate(c)]


def write_map(cands, private=PRIVATE, force=False):
    """Write ID->path map. Refuses to replace an existing DIFFERENT map unless force (old one is kept
    as inventory_map.prev.json, 0600). Returns True if written/unchanged, False if refused."""
    os.makedirs(private, exist_ok=True)
    p = os.path.join(private, "inventory_map.json")
    new = {cid: d for cid, d, _ in cands}
    if os.path.exists(p):
        try:
            old = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            old = None
        if old == new:
            return True
        if not force:
            return False
        prev = os.path.join(private, "inventory_map.prev.json")
        with open(p, "rb") as f, open(prev, "wb") as g:
            g.write(f.read())
        os.chmod(prev, 0o600)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(new, f, ensure_ascii=False, indent=1)
    os.chmod(p, 0o600)
    return True


_CHILD = (
    "import sys,json\n"
    "import cv2\n"
    "c=cv2.VideoCapture(json.loads(sys.stdin.readline()))\n"
    "r=[c.get(cv2.CAP_PROP_FPS),c.get(cv2.CAP_PROP_FRAME_COUNT),c.get(cv2.CAP_PROP_FRAME_WIDTH),c.get(cv2.CAP_PROP_FRAME_HEIGHT)]\n"
    "c.release()\n"
    "sys.stdout.write('\\n'+json.dumps([float(x) for x in r])+'\\n')\n"
)


def _probe_one(path, timeout=20):
    """Isolated child: stderr -> DEVNULL (OpenCV/ffmpeg warnings may embed paths); only a numeric
    JSON list is accepted from the child's stdout. Any failure -> None (counted as unprobed)."""
    import subprocess
    try:
        r = subprocess.run([sys.executable, "-I", "-c", _CHILD], input=(json.dumps(path) + "\n").encode(),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout)  # path via stdin, not argv
        v = json.loads(r.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
        if len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
            return v
    except Exception:  # noqa: BLE001 - never echo anything about the failure
        pass
    return None


def probe(paths, timeout=20):
    """Aggregates only (no names): duration/size stats, duration buckets, fps/resolution sets."""
    durs, sizes, fps_set, res_set, bad, fast, both = [], [], set(), set(), 0, 0, 0
    for p in paths:
        v = _probe_one(p, timeout)
        if not v or not (v[0] > 0 and v[1] > 0 and v[2] > 0 and v[3] > 0):
            bad += 1
            continue
        fps, n, w, h = v
        durs.append(n / fps)
        try:
            sizes.append(os.stat(p).st_size)
        except OSError:
            pass
        fps_set.add(round(fps, 2))
        res_set.add(f"{int(w)}x{int(h)}")
        fast += fps >= 29
        both += fps >= 29 and 10 <= n / fps <= 145
    out = {"probed_ok": len(durs), "unprobed": bad, "fps_ge_29": fast,
           "count_duration_10_145": sum(10 <= d <= 145 for d in durs), "count_fps_ge_29_and_dur_ok": both,
           "fps_set": sorted(fps_set), "resolution_set": sorted(res_set)}
    if durs:
        out.update(duration_min=round(min(durs), 2), duration_median=round(statistics.median(durs), 2),
                   duration_max=round(max(durs), 2),
                   duration_buckets={"<=30": sum(d <= 30 for d in durs), "30-60": sum(30 < d <= 60 for d in durs),
                                     "60-120": sum(60 < d <= 120 for d in durs), ">120": sum(d > 120 for d in durs)})
    if sizes:
        out.update(size_bytes_min=min(sizes), size_bytes_median=int(statistics.median(sizes)), size_bytes_max=max(sizes))
    return out


# ---------------------------------------------------------------- label readiness
def _hint(name, hints):
    toks = [t for t in re.split(r"[^a-z0-9]+", str(name).strip().lower()) if t]
    return any(t == h or (len(h) >= 4 and h in t) for t in toks for h in hints)


def _safe_name(n):
    n = str(n).strip().lower()
    return n if n in NAME_ALLOW else None


def _read_csv(p):
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        rows = list(csv.reader(f))
    return (rows[0], len(rows) - 1) if rows else ([], 0)


def _read_json(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        o = json.load(f)
    if isinstance(o, dict):
        for v in o.values():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                o = v
                break
    if isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
        keys = []
        for x in o[:200]:
            for k in x:
                if k not in keys:
                    keys.append(k)
        return keys, len(o)
    if isinstance(o, dict):
        return list(o.keys()), 1
    return [], len(o) if isinstance(o, list) else 0


def _read_xlsx(p):
    import openpyxl
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    try:
        ws = wb.worksheets[0]
        it = ws.iter_rows(values_only=True)
        hdr = next(it, None) or ()
        n = sum(1 for _ in it)
        return [("" if c is None else str(c)) for c in hdr], n
    finally:
        wb.close()


def label_readiness(dirpath):
    """Aggregate-only label check for one directory. Returns dict; no row values, no file names."""
    res = {}
    names_seen = set()
    for ext, reader in ((".csv", _read_csv), (".json", _read_json), (".xlsx", _read_xlsx)):
        files = sorted(f for f in os.listdir(dirpath)
                       if _ext(f) == ext and os.path.isfile(os.path.join(dirpath, f)))
        if not files:
            continue
        if ext == ".xlsx":
            try:
                import openpyxl  # noqa: F401
            except ImportError:
                res["xlsx"] = "xlsx: parser unavailable"
                continue
        st = {"files": len(files), "parsed": 0, "unparsed": 0, "rows_total": 0, "cols_min": None,
              "cols_max": None, "files_with_timestamp_col": 0, "files_with_video_id_col": 0}
        for f in files:
            try:
                hdr, n = reader(os.path.join(dirpath, f))
            except Exception:  # noqa: BLE001 - never echo error text (could contain content)
                st["unparsed"] += 1
                continue
            st["parsed"] += 1
            st["rows_total"] += n
            c = len(hdr)
            st["cols_min"] = c if st["cols_min"] is None else min(st["cols_min"], c)
            st["cols_max"] = c if st["cols_max"] is None else max(st["cols_max"], c)
            st["files_with_timestamp_col"] += any(_hint(h, TS_HINTS) for h in hdr)
            st["files_with_video_id_col"] += any(_hint(h, VID_HINTS) for h in hdr)
            names_seen.update(x for x in map(_safe_name, hdr) if x)
        res[ext[1:]] = st
    res["generic_column_names_seen"] = sorted(names_seen)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--probe", nargs="?", const="ALL", metavar="ID",
                    help="cv2 (isolated child) aggregates; optional candidate ID, default whole root")
    ap.add_argument("--force", action="store_true", help="overwrite an existing, different inventory_map.json")
    ap.add_argument("--labels", metavar="ID", help="label readiness for a candidate ID (e.g. C01)")
    a = ap.parse_args(argv)
    root = os.path.abspath(os.path.expanduser(a.root))
    if not os.path.isdir(root):
        print("root: not a directory")
        return 2
    exts, dirs = scan(root)
    cands = candidates(dirs)
    if not write_map(cands, PRIVATE, a.force):
        print("inventory_map exists — use --force to overwrite")
        return 2
    if a.labels:
        m = {cid: d for cid, d, _ in cands}
        if a.labels not in m:
            print("unknown candidate id")
            return 2
        try:
            print(json.dumps(label_readiness(m[a.labels]), indent=1))
        except Exception:  # noqa: BLE001
            print("labels: unreadable")
        return 0
    known = {e: n for e, n in exts.items() if e in PRINT_EXT}
    other = sum(n for e, n in exts.items() if e not in PRINT_EXT)  # arbitrary extensions can leak names
    print("extensions:", ", ".join(f"{e}={n}" for e, n in sorted(known.items(), key=lambda x: -x[1])), f"other_count={other}")
    print(f"candidates: {len(cands)} (map written to local_private/inventory_map.json)")
    print(f"{'id':<5}{'media':>7}{'bytes':>15}{'csv':>5}{'json':>6}{'xlsx':>6}")
    for cid, _d, v in cands:
        lb = v["labels"]
        print(f"{cid:<5}{len(v['media']):>7}{v['bytes']:>15}{lb['.csv']:>5}{lb['.json']:>6}{lb['.xlsx']:>6}")
    if a.probe:
        sel = [(c, v) for c, _d, v in cands if a.probe in ("ALL", c)]
        if not sel:
            print("unknown candidate id")
            return 2
        print(f"probe[{a.probe}]:", json.dumps(probe([p for _c, v in sel for p in v["media"]])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
