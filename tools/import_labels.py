"""Local importer of a human-made XLSX with labels. usage:
  python -m tools.import_labels --candidate C02 [--profile]
  python -m tools.import_labels --candidate C02 --import --mapping mapping.json

PRIVACY CONTRACT: stdout/stderr carry only generic counts and verdict words. Never headers, cell
values, file names or paths -- not even in errors (all exceptions are swallowed into fixed words).
Output records go to local_private/labels/<cid>.imported.jsonl (separate from human labels) and are
`imported_unconfirmed`: they are NOT ground truth and never validate a rule.
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
import time

from .annotate import _atomic_write, list_clips, sha_prefix
from .inventory import PRIVATE

TIME_RE = re.compile(r"(?<![\d:])(\d{1,3}):(\d{2})(?::(\d{2}))?(?:[.,](\d+))?(?![\d:])")
SEC_RE = re.compile(r"(?<![\d.,:])(\d+(?:[.,]\d+)?)\s*(?:s|sec|с|сек)(?![A-Za-zА-Яа-я])", re.I)
BINARY = {"0", "1", "true", "false", "yes", "no", "да", "нет", "violation", "no_violation"}
UNITS = ("sec", "mm:ss", "hh:mm:ss", "frames", "excel_time", "text_list")
CYR = re.compile("[А-Яа-яЁё]")
RU_POS = {"нарушение", "нарушения", "есть нарушение", "есть нарушения", "с нарушением", "с нарушениями"}
RU_NEG = {"нет нарушения", "нет нарушений", "без нарушения", "без нарушений", "нарушений нет", "не нарушение"}
_UNIT = r"(?:секунды|секунда|секунд|сек\.?|с\.?)"
RU_SEC_RE = re.compile(r"^\d+([.,]\d+)?\s*" + _UNIT + "$", re.I)
MINSEC_RE = re.compile(r"^(\d+)\s*(?:минуты|минута|минут|мин\.?)\s*(\d+(?:[.,]\d+)?)\s*" + _UNIT + "$", re.I)
SECTOK = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s*" + _UNIT + r"(?![A-Za-zА-Яа-я])", re.I)
SEP = r"\s*(?:,|;|/|или)\s*"


def ru_kind(s):
    """Whole-cell classification of a (stripped) string: pos|neg|sec1|None."""
    k = " ".join(s.strip().lower().split())
    if k in RU_POS:
        return "pos"
    if k in RU_NEG:
        return "neg"
    if RU_SEC_RE.match(k):
        return "sec1"
    m = MINSEC_RE.match(k)
    return "minsec" if m and float(m.group(2).replace(",", ".")) < 60 else None


def ru_single_seconds(s):
    """Seconds of a sec1 / minsec cell, else None."""
    k = " ".join(s.strip().lower().split())
    kind = ru_kind(k)
    if kind == "sec1":
        return float(SECTOK.search(k).group(1).replace(",", "."))
    if kind == "minsec":
        m = MINSEC_RE.match(k)
        return int(m.group(1)) * 60 + float(m.group(2).replace(",", "."))
    return None


def ru_shape(s):
    """Shape class of an unclassified cell (never the cell itself): list|range|minutes|other, plus tokens."""
    k = " ".join(s.strip().lower().split())
    toks = [float(m.group(1).replace(",", ".")) for m in SECTOK.finditer(k)]
    t = SECTOK.sub("T", k)
    if len(toks) >= 2 and re.fullmatch(r"T(?:" + SEP + "T)+", t):
        return "list", toks
    if len(toks) == 2 and re.fullmatch(r"(?:с\s*)?T\s*(?:[-–—]|по|до)\s*T", t):
        return "range", toks
    if "мин" in k:
        return "minutes", toks
    return "other", toks


# ------------------------------------------------------------------ io
def locate_xlsx(private, cid):
    """Exactly one .xlsx in dirname(map[cid]) -> path; else (None, count). Path is never printed."""
    try:
        m = json.load(open(os.path.join(private, "inventory_map.json"), encoding="utf-8"))
        parent = os.path.dirname(m[cid].rstrip("/"))
        found = glob.glob(os.path.join(glob.escape(parent), "*.xlsx"))
    except Exception:  # noqa: BLE001
        return None, 0
    return (found[0], 1) if len(found) == 1 else (None, len(found))


def load_rows(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return [list(r) for r in wb.worksheets[0].iter_rows(values_only=True)]
    finally:
        wb.close()


def clip_index(private, cid):
    m = json.load(open(os.path.join(private, "inventory_map.json"), encoding="utf-8"))
    clips = list_clips(m[cid])
    key = {}
    for i, p in enumerate(clips):
        b = os.path.basename(p).lower()
        key[b] = i
        key.setdefault(os.path.splitext(b)[0], i)
    return clips, key


def _s(v):
    return "" if v is None else str(v).strip()


# ------------------------------------------------------------------ time parsing
def time_tokens(text):
    """Seconds for every time-like token in text. 1-colon tokens are returned as ('m', a, b)."""
    out = []
    rest = text
    for m in TIME_RE.finditer(text):
        a, b, c, f = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
        frac = float("0." + f) if f else 0.0
        out.append((2, a, b, int(c), frac) if c is not None else (1, a, b, 0, frac))
    rest = TIME_RE.sub(" ", rest)
    for m in SEC_RE.finditer(rest):
        out.append((0, float(m.group(1).replace(",", ".")), 0, 0, 0.0))
    return out


def tok_seconds(t, colon1="mm:ss"):
    n, a, b, c, frac = t
    if n == 0:
        return a
    if n == 1:
        return (a * 60 + b if colon1 == "mm:ss" else a * 3600 + b * 60) + frac
    return a * 3600 + b * 60 + c + frac


def cell_seconds(v, unit, fps=30.0, colon1="mm:ss"):
    """Parse one cell under an explicit unit -> list of seconds. Raises ValueError if unparseable."""
    if v is None or _s(v) == "":
        return []
    if unit == "excel_time":
        if isinstance(v, dt.datetime):
            v = v.time()
        if isinstance(v, dt.time):
            return [v.hour * 3600 + v.minute * 60 + v.second + v.microsecond / 1e6]
        if isinstance(v, dt.timedelta):
            return [v.total_seconds()]
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v < 1:
            return [float(v) * 86400]
        raise ValueError
    if unit in ("sec", "frames"):
        if isinstance(v, bool):
            raise ValueError
        x = float(v) if isinstance(v, (int, float)) else float(_s(v).replace(",", "."))
        return [x / fps if unit == "frames" else x]
    if unit in ("mm:ss", "hh:mm:ss"):
        if isinstance(v, dt.time):
            return [v.hour * 3600 + v.minute * 60 + v.second]
        toks = time_tokens(_s(v))
        if len(toks) != 1 or toks[0][0] == 0 or (toks[0][0] == 2) != (unit == "hh:mm:ss"):
            raise ValueError
        return [tok_seconds(toks[0], "mm:ss")]
    if unit == "text_list":
        toks = time_tokens(_s(v))
        if not toks:
            raise ValueError
        return [tok_seconds(t, colon1) for t in toks]
    raise ValueError


# ------------------------------------------------------------------ profile
def _bucket(n):
    return "le10" if n <= 10 else ("11_40" if n <= 40 else "gt40")


def profile_column(cells, stems):
    c = dict(n_empty=0, n_numeric_int=0, n_numeric_float=0, n_excel_time=0, n_hhmm=0, n_hhmmss=0,
             n_mmss_list=0, n_sec_token=0, n_contains_digits=0, n_text_no_digits=0, n_cyrillic=0,
             len_le10=0, len_11_40=0, len_gt40=0, n_matches_video_stem=0, n_ru_binary_pos=0, n_ru_binary_neg=0,
             n_ru_seconds_single=0, n_ru_minutes_seconds_compound=0, shape_list_seconds=0, shape_range_seconds=0, shape_minutes=0, shape_other=0)
    tokcells, ncol, binary_hits, nonempty = 0, set(), 0, 0
    first_parts, tokmix = [], False
    for v in cells:
        s = _s(v)
        if s == "":
            c["n_empty"] += 1
            continue
        nonempty += 1
        c["len_" + _bucket(len(s))] += 1
        c["n_matches_video_stem"] += s.lower() in stems
        rk = ru_kind(s) if isinstance(v, str) else None
        c["n_ru_binary_pos"] += rk == "pos"
        c["n_ru_binary_neg"] += rk == "neg"
        c["n_ru_seconds_single"] += rk == "sec1"
        c["n_ru_minutes_seconds_compound"] += rk == "minsec"
        if rk is None and s.lower() not in stems:
            c["shape_" + {"list": "list_seconds", "range": "range_seconds", "minutes": "minutes",
                          "other": "other"}[ru_shape(s)[0]]] += 1
        c["n_cyrillic"] += bool(CYR.search(s))
        c["n_contains_digits"] += any(ch.isdigit() for ch in s)
        if s.lower() in BINARY:
            binary_hits += 1
        if isinstance(v, (dt.time, dt.datetime, dt.timedelta)):
            c["n_excel_time"] += 1
            tokcells += 1
            ncol.add("x")
            continue
        if isinstance(v, bool):
            c["n_text_no_digits"] += 1
            continue
        if isinstance(v, int) or re.fullmatch(r"\d+", s):
            c["n_numeric_int"] += 1
            continue
        if isinstance(v, float) or re.fullmatch(r"\d+[.,]\d+", s):
            c["n_numeric_float"] += 1
            continue
        toks = time_tokens(s)
        if re.fullmatch(r"\d{1,2}:\d{2}", s):
            c["n_hhmm"] += 1
        if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", s):
            c["n_hhmmss"] += 1
        if len(toks) >= 2:
            c["n_mmss_list"] += 1
        if SEC_RE.search(s):
            c["n_sec_token"] += 1
        if not any(ch.isdigit() for ch in s):
            c["n_text_no_digits"] += 1
        if toks:
            tokcells += 1
            for t in toks:
                ncol.add(t[0])
                if t[0] == 1:
                    first_parts.append(t[1])
            if len(toks) >= 2 or not re.fullmatch(r"[\d:.,]+", s):
                tokmix = True
    c["_nonempty"], c["_tokcells"], c["_binary_hits"] = nonempty, tokcells, binary_hits
    c["_numeric"] = c["n_numeric_int"] + c["n_numeric_float"]
    c["_colons"], c["_max_first1"], c["_tokmix"] = ncol, max(first_parts, default=0), tokmix
    return c


def column_format(c):
    """Conservative: 'none' | 'ambiguous' | unambiguous format name. Needs >=95 % consistent cells."""
    ne = c["_nonempty"]
    if ne == 0:
        return "none"
    if c["n_ru_seconds_single"] == ne:
        return "ru_seconds"  # unit is stated in the text itself
    if c["n_ru_seconds_single"] + c["n_ru_minutes_seconds_compound"] == ne:
        return "ru_time_text"  # seconds and min+sec, units stated in the text
    if c["_tokcells"] == 0 and c["_numeric"] == 0:
        return "none"
    if c["_numeric"] and c["_numeric"] >= 0.5 * ne:
        return "ambiguous"  # bare numbers: sec vs frames vs minutes cannot be told apart
    if c["_tokcells"] < 0.95 * ne:
        return "ambiguous"
    ks = c["_colons"]
    if ks == {"x"}:
        return "excel_time"
    if "x" in ks or len(ks) != 1:
        return "ambiguous"
    (k,) = ks
    if c["_tokmix"]:
        return "ambiguous" if k == 1 and c["_max_first1"] <= 23 else "text_list"
    if k == 2:
        return "hh:mm:ss"
    if k == 1:
        return "mm:ss" if c["_max_first1"] > 23 else "ambiguous"  # h:mm vs m:ss
    return "ambiguous"


def column_binary(c):
    ne = c["_nonempty"]
    if ne and c["n_ru_binary_pos"] + c["n_ru_binary_neg"] == ne:
        return "ru"
    if ne == 0 or c["_binary_hits"] == 0:
        return "none"
    return "yes" if c["_binary_hits"] >= 0.95 * ne else "ambiguous"


def verdicts(profiles, name_col):
    fmts = {i: column_format(p) for i, p in profiles.items() if i != name_col}
    amb = [i for i, f in fmts.items() if f == "ambiguous"]
    good = [(i, f) for i, f in fmts.items() if f not in ("none", "ambiguous")]
    if amb or len(good) > 1:
        tf = "ambiguous"
    elif good:
        tf = f"unambiguous:{good[0][1]}"
    else:
        tf = "none"
    bins = {i: column_binary(p) for i, p in profiles.items() if i != name_col}
    byru = [i for i, b in bins.items() if b == "ru"]
    by = [i for i, b in bins.items() if b == "yes"]
    if len(byru) == 1 and not by:
        return tf, f"col:{byru[0]} (ru_binary)"
    bl = "ambiguous" if (len(by) > 1 or any(b == "ambiguous" for b in bins.values())) else (
        f"col:{by[0]}" if by else "none")
    return tf, bl


def profile(rows, stems, header_rows=1):
    body = rows[header_rows:]
    ncols = max((len(r) for r in rows), default=0)
    profs = {i + 1: profile_column([r[i] if i < len(r) else None for r in body], stems) for i in range(ncols)}
    name_col = max(profs, key=lambda i: profs[i]["n_matches_video_stem"], default=None)
    if name_col is not None and profs[name_col]["n_matches_video_stem"] == 0:
        name_col = None
    return profs, name_col, *verdicts(profs, name_col)


# ------------------------------------------------------------------ import
def import_records(rows, mapping, key, clips):
    """Pure conversion -> (records, counts). Status only from an explicit value list; empty != no_violation."""
    hdr = int(mapping.get("header_rows", 1))
    fn, tc = mapping.get("filename_col"), mapping.get("timecode_col")
    sc, nc = mapping.get("status_col"), mapping.get("note_col")
    unit, fps = mapping.get("timecode_unit"), float(mapping.get("fps", 30))
    colon1 = mapping.get("colon1_is", "mm:ss")
    nov = {str(x).strip().lower() for x in (mapping.get("no_violation_values") or []) if str(x).strip()}
    yes = {str(x).strip().lower() for x in (mapping.get("violation_values") or []) if str(x).strip()}
    if fn is None or (tc is not None and unit not in UNITS) or not 1 <= fps <= 240:
        raise ValueError("mapping")
    cell = lambda r, c: r[c - 1] if c and c - 1 < len(r) else None  # noqa: E731
    recs, shas = [], {}
    cnt = dict(rows=0, matched=0, unmatched=0, events=0, parse_errors=0, statuses_set=0)
    for r in rows[hdr:]:
        cnt["rows"] += 1
        i = key.get(_s(cell(r, fn)).lower())
        if i is None:
            cnt["unmatched"] += 1
            continue
        cnt["matched"] += 1
        sha = shas.setdefault(i, sha_prefix(clips[i]))
        if tc:
            try:
                for sec in cell_seconds(cell(r, tc), unit, fps, colon1):
                    if not 0 <= sec < 86400:
                        raise ValueError
                    rec = {"kind": "imported_event", "clip": i, "sha": sha, "sha16": sha,
                           "timestamp_sec": round(sec, 3), "source": "xlsx", "status": "imported_unconfirmed"}
                    if nc and _s(cell(r, nc)):
                        rec["note"] = _s(cell(r, nc))[:2000]
                    recs.append(rec)
                    cnt["events"] += 1
            except (ValueError, TypeError, OverflowError):
                cnt["parse_errors"] += 1
        st = "unknown"
        v = _s(cell(r, sc)).lower() if sc else ""
        if v and v in nov:
            st = "reviewed_no_violation"
        elif v and v in yes:
            st = "reviewed_violation"
        cnt["statuses_set"] += st != "unknown"
        recs.append({"kind": "imported_clip_status", "clip": i, "sha": sha, "sha16": sha, "status": st,
                     "state": "imported_unconfirmed", "source": "xlsx", "ts": round(time.time(), 3)})
    cnt["timecode_format"] = unit if tc else "none"
    return recs, cnt


def detect_ru(rows, key):
    profs, name_col, _tf, _bl = profile(rows, set(key))
    bins = [i for i, p in profs.items() if i != name_col and column_binary(p) == "ru"]
    secs = [i for i, p in profs.items() if i not in (name_col, *bins) and p["n_ru_seconds_single"] + p["n_ru_minutes_seconds_compound"] > 0]
    if name_col is None or len(bins) != 1 or len(secs) > 1:
        raise ValueError("auto-ru")
    return name_col, bins[0], (secs[0] if secs else None)


def _probe_durations(clips, need):
    from .inventory import _probe_one
    out = {}
    for i in need:
        v = _probe_one(clips[i])
        out[i] = v[1] / v[0] if v and v[0] > 0 and v[1] > 0 else None
    return out


def import_auto_ru(rows, key, clips, hdr=1, durations=None):
    """Fixed Russian allow-lists only. Anything not explicit or failing a cross-check -> needs_user.
    durations: optional {clip_idx: seconds|None}; default = isolated probe (no names)."""
    fn, bc, tc = detect_ru(rows, key)
    cnt = dict(rows=0, matched=0, unmatched=0, events=0, parse_errors=0, statuses_set=0, needs_user=0,
               positive_with_time=0, positive_without_time=0, negative_with_time=0,
               n_time_within_duration=0, n_time_exceeds_duration=0, n_duration_unknown=0)
    cell = lambda r, c: r[c - 1] if c and c - 1 < len(r) else None  # noqa: E731
    plan, shas = [], {}
    for n, r in enumerate(rows[hdr:], start=1):
        cnt["rows"] += 1
        i = key.get(_s(cell(r, fn)).lower())
        if i is None:
            cnt["unmatched"] += 1
            continue
        cnt["matched"] += 1
        shas.setdefault(i, sha_prefix(clips[i]))
        bk = ru_kind(_s(cell(r, bc)))
        ts = _s(cell(r, tc)) if tc else ""
        evs, reason = [], None
        if ts:
            one = ru_single_seconds(ts)
            if one is not None:
                evs = [(one, None)]
            else:
                shape, toks = ru_shape(ts)
                if shape == "list":
                    evs = [(x, None) for x in toks]
                elif shape == "range":
                    evs = [(toks[0], toks[1])]
                else:
                    reason = "shape_" + shape
        elif bk == "pos":
            reason = "positive_without_time"
        cnt["positive_with_time"] += bk == "pos" and bool(ts)
        cnt["positive_without_time"] += bk == "pos" and not ts
        if bk == "neg" and ts:
            cnt["negative_with_time"] += 1
            reason, evs = "negative_with_time", []
        plan.append((n, i, bk, evs, reason))
    need = sorted({i for _n, i, _b, evs, _r in plan if evs})
    dur = durations if durations is not None else _probe_durations(clips, need)
    recs = []
    for n, i, bk, evs, reason in plan:
        sha = shas[i]
        st = {"pos": "reviewed_violation", "neg": "reviewed_no_violation"}.get(bk, "unknown")
        cnt["statuses_set"] += st != "unknown"
        recs.append({"kind": "imported_clip_status", "clip": i, "sha": sha, "sha16": sha, "status": st,
                     "state": "imported_unconfirmed", "source": "xlsx", "ts": round(time.time(), 3)})
        d = dur.get(i)
        if evs and d is None:
            cnt["n_duration_unknown"] += 1
            reason, evs = reason or "duration_unknown", []
        elif evs and any(a < 0 or a > d or (b is not None and b > d) for a, b in evs):
            cnt["n_time_exceeds_duration"] += len(evs)
            reason, evs = "time_exceeds_duration", []
        else:
            cnt["n_time_within_duration"] += len(evs)
        for a, b in evs:
            rec = {"kind": "imported_event", "clip": i, "sha": sha, "sha16": sha, "timestamp_sec": round(a, 3),
                   "source": "xlsx", "status": "imported_unconfirmed"}
            if b is not None:
                rec["range_end_sec"] = round(b, 3)
            recs.append(rec)
            cnt["events"] += 1
        if reason:
            cnt["needs_user"] += 1
            recs.append({"kind": "imported_needs_user", "clip": i, "sha": sha, "row": n, "reason": reason})
    cnt["timecode_format"] = "ru_time_text"
    return recs, cnt


def run_import(private, cid, mapping=None, auto_ru=False):
    path, n = locate_xlsx(private, cid)
    if path is None:
        raise LookupError(n)
    clips, key = clip_index(private, cid)
    rows = load_rows(path)
    recs, cnt = import_auto_ru(rows, key, clips) if auto_ru else import_records(rows, mapping, key, clips)
    _atomic_write(os.path.join(private, "labels", cid + ".imported.jsonl"),
                  "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    return cnt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--profile", action="store_true")
    g.add_argument("--import", dest="do_import", action="store_true")
    ap.add_argument("--mapping")
    ap.add_argument("--auto-ru", action="store_true", help="fixed Russian allow-list import, no mapping")
    a = ap.parse_args(argv)
    path, n = locate_xlsx(PRIVATE, a.candidate)
    if path is None:
        print(f"xlsx files found: {n}")
        return 2
    try:
        if a.do_import:
            cnt = run_import(PRIVATE, a.candidate, None if a.auto_ru else json.load(open(a.mapping, encoding="utf-8")), a.auto_ru)
            print(" ".join(f"{k}={v}" for k, v in cnt.items()))
            return 0
        _clips, key = clip_index(PRIVATE, a.candidate)
        profs, name_col, tf, bl = profile(load_rows(path), set(key))
    except Exception:  # noqa: BLE001 - never echo error text
        print("import_labels: failed (details suppressed)")
        return 1
    print(f"columns={len(profs)} filename_col={name_col}")
    for i, p in profs.items():
        print(f"col {i}: " + " ".join(f"{k}={v}" for k, v in p.items() if not k.startswith("_")))
    print(f"timecode_format: {tf}")
    print(f"binary_label: {bl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
