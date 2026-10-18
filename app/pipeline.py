"""Two-stage observational analysis.

Stage 1: whole clip at SAMPLE_FPS (default 4) on 640 px frames: YOLOX person boxes + coarse
         motion grid.  The operator zone is estimated adaptively (densest person-box region)
         unless overridden (OPERATOR_ZONE env "x0,y0,x1,y1" normalised, or rules zone.override).
Stage 2: denser re-analysis (rules stage2_fps) only inside candidate windows, which confirms
         and re-times candidates.

Every emitted item is an OBSERVABLE pattern with a heuristic (uncalibrated) confidence; the
description separates what was observed from what remains unverified.  Legal norms are only
cited as candidates.  Detection quality is NOT measured.  Any failure, including missing
weights or an uninformative detector result, raises AnalysisError -> job `failed`,
never `no_violation`.
"""
import base64
import json
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import detector as det_mod
from . import detector_rules as rules_mod
from . import norms

UNATTENDED = "operator_zone_unattended_with_counter_activity"
SECOND = "second_person_in_operator_zone"


class AnalysisError(Exception):
    """Any analysis failure -> job `failed`. `.code` is from the closed set CODES."""
    CODES = frozenset({"model_missing", "model_load_failed", "model_invalid", "invalid_config",
                       "corrupt_video", "too_long", "too_few_frames", "decode_error", "time_budget",
                       "no_validated_rule", "too_few_detections", "zone_unreliable", "jpeg_failed"})

    def __init__(self, msg: str = "", code: str = "decode_error"):
        super().__init__(msg)
        self.code = code if code in self.CODES else "decode_error"


@dataclass
class Probe:
    fps: float
    frames: int
    duration: float
    width: int
    height: int


def probe(path: str, max_duration: int) -> Probe:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise AnalysisError("cannot open video", code="corrupt_video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
        if not ok or fps <= 0 or n <= 0 or w <= 0:
            raise AnalysisError("corrupt or empty video", code="corrupt_video")
        dur = n / fps
        if dur > max_duration:
            raise AnalysisError(f"video too long: {dur:.0f}s > {max_duration}s", code="too_long")
        return Probe(fps, n, dur, w, h)
    finally:
        cap.release()


DEFAULT_RULESET = rules_mod.DEFAULT_RULESET


def describe_model(settings=None) -> dict:
    """{name, license, weights_sha256_prefix, validated: False, providers, rules:[{type,status,...}]}."""
    path = getattr(settings, "onnx_model", "") if settings is not None else os.environ.get("ONNX_MODEL", "")
    d = det_mod.describe(path)
    d["validated"] = False
    try:
        d["rules"] = list(rule_states(rules_mod.load(settings)).values())
    except rules_mod.RulesError as e:
        d["rules"], d["rules_error"] = [], e.code  # static code, never a path or message
    d["allow_unvalidated_rules"] = os.environ.get("ALLOW_UNVALIDATED_RULES") == "1"
    return d


def _check(deadline):
    if deadline and time.time() > deadline:
        raise AnalysisError("time budget exceeded", code="time_budget")


def _get_detector(settings, injected):
    if injected is not None:
        return injected
    try:
        return det_mod.load(getattr(settings, "onnx_model", ""))
    except det_mod.DetectorError as e:
        raise AnalysisError(str(e), code=e.code) from e


def _detect(det, frame) -> np.ndarray:
    try:
        b = np.asarray(det.detect(frame), np.float32).reshape(-1, 5).copy()
    except det_mod.DetectorError as e:
        raise AnalysisError(str(e), code=e.code) from e
    h, w = frame.shape[:2]
    b[:, [0, 2]] /= w
    b[:, [1, 3]] /= h
    return b


def _resize_w(fr, width):
    return cv2.resize(fr, (width, max(1, int(round(width * fr.shape[0] / fr.shape[1]))))) \
        if fr.shape[1] != width else fr


# ----------------------------------------------------------------------------- stage 1
def _stage1(path, pr, det, rules, fps, deadline):
    cap = cv2.VideoCapture(path)
    step = max(1, int(round(pr.fps / fps)))
    gw, gh = rules["motion_grid"]
    samples, prev, idx = [], None, 0
    try:
        while cap.grab():
            if idx % step == 0:
                _check(deadline)
                ok, fr = cap.retrieve()
                if not ok:
                    raise AnalysisError("decode error", code="decode_error")
                small = _resize_w(fr, rules["stage1_width"])
                g = cv2.GaussianBlur(cv2.cvtColor(_resize_w(small, 320), cv2.COLOR_BGR2GRAY), (5, 5), 0)
                if prev is None:
                    grid = np.zeros((gh, gw), np.float32)
                else:
                    diff = (cv2.absdiff(g, prev) > rules["motion_pixel_thresh"]).astype(np.float32)
                    grid = cv2.resize(diff, (gw, gh), interpolation=cv2.INTER_AREA)
                prev = g
                samples.append({"t": idx / pr.fps, "boxes": _detect(det, small),
                                "grid": grid.astype(np.float16)})
            idx += 1
    finally:
        cap.release()
    if len(samples) < 2:
        raise AnalysisError("too few decodable frames", code="too_few_frames")
    return samples


# ----------------------------------------------------------------------------- zone
_centers, _inside, _pad, _iou = det_mod.centers, det_mod.inside, det_mod.pad, det_mod.iou
_smooth, _runs, _merge_intervals = det_mod.smooth, det_mod.runs, det_mod.merge_intervals
_estimate_zone, _cell_mask = det_mod.estimate_zone, det_mod.cell_mask


def _zone_override(rules):
    raw = os.environ.get("OPERATOR_ZONE", "").strip()
    z = [float(v) for v in raw.split(",")] if raw else rules["zone"].get("override")
    if z is None:
        return None
    if len(z) != 4 or not (0 <= z[0] < z[2] <= 1 and 0 <= z[1] < z[3] <= 1):
        raise AnalysisError("invalid operator zone override (need normalised x0,y0,x1,y1)", code="invalid_config")
    return [float(v) for v in z]


def _two_in_zone(b, z, max_iou):
    m = b[_inside(b, z)]
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            if _iou(m[i], m[j]) <= max_iou:
                return True, float(min(m[i, 4], m[j, 4]))
    return False, 0.0


# ----------------------------------------------------------------------------- signals
def _signals(samples, zone, zone2, ev_rules):
    occ = np.array([int(_inside(s["boxes"], zone).sum()) for s in samples])
    other = np.array([int((~_inside(s["boxes"], zone)).sum()) for s in samples])
    two = [_two_in_zone(s["boxes"], zone2, ev_rules[SECOND]["max_pair_iou"]) for s in samples]
    return occ, other, np.array([t[0] for t in two]), np.array([t[1] for t in two])


def _candidates(samples, zone, zone2, rules, k, dt, frac):
    """Observation candidates are searched for every pattern regardless of rule activation."""
    ev = rules["rules"]
    times = [s["t"] for s in samples]
    occ, _, two, _ = _signals(samples, zone, zone2, ev)
    out = [(UNATTENDED, a, b) for a, b in _runs(_smooth(occ == 0, k), times, dt)
           if b - a >= frac * ev[UNATTENDED]["min_sec"]]
    out += [(SECOND, a, b) for a, b in _runs(_smooth(two, k), times, dt)
            if b - a >= frac * ev[SECOND]["min_sec"]]
    return out


# ----------------------------------------------------------------------------- stage 2
def _stage2(path, pr, det, rules, cands, deadline, sec_per_frame):
    """Dense re-analysis only around candidate BOUNDARIES (where timing matters); the
    interior is already covered by stage 1.  Frame budget = min(configured cap, what the
    remaining deadline allows at the measured per-frame cost)."""
    m, fps2 = rules["stage2_margin_sec"], rules["stage2_fps"]
    edges = [(max(0.0, t - m), min(pr.duration, t + m)) for _, a, b in cands for t in (a, b)]
    windows = _merge_intervals(edges)
    budget = rules["stage2_max_total_sec"]
    if deadline:
        spare = deadline - time.time() - rules["deadline_reserve_sec"]
        budget = min(budget, max(0.0, spare) / max(sec_per_frame, 1e-3) / fps2)
    done, samples, truncated = [], [], False
    step = max(1, int(round(pr.fps / fps2)))
    cap = cv2.VideoCapture(path)
    try:
        for a, b in windows:
            if budget < b - a:
                truncated = True
                continue
            budget -= b - a
            f0, f1 = int(a * pr.fps), int(b * pr.fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            got = 0
            for f in range(f0, f1 + 1):
                if not cap.grab():
                    break
                if (f - f0) % step:
                    continue
                _check(deadline)
                ok, fr = cap.retrieve()
                if not ok:
                    break
                samples.append({"t": f / pr.fps, "boxes": _detect(det, fr)})
                got += 1
            if got == 0:
                raise AnalysisError("stage-2 decode failed", code="decode_error")
            done.append((a, b))
    finally:
        cap.release()
    return samples, done, truncated


def _combine(s1, s2, done):
    """Stage-2 samples inside refined windows + stage-1 samples elsewhere, time-ordered."""
    keep = [s for s in s1 if not any(w0 <= s["t"] <= w1 for w0, w1 in done)]
    return sorted(keep + s2, key=lambda s: s["t"])


# ----------------------------------------------------------------------------- events
def _ring_motion(samples, zone, rules):
    gw, gh = rules["motion_grid"]
    zm = _cell_mask(zone, gw, gh)
    rm = _cell_mask(_pad(zone, (rules["zone"]["activity_ring_scale"] - 1) / 2), gw, gh) & ~zm
    if not rm.any():
        return np.zeros(len(samples))
    return np.array([float(s["grid"][rm].mean()) for s in samples])


def _confirm(kind, a, b, s1, s2, refined, zone, zone2, ring, occ1, rules, dt):
    ev, frac = rules["rules"], 1.0
    sub = [s for s in s2 if a - rules["stage2_margin_sec"] <= s["t"] <= b + rules["stage2_margin_sec"]]
    if not sub:
        return []
    times = [s["t"] for s in sub]
    occ, other, two, twosc = _signals(sub, zone, zone2, ev)
    k = rules["smooth_samples_stage2"] if refined else rules["smooth_samples_stage1"]
    mask = _smooth(occ == 0, k) if kind == UNATTENDED else _smooth(two, k)
    out = []
    for t0, t1 in _runs(mask, times, dt):
        if t1 < a or t0 > b or t1 - t0 < frac * ev[kind]["min_sec"]:
            continue
        sel = np.array([t0 <= t < t1 for t in times])
        dur = t1 - t0
        dur_f = float(np.clip(dur / (2 * ev[kind]["min_sec"]), 0.5, 1.0))
        ref_f = 1.0 if refined else 0.8
        obs = {"start_sec": round(t0, 2), "end_sec": round(t1, 2), "duration_sec": round(dur, 1),
               "samples": int(sel.sum()), "stage2_refined": refined}
        if kind == UNATTENDED:
            s1sel = np.array([t0 <= s["t"] < t1 for s in s1])
            act = float(ring[s1sel].mean()) if s1sel.any() else 0.0
            cl = float((other[sel] > 0).mean())
            outside = ~s1sel
            zocc = float((occ1[outside] > 0).mean()) if outside.any() else 0.0
            r = ev[UNATTENDED]
            if act < r["activity_motion_thresh"] and cl < r["client_presence_ratio"]:
                out.append({"type": kind, "suppressed": "no counter activity observed", **obs})
                continue
            act_f = max(min(1.0, act / (2 * r["activity_motion_thresh"])), cl)
            zone_f = min(1.0, zocc / rules["zone"]["full_trust_occupancy"])
            cons = float((occ[sel] == 0).mean())
            conf = zone_f * cons * dur_f * (0.5 + 0.5 * act_f) * ref_f
            obs.update(zone_occupancy_elsewhere=round(zocc, 3), activity_motion=round(act, 4),
                       other_person_ratio=round(cl, 3), no_person_in_zone_ratio=round(cons, 3))
        else:
            cons = float(two[sel].mean())
            sc = float(twosc[sel & two].mean()) if (sel & two).any() else 0.0
            conf = cons * dur_f * min(1.0, sc / 0.6) * ref_f
            obs.update(two_persons_ratio=round(cons, 3), second_box_score=round(sc, 3))
        out.append({"type": kind, "confidence": round(float(conf), 3), **obs})
    return out


# ----------------------------------------------------------------------------- rule activation
NEVER_VERDICT = set(rules_mod.NEVER_VERDICT)  # live guard: second person never drives a verdict (F7)


def rule_states(rules) -> dict:
    return rules_mod.states(rules, never=NEVER_VERDICT)


def _merge_events(evs, gap):
    out = []
    for e in sorted(evs, key=lambda e: (e["type"], e["start_sec"])):
        p = out[-1] if out else None
        if p and p["type"] == e["type"] and e["start_sec"] - p["end_sec"] < gap:
            p["end_sec"] = max(p["end_sec"], e["end_sec"])
            p["duration_sec"] = round(p["end_sec"] - p["start_sec"], 1)
            p["confidence"] = max(p["confidence"], e["confidence"])
        else:
            out.append(dict(e))
    return out


def _evidence(path, pr, e, zone, s2, deadline):
    t0, t1 = e["start_sec"], e["end_sec"]
    d = min(1.0, (t1 - t0) / 4)
    cap = cv2.VideoCapture(path)
    out = []
    try:
        for t in (t0 + d, (t0 + t1) / 2, max(t0, t1 - d)):
            _check(deadline)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * pr.fps))
            ok, fr = cap.read()
            if not ok:
                continue
            fr = _resize_w(fr, 640)
            h, w = fr.shape[:2]
            cv2.rectangle(fr, (int(zone[0] * w), int(zone[1] * h)), (int(zone[2] * w), int(zone[3] * h)),
                          (0, 220, 255), 2)
            near = min(s2, key=lambda s: abs(s["t"] - t)) if s2 else None
            for b in (near["boxes"] if near else []):
                cv2.rectangle(fr, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), (0, 200, 0), 2)
            ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok and not any(abs(o["timestamp_sec"] - t) < 1e-3 for o in out):
                out.append({"timestamp_sec": round(t, 2), "image_base64": base64.b64encode(buf.tobytes()).decode()})
    finally:
        cap.release()
    if not out:
        raise AnalysisError("evidence frame extraction failed", code="jpeg_failed")
    return out[:3]


# ----------------------------------------------------------------------------- entry
def _observe(path, pr, rules, settings, detector, deadline):
    """Observation core shared by analyze() and export_observations(); no verdict here."""
    T = time.time()
    det = _get_detector(settings, detector)
    fps1 = float(getattr(settings, "sample_fps", 4) or 4)
    s1 = _stage1(path, pr, det, rules, fps1, deadline)
    t1 = time.time() - T
    person_ratio = float(np.mean([len(s["boxes"]) > 0 for s in s1]))
    floor_p = max(rules["min_person_sample_ratio"], rules_mod.FLOOR_PERSON_RATIO)  # hard floor (H3)
    if person_ratio < floor_p:
        raise AnalysisError(f"detector found persons in {person_ratio:.0%} of sampled frames (< "
                            f"{floor_p:.0%}); result would be uninformative",
                            code="too_few_detections")
    zone, zsrc, support = _zone_override(rules), "задана вручную", None
    if zone is None:
        zone, support = _estimate_zone(s1, rules)
        zsrc = "оценена автоматически"
        if zone is None:
            raise AnalysisError("operator zone could not be estimated", code="zone_unreliable")
    zone2 = _pad(zone, rules["rules"][SECOND]["zone_pad"])
    occ1 = np.array([int(_inside(s["boxes"], zone).sum()) for s in s1])
    occupancy = float((occ1 > 0).mean())
    floor_o = max(rules["zone"]["min_occupancy_to_analyse"], rules_mod.FLOOR_ZONE_OCCUPANCY)
    if occupancy < floor_o:
        raise AnalysisError(f"operator zone occupied in {occupancy:.0%} of samples (< "
                            f"{floor_o:.0%}); zone unreliable",
                            code="zone_unreliable")
    dt1 = 1.0 / fps1
    cands = _candidates(s1, zone, zone2, rules, rules["smooth_samples_stage1"], dt1,
                        rules["candidate_fraction_of_min_sec"])
    s2, done, truncated = (_stage2(path, pr, det, rules, cands, deadline, t1 / len(s1))
                           if cands else ([], [], False))
    src, ring, found = _combine(s1, s2, done), _ring_motion(s1, zone, rules), []
    for kind, a, b in cands:
        refined = all(any(w0 <= t <= w1 for w0, w1 in done) for t in (a, b))
        found += _confirm(kind, a, b, s1, src, refined, zone, zone2, ring, occ1, rules, dt1)
    scored = _merge_events([e for e in found if "confidence" in e], rules["min_gap_between_events_sec"])
    meta = {"duration_sec": round(pr.duration, 1), "samples_stage1": len(s1), "samples_stage2": len(s2),
            "stage2_windows": [[round(a, 1), round(b, 1)] for a, b in done], "stage2_truncated": truncated,
            "person_sample_ratio": round(person_ratio, 3), "operator_zone": [round(v, 3) for v in zone],
            "operator_zone_source": zsrc, "operator_zone_support": support,
            "operator_zone_occupancy": round(occupancy, 3), "candidates": len(cands),
            "suppressed_no_activity": sum("suppressed" in e for e in found),
            "timing_sec": {"stage1": round(t1, 2), "observe": round(time.time() - T, 2)},
            "model": dict(getattr(det, "info", {"name": "injected"}), validated=False),
            "quality": "detection quality unmeasured; thresholds unvalidated"}
    return {"scored": sorted(scored, key=lambda e: e["start_sec"]), "zone": zone, "zsrc": zsrc,
            "frames": s2 + s1, "meta": meta}


def _load_rules(settings):
    try:
        return rules_mod.load(settings)
    except rules_mod.RulesError as e:
        raise AnalysisError(str(e), code="invalid_config") from None


def analyze(path: str, settings, detector=None, deadline: float | None = None,
            observations_dir: str | None = None) -> dict:
    """API analysis. observations_dir (local tools only, never set by the API) additionally
    writes the clip's observations there, even when no rule is active.  Native (OpenCV/FFmpeg)
    stderr, which can contain file paths, is silenced for the duration (review M5)."""
    with det_mod.quiet_native_stderr():
        return _analyze(path, settings, detector, deadline, observations_dir)


def _analyze(path, settings, detector, deadline, observations_dir):
    rules = _load_rules(settings)
    pr = probe(path, settings.max_duration_sec)
    states = rule_states(rules)
    active = [k for k, st in states.items() if st["active"]]
    no_rule = AnalysisError("no validated rule active: coverage error", code="no_validated_rule")
    if not active and not observations_dir:
        raise no_rule
    ob = _observe(path, pr, rules, settings, detector, deadline)
    if observations_dir:
        _write_observations(path, ob, observations_dir)
    if not active:
        raise no_rule
    scored, zone = ob["scored"], ob["zone"]
    # (a) OBSERVATIONS: every pattern, for local human review; never drives the verdict by itself.
    observations = [{"type": e["type"], "start_sec": e["start_sec"], "end_sec": e["end_sec"],
                     "confidence": e["confidence"], "confidence_kind": "heuristic_uncalibrated",
                     "rule_status": states[e["type"]]["status"], "drives_verdict": e["type"] in active,
                     "observed": {k: v for k, v in e.items() if k not in ("type", "confidence")},
                     "candidate_norms_for_human_review":
                         [norms.NORMS[k][0] for k in rules["rules"][e["type"]]["candidate_norms_for_human_review"]]}
                    for e in scored]
    # (b) ACTIVE verdict rules only.
    emit = [e for e in scored if e["type"] in active and e["confidence"] >= rules_mod.emit_threshold(rules, e["type"])]
    emit = sorted(emit, key=lambda e: -e["confidence"])[: max(1, rules["max_events"])]
    violations = []
    for e in sorted(emit, key=lambda e: e["start_sec"]):
        keys = rules["rules"][e["type"]]["candidate_norms_for_human_review"]
        st = states[e["type"]]
        violations.append({
            "type": e["type"], "timecode_sec": e["start_sec"], "end_sec": e["end_sec"],
            "confidence": e["confidence"], "confidence_kind": "heuristic_uncalibrated",
            "rule_status": st["status"],
            "observed": {k: v for k, v in e.items() if k not in ("type", "confidence")},
            "candidate_norms_for_human_review": [norms.NORMS[k][0] for k in keys],
            "description": norms.describe_observation(e, ob["zsrc"], st) + norms.citations(keys),
            "evidence_frames": _evidence(path, pr, e, zone, ob["frames"], deadline),
        })
    unvalidated = [k for k in active if not states[k]["validated"]]
    meta = dict(ob["meta"], below_threshold=len(scored) - len(emit), active_rules=active,
                rules=list(states.values()),
                verdict_basis=("UNVALIDATED rules under demo-only opt-in ALLOW_UNVALIDATED_RULES=1: "
                               + ", ".join(unvalidated)) if unvalidated else "validated rules")
    return {"verdict": "violation" if violations else "no_violation", "violations": violations,
            "observations": observations, "meta": meta}


LOCAL_PRIVATE = os.environ.get("LOCAL_PRIVATE_DIR") or os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "local_private")


def _write_observations(path, ob, out_dir):
    rows = [{"type": e["type"], "start_sec": e["start_sec"], "end_sec": e["end_sec"],
             "confidence": e["confidence"]} for e in ob["scored"]]
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, det_mod.sha256_file(path)[:16] + ".json")
    with open(dst + ".tmp", "w") as f:
        json.dump(rows, f)
    os.replace(dst + ".tmp", dst)
    return dst


def export_observations(path: str, out_dir: str | None = None, settings=None, detector=None,
                        deadline: float | None = None) -> str:
    """LOCAL-ONLY: write [{type, start_sec, end_sec, confidence}] to <out_dir>/<sha16>.json
    (default <project>/local_private/observations, or $LOCAL_PRIVATE_DIR/observations) for the
    human accept/reject review. Ignores rule activation; not reachable from the API; prints nothing."""
    from . import config
    settings = settings or config.get()
    rules = _load_rules(settings)
    with det_mod.quiet_native_stderr():
        pr = probe(path, settings.max_duration_sec)
        ob = _observe(path, pr, rules, settings, detector, deadline)
    return _write_observations(path, ob, out_dir or os.path.join(LOCAL_PRIVATE, "observations"))
