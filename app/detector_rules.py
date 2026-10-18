"""Ruleset loading, strict validation and rule activation for the observation pipeline.

Integrity contract (review H2/H3/M1):
- only KNOWN rule types; unknown keys -> invalid_config;
- coverage gates have HARD floors in code (JSON can only tighten them);
- a rule is "validated" only with a complete tools/evaluate.py holdout record; a rule that
  claims validation with an incomplete/implausible record -> invalid_config (job failed);
- second_person_in_operator_zone can never drive the verdict.
Errors carry static codes only; messages never contain paths or values from the file.
"""
import datetime
import json
import math
import os

from .norms import NORMS

UNATTENDED = "operator_zone_unattended_with_counter_activity"
SECOND = "second_person_in_operator_zone"
KNOWN_RULES = frozenset({UNATTENDED, SECOND})
NEVER_VERDICT = frozenset({SECOND})  # presence of a second person never yields verdict=violation (F7)
DEFAULT_RULESET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules_default.json")

FLOOR_PERSON_RATIO = 0.10
FLOOR_ZONE_OCCUPANCY = 0.15
MIN_HOLDOUT_CLIPS = 5
MAX_HOLDOUT_CLIPS = 10000
EVAL_TOOL = "tools/evaluate.py"
MAX_EVENTS_CAP = 50  # public contract limit (app/worker.py MAX_VIOLATIONS)

# closed schema (unknown keys -> invalid_config)
TOP_KEYS = frozenset({"_status", "_verdict_semantics", "stage1_width", "stage2_fps", "stage2_margin_sec",
                      "stage2_max_total_sec", "deadline_reserve_sec", "motion_pixel_thresh", "motion_grid",
                      "smooth_samples_stage1", "smooth_samples_stage2", "min_person_sample_ratio", "zone",
                      "rules", "candidate_fraction_of_min_sec", "min_confidence_to_emit",
                      "min_gap_between_events_sec", "max_events"})
ZONE_KEYS = frozenset({"override", "heatmap_grid", "pad", "min_occupancy_to_analyse", "full_trust_occupancy",
                       "activity_ring_scale"})
_RULE_COMMON = {"enabled", "status", "validated", "holdout", "min_sec", "candidate_norms_for_human_review",
                "min_confidence_to_emit"}
RULE_KEYS = {UNATTENDED: frozenset(_RULE_COMMON | {"activity_motion_thresh", "client_presence_ratio"}),
             SECOND: frozenset(_RULE_COMMON | {"observation_only", "zone_pad", "max_pair_iou"})}


class RulesError(Exception):
    """Static, content-free ruleset error. code: invalid_config | ruleset_unreadable."""

    def __init__(self, msg: str, code: str = "invalid_config"):
        super().__init__(msg)
        self.code = code


def _num(v, lo, hi):
    return (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and lo <= v <= hi)


def holdout_ok(h) -> bool:
    """Complete, plausible holdout record: n_clips int in [5, 10000], precision and recall finite
    in (0, 1], evaluated_at a strict ISO date (whole string) not in the future, non-empty split_id,
    tool == "tools/evaluate.py".

    KNOWN, ACCEPTED LIMITATION: the record is a DECLARATION copied from tools/evaluate.py output
    and reviewed by a human; it is not cryptographically tied to an evaluation run, so this check
    is a plausibility gate, not proof."""
    if not isinstance(h, dict):
        return False
    n = h.get("n_clips")
    if not (isinstance(n, int) and not isinstance(n, bool) and MIN_HOLDOUT_CLIPS <= n <= MAX_HOLDOUT_CLIPS):
        return False
    if not all(_num(h.get(k), 0, 1) and h[k] > 0 for k in ("precision", "recall")):
        return False
    if h.get("tool") != EVAL_TOOL or not (isinstance(h.get("split_id"), str) and h["split_id"].strip()):
        return False
    d = h.get("evaluated_at")
    try:
        when = datetime.date.fromisoformat(d) if isinstance(d, str) else None
    except ValueError:
        return False
    return when is not None and when <= datetime.date.today()


def validate(rules) -> dict:
    if not isinstance(rules, dict) or not isinstance(rules.get("rules"), dict) or not isinstance(rules.get("zone"), dict):
        raise RulesError("ruleset structure invalid")
    if set(rules) - TOP_KEYS or set(rules["zone"]) - ZONE_KEYS:
        raise RulesError("ruleset has unknown keys")
    if set(rules["rules"]) != KNOWN_RULES:
        raise RulesError("ruleset has unknown or missing rule types")
    if not _num(rules.get("candidate_fraction_of_min_sec"), 0.0, 1.0) or rules["candidate_fraction_of_min_sec"] <= 0:
        raise RulesError("candidate_fraction_of_min_sec must be within (0, 1]")
    if not _num(rules.get("min_gap_between_events_sec"), 0.0, math.inf):
        raise RulesError("min_gap_between_events_sec must be >= 0")
    if not _num(rules.get("min_person_sample_ratio"), FLOOR_PERSON_RATIO, 1.0):
        raise RulesError("min_person_sample_ratio below hard floor or invalid")
    if not _num(rules["zone"].get("min_occupancy_to_analyse"), FLOOR_ZONE_OCCUPANCY, 1.0):
        raise RulesError("zone.min_occupancy_to_analyse below hard floor or invalid")
    if not _num(rules.get("min_confidence_to_emit"), 0.0, 1.0):
        raise RulesError("min_confidence_to_emit must be within [0, 1]")
    me = rules.get("max_events")
    if not (isinstance(me, int) and not isinstance(me, bool) and 1 <= me <= MAX_EVENTS_CAP):
        raise RulesError("max_events must be an integer within [1, 50]")
    for k, r in rules["rules"].items():
        if not isinstance(r, dict) or not isinstance(r.get("enabled", False), bool) or set(r) - RULE_KEYS[k]:
            raise RulesError("rule entry invalid or has unknown keys")
        if not _num(r.get("min_sec"), 0.0, math.inf) or r["min_sec"] <= 0:
            raise RulesError("rule min_sec must be > 0")
        cn = r.get("candidate_norms_for_human_review")
        if not (isinstance(cn, list) and all(isinstance(x, str) and x in NORMS for x in cn)):
            raise RulesError("rule candidate_norms_for_human_review must list known norm keys")
        if "min_confidence_to_emit" in r and not _num(r["min_confidence_to_emit"], 0.0, 1.0):
            raise RulesError("rule min_confidence_to_emit must be within [0, 1]")
        claims = r.get("status") == "validated" or r.get("validated") is True
        if claims and not (r.get("status") == "validated" and r.get("validated") is True and holdout_ok(r.get("holdout"))):
            raise RulesError("rule claims validation without a complete evaluate.py holdout record")
    return rules


def load(settings) -> dict:
    try:
        with open(getattr(settings, "ruleset", "") or DEFAULT_RULESET, encoding="utf-8") as f:
            rules = json.load(f)
    except (OSError, ValueError):
        raise RulesError("ruleset unreadable", code="ruleset_unreadable") from None
    return validate(rules)


def emit_threshold(rules, kind) -> float:
    """Per-rule threshold if given, else global; both bounded to [0, 1] by validate()."""
    return float(rules["rules"][kind].get("min_confidence_to_emit", rules["min_confidence_to_emit"]))


def states(rules, never=NEVER_VERDICT) -> dict:
    allow = os.environ.get("ALLOW_UNVALIDATED_RULES") == "1"
    out = {}
    for k, r in rules["rules"].items():
        validated = r.get("status") == "validated" and r.get("validated") is True and holdout_ok(r.get("holdout"))
        eligible = k in KNOWN_RULES and k not in never and not r.get("observation_only", False)
        enabled = r.get("enabled") is True
        out[k] = {"type": k, "status": "validated" if validated else "candidate", "enabled": enabled,
                  "validated": validated, "verdict_eligible": eligible,
                  "active": enabled and eligible and (validated or allow),
                  "holdout": r.get("holdout") if validated else None}
    return out
