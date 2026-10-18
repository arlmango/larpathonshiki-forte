# PROTOCOL STAGE 8b: sufficiency gate on the interval snapshot (pre-registration)

**Status: DRAFT for independent review.**
- On approval, the coordinator locks this file and the two code files (chmod 444) and records the lock digest.
- After that, ONE `--gate-only` run is allowed. Any later change is a dated amendment appended at the end.

Implementation:
- `tools/stage8_audit.py` computes the gate (counts only).
- `tools/stage8_selftest.py` is the synthetic self-test.

**Number of attempts: 1.**

## 0. Scope and disclosure (binding)
- This is a **sufficiency check, not an experiment.**
  - No model, score, feature cache, threshold fit or representation search is computed or opened.
  - The only output is counts.
- The train set (C02 train, 17 clips, 5 imported positives) has already been reused **4+ times**: stages 4, 5 and 7,
  plus the human audit.
  - Nothing here is inference.
  - Nothing here may be presented as 5 confirmed events.
- Holdout: CLOSED.
  - Train selection reads the shared imported-label and split files locally, and discards every non-train record.
  - Non-train records in the snapshot are also discarded and only counted.
  - Holdout videos, features and scores are never opened or computed.
- Production stays fail-closed (`failed/no_validated_rule`). Verdicts stay strictly `violation | no_violation`.
- Stage 5/7 GO criteria remain the only bars for any future GO. **This gate does not lower them.**
- Privacy: outputs carry aggregates only. No shas, paths, names, per-clip rows, interval times or label times.

## 1. Data
- **Labels.** Only the read-only snapshot `labels/snapshots/C02.intervals.v1.20261018T0333Z.jsonl`.
  - Its sha256 `31c0a7548a928fd2d6e45ac0b6cab8d56fef64cd3e5970266a60d6ab57a9ca1b` is pinned in code.
  - The sha is checked before parsing. A mismatch means refused.
  - The latest revision per `task_id` is used (the same rule as `annotate_intervals.read_log`).
  - Payload semantics follow `annotate_intervals.classify` and its schema:
    - per interval: `start`/`end`, `*_uncertain`, `*_unknown`, `visibility` ∈ {good, partial, poor} and `confidence` ∈ {sure, unsure};
    - per clip: `clip_flags`;
    - `same_clip_controls`;
    - control tasks.
- **Units.** Units come from `local_safe.select_split_units(LP, cid, "train", dry_run=True)`. No clip file is opened.
  - A positive-task record whose sha16 is not a train `reviewed_violation` unit is excluded and counted.
  - So is a control-task record whose sha16 is not a train `reviewed_no_violation` unit.

## 2. Denominators (units = clips; separate, never merged)
- **P: positives with numeric intervals.** Done positive tasks of class `intervals` that have at least one
  interval with both boundaries numeric. Several intervals in one clip are still **one** unit.
- **C: confirmed-confident control clips.** Distinct clips with at least one control that meets all of these:
  - human-confirmed comparable operation visible, and no violation visible (both checkboxes);
  - `visibility = good` and `confidence = sure`;
  - numeric boundaries.

  C has two sources, both counted:
  - **same-clip:** an entry in a done positive task of class `intervals` that counts toward P (at least one
    numeric interval) and does not overlap any event interval of that clip. Same-clip controls in other positive
    tasks (unknown-flagged, or with no numeric interval) are counted in `n_same_clip_controls_in_unknown_tasks`
    and never in C;
  - **external:** a control task classified `normal_confident` on a train negative.
- **U: unknown tasks.** Done positive tasks with a clip flag (`not_visible` / `unknown` / `event_unconfirmed`).
  - They are reported as their own denominator. **They are never negatives.**
- **Imported negatives are excluded.** Imported `reviewed_no_violation` status is not a verified normal.
  - An imported negative counts toward C only through a completed external control task.
- **Counted separately and never in C:**
  - controls with low confidence or partial/poor visibility;
  - controls that are unconfirmed or answered `not_comparable` / `not_visible` / `unknown`;
  - controls that overlap an event interval.

## 3. Screen thresholds (fixed before the run)
**GATE PASS iff P ≥ 4 AND C ≥ 3.**

The 4/3 thresholds are a **minimal operational screen**:
- They are not a validity certificate.
- They are not a guarantee of sufficiency or power.
- They are not a lowering of earlier bars.

Where the thresholds come from:
- **A chosen constraint, not a universal necessary condition.** Every leave-one-clip-out training fold must keep
  ≥ 3 positives (hence P ≥ 4) and ≥ 2 controls (hence C ≥ 3).
- **Examples for orientation only.** In a hypothetical unpaired design with *distinct, exchangeable* clips, an exact
  clip-label permutation can reach:
  - C(6,3) = 20 labelings: minimum p = 0.05;
  - C(7,3) = 35 labelings (4 + 3): minimum p ≈ 0.029.
  - These describe only those hypothetical designs and guarantee neither sufficiency nor power.
- **The current design is not three independent clip units.** The single confirmed control lies *inside* a
  positive clip.
  - For one independent within-clip pair, a one-sided sign-flip test has a minimum p of **1/2**.
  - No test, fit or threshold is meaningful at that size.
- **Same-clip controls are paired with their positive.** A design that reaches C ≥ 3 mainly through same-clip
  controls is weaker than the unpaired reference. So passing the screen is required by this protocol before any
  further step; never sufficient.

## 4. Outcome rule
- **Gate fails:** `STOP_INSUFFICIENT_DATA`, with the exact unmet numbers. Examples:
  - `positives_with_numeric_intervals 2 < 4`
  - `confirmed_confident_control_clips 1 < 3`
- **Gate passes:** `GATE_PASS_SEPARATE_PROTOCOL_REQUIRED`.
  - This protocol never trains, fits thresholds or searches representations.
  - Any such step needs a new pre-registration, a review and a user decision.
- There is no other outcome. There is no rescue, no relabelling of unknown tasks, and no use of imported negatives.

**Expectation, stated before the run.** Coordinator-recorded aggregates: 2 positives with numeric intervals,
1 same-clip confident control, 3 unknown, 0 external controls. This gives `STOP_INSUFFICIENT_DATA`, with both
thresholds unmet.

## 5. What is reported
Everything is reported on stdout and in `<LP>/train_probe/stage8_summary.json` (O_EXCL, mode 0400). Counts only:

**Positive tasks:**
- `n_positive_tasks`, `_shown`, `_media_error`, `_done`, `_not_done`.

**Intervals:**
- `n_with_numeric_intervals` (P);
- `n_intervals`, `n_intervals_numeric`;
- `n_boundary_uncertain`, `n_boundary_unknown`;
- `n_good_sure`;
- `interval_visibility` and `interval_confidence` (enum counts).

**Unknown tasks:**
- `n_unknown_tasks` (U);
- `unknown_task_flags` (flag counts).

**Same-clip controls:**
- `n_same_clip_control_entries`;
- `n_same_clip_controls_confirmed_confident` (clips);
- `n_same_clip_controls_low_or_unconfirmed`;
- `n_same_clip_controls_in_unknown_tasks`;
- `n_event_control_overlap`;
- `n_control_not_found_set`.

**External controls:**
- `n_external_control_tasks_generated`;
- `n_external_controls` (confident clips);
- `_low_confidence`, `_unclear`, `_not_done`.

**Exclusions and train units:**
- `n_records_excluded_not_train`;
- train unit counts.

**Gate:**
- the thresholds, C, pass, the unmet list, and `training_allowed_in_this_protocol = false`;
- the outcome.

**Integrity:**
- the lock digest and the sha256-16 of the three locked files;
- snapshot `sha256_verified` / `unchanged`;
- the 8 immutables before/after (`unchanged`/`CHANGED`, and the number matching their recorded sha16);
- the disclosure line.

## 6. Considered and excluded
**A frozen, label-free, interval-aware descriptive audit** was considered and excluded. It would have used:
- the stage-7 novelty ‖z̄_t‖ and a motion analogue;
- argmax-in-interval hits against a uniform-time and a position-prior baseline;
- one event-vs-control mean comparison.

Reason: **no new conclusion is reachable at n = 2.** A hit could not separate a violation from any scene change,
a miss would not show absence of signal, and the train set is already reused 4+ times.

## 7. Machinery and fail-closed
- `--print-lock` prints the lock digest: sha256-16 over this file, `stage8_audit.py` and `stage8_selftest.py`.
  It reads no data.
- `--gate-only --expect-lock <digest>` is refused, with exit 2, static message and no summary, when:
  - the digest differs;
  - the summary already exists (single attempt);
  - the snapshot sha differs or the snapshot is unreadable;
  - the split, labels or map are malformed.
- The 8 immutables (`annotate_intervals.IMMUTABLE`) and the snapshot are hashed before and after the run.
- Errors print only the exception class (`local_safe.run_guarded`).
- Budget: counts over one ~34 KB file, so seconds. Hard bound 5 min. No caches are read, so no RSS watchdog is
  needed.

## 8. Synthetic self-test (`python -m tools.stage8_selftest DIR`)
**Fixture.**
- `stage7_selftest.build`: 17 train + 4 holdout dummy clip files, with the holdout at chmod 000.
- A synthetic snapshot whose payloads pass the real validator `annotate_intervals.clean_payload`.
- A superseded revision, which must be ignored.

**Scenarios and checks.**
- `fail_real_like`:
  - setup: 2 numeric positives, 3 unknown, 1 same-clip confident control, 0 external, plus 1 record on a holdout
    sha;
  - expected: STOP, 2 unmet, exact counts, and the holdout record excluded and counted.
- `pass`:
  - setup: 4 numeric, 1 unknown, 1 same-clip + 2 external confident controls, 1 external low-confidence control;
  - expected: GATE_PASS.
- `fail_controls`:
  - setup: 4 numeric, an unknown-boundary-only positive, 2 external confident controls, 1 unclear, 1 same-clip
    low-confidence control, 1 overlapping control;
  - expected: STOP, with only the control threshold unmet.
- In every scenario:
  - an open() spy shows no holdout path, cache or clip file opened;
  - the holdout stays at mode 000;
  - no sha16, sentinel, fixture path or planted interval time appears in stdout, stderr or the summary;
  - the summary is mode 0400;
  - a second run, a wrong lock and a wrong snapshot sha each give exit 2 and no summary;
  - `--print-lock` opens no private file.
- `ctl_in_unknown`:
  - setup: 4 numeric positives, plus an unknown-flagged task carrying a good/sure same-clip control;
  - expected: C 0 and `n_same_clip_controls_in_unknown_tasks` 1, so STOP with only the control threshold unmet.
- Result: 5/5 PASS.
- Mutation checks (in memory, not part of the self-test):
  - a lowered control threshold is caught by `fail_controls`;
  - a planted time in the output is caught by the leak check in all 4 scenarios.
